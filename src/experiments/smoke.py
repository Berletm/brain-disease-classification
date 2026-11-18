import csv
import hashlib
import json
from pathlib import Path
import time

import mlflow
import numpy as np
from omegaconf import OmegaConf
import torch

from src.data.projections import ProjectionDataset
from src.experiments.nested import project, projection_layout
from src.experiments.training import fit


def select_training_records(records, scans_per_class):
    selected = []

    for label in range(6):
        candidates = [row for row in records if int(row['class']) == label]
        candidates.sort(key=lambda row: (not bool(row.get('patient_id')), row['sample_id']))

        if len(candidates) < scans_per_class:
            raise ValueError('Not enough scans for the engineering smoke test')

        selected.extend(candidates[:scans_per_class])

    return selected


def run_smoke(cfg, output):
    planes = [plane for plane, _, _ in projection_layout(cfg)]
    plane_count = len(planes)

    if not cfg.dataset.root or not cfg.data.manifest:
        raise ValueError('Set dataset.root and data.manifest')

    torch.set_num_threads(cfg.smoke.threads)
    root = Path(cfg.dataset.root).resolve()

    with Path(cfg.data.manifest).open() as stream:
        records = select_training_records(
            list(csv.DictReader(stream)),
            cfg.smoke.scans_per_class,
        )

    sources = {
        name: source.label
        for name, source in cfg.dataset.sources.items()
        if source.enabled
    }

    for row in records:
        path = (root / row['relative_path']).resolve()

        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Invalid MRI path in the smoke-test manifest')

        if (
            row['source_folder'] not in sources
            or int(row['class']) != sources[row['source_folder']]
        ):
            raise ValueError('Manifest labels disagree with source configuration')

        checksum = hashlib.sha256()

        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                checksum.update(block)

        if checksum.hexdigest() != row['sha256']:
            raise ValueError('MRI changed after manifest creation')

    (output / 'smoke_samples.json').write_text(json.dumps(records, indent=2))
    (output / 'smoke_protocol.json').write_text(json.dumps({
        'purpose': 'Engineering check of preprocessing, optimization and artifact logging',
        'training_scans': len(records),
        'held_out_scans': 0,
        'unresolved_patient_ids': sum(not row.get('patient_id') for row in records),
        'performance_claims': False,
        'projection_planes': planes,
        'limitation': 'No independent evaluation; no clinical or generalization metrics',
    }, indent=2))

    started = time.perf_counter()
    indices = np.arange(len(records))

    train_x, _ = project(indices, [], records, root, cfg, output / 'preprocessing')
    labels = np.array([int(row['class']) for row in records])
    dataset = ProjectionDataset(
        train_x,
        labels,
        training=cfg.preprocessing.augment,
        plane_count=plane_count,
    )

    history = []

    if cfg.get('device', 'cpu').startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()

    params = OmegaConf.to_container(cfg.experiment.params, resolve=True)
    model, _, _ = fit(
        dataset,
        None,
        params,
        cfg,
        cfg.cv.epochs,
        cfg.tuning.learning_rates[0],
        6,
        training_log=history,
    )

    if not history or not all(np.isfinite(row['loss']) for row in history):
        raise ValueError('Smoke training did not produce finite losses')

    if not any(row['head_gradient_norm'] > 0 for row in history):
        raise ValueError('No classification-head gradient during smoke training')

    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError('Non-finite model parameters after training')

    torch.save(model.state_dict(), output / 'classifier.pt')
    (output / 'training_history.json').write_text(json.dumps(history, indent=2))

    metrics = {
        'training_scans': len(records),
        'optimization_steps': len(history),
        'mean_training_loss': float(np.mean([row['loss'] for row in history])),
        'duration_seconds': time.perf_counter() - started,
    }
    if cfg.get('device', 'cpu').startswith('cuda'):
        metrics['peak_cuda_allocated_mib'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        metrics['peak_cuda_reserved_mib'] = torch.cuda.max_memory_reserved() / (1024 ** 2)

    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    for step, row in enumerate(history):
        mlflow.log_metric('training_loss', row['loss'], step=step)
        mlflow.log_metric('head_gradient_norm', row['head_gradient_norm'], step=step)

    mlflow.log_artifact(str(output / 'training_history.json'))
    mlflow.log_artifact(str(output / 'smoke_protocol.json'))
    print(json.dumps(metrics))
