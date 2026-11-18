import csv
import gc
import hashlib
import inspect
import json
import shutil
from pathlib import Path

import mlflow
import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf

from src.evaluation.protocol import (
    nested_group_folds,
    oof_metrics,
    patient_bootstrap_interval,
)
from src.preprocessing.fold import FoldMPCA
from src.preprocessing.cache import (
    cached_projections,
    implementation_fingerprint,
    source_identities,
)
from src.data.projections import ProjectionDataset
from src.experiments.training import fit, predict


def read_manifest(path, root):
    with Path(path).open() as stream:
        records = list(csv.DictReader(stream))

    if not records or any(not row.get('patient_id') for row in records):
        raise ValueError('Manifest needs explicit patient_id for every scan; do not substitute scan IDs')

    ids = [row['sample_id'] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate sample IDs in manifest')

    for row in records:
        path = (root / row['relative_path']).resolve()

        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Invalid or unavailable MRI path')

        if row.get('sha256'):
            digest = hashlib.sha256()

            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)

            if digest.hexdigest() != row['sha256']:
                raise ValueError('MRI changed since manifest creation; rebuild/review the manifest')

    # Exact duplicates with distinct patient IDs must not cross folds.
    hashes = {}
    for row in records:
        checksum = row.get('sha256')

        if checksum and checksum in hashes and hashes[checksum] != row['patient_id']:
            raise ValueError('Identical files assigned to different patients')

        if checksum:
            hashes[checksum] = row['patient_id']

    return records


def projection_layout(cfg):
    planes = {
        'axial': (0, (1, 2, 0)),
        'frontal': (1, (0, 2, 1)),
        'sagital': (2, (0, 1, 2)),
    }
    experiment = cfg.get('experiment', {})

    if experiment.get('model') == 'SingleBranch':
        if experiment.get('dataset') != 'mpca':
            raise ValueError('SingleBranch raw runner currently supports MPCA projections only')

        selected = [experiment.get('plane')]

        if selected[0] not in planes:
            raise ValueError('SingleBranch needs an explicit axial, frontal or sagital plane')
    else:
        selected = list(planes)

    return [(name, *planes[name]) for name in selected]


def project(train, other, records, root, cfg, directory):
    training_ids = [records[index]['sample_id'] for index in train]
    held_out_ids = [records[index]['sample_id'] for index in other]

    if set(training_ids) & set(held_out_ids):
        raise ValueError('Training and held-out projection IDs overlap')

    def volumes(indices):
        result = []

        for index in indices:
            image = nib.load(root / records[index]['relative_path'])

            if nib.aff2axcodes(image.affine) != ('L', 'A', 'S'):
                raise ValueError('All source MRI must have validated LAS orientation')

            result.append(image.get_fdata())

        return result

    settings = cfg.preprocessing.get('cache', {})
    use_cache = settings.get('enabled', False)
    training = held_out = None
    outputs = [], []
    cache_records = []

    if use_cache:
        source = inspect.getsource(project) + inspect.getsource(projection_layout)
        implementation = implementation_fingerprint(source)
        identities = {
            'train': source_identities(train, records, root),
            'held_out': source_identities(other, records, root),
        }

    for plane, compressed_axis, transpose in projection_layout(cfg):
        if 'ranks' in cfg.preprocessing:
            ranks = list(cfg.preprocessing.ranks[plane])
        else:
            ranks = list(cfg.preprocessing.shape)
            ranks[compressed_axis] = 3

        def build(pipeline_directory):
            nonlocal training, held_out

            if training is None:
                training = volumes(train)
                held_out = volumes(other)

            pipeline = FoldMPCA(
                cfg.preprocessing.shape,
                ranks,
                plane,
                cfg.preprocessing.max_iter,
            )
            pipeline.fit(training, training_ids)
            pipeline.save(pipeline_directory)

            if use_cache and settings.get('format', 'png') == 'png':
                def images(values):
                    for volume in values:
                        projection = pipeline.transform([volume])[0]
                        yield np.transpose(projection, transpose)

                return images(training), images(held_out)

            train_projection = np.transpose(
                pipeline.transform(training),
                (0,) + tuple(axis + 1 for axis in transpose),
            )

            if held_out:
                held_out_projection = np.transpose(
                    pipeline.transform(held_out),
                    (0,) + tuple(axis + 1 for axis in transpose),
                )
            else:
                held_out_projection = np.empty(
                    (0, *train_projection.shape[1:]), dtype=train_projection.dtype,
                )

            return train_projection, held_out_projection

        if use_cache:
            signature = {
                'format_version': 2,
                'output_format': settings.get('format', 'png'),
                **identities,
                'shape': list(cfg.preprocessing.shape),
                'ranks': ranks,
                'max_iter': int(cfg.preprocessing.max_iter),
                'plane': plane,
                'transpose': list(transpose),
                'implementation': implementation,
            }
            arrays, entry, key, hit = cached_projections(
                settings.directory,
                signature,
                build,
                mmap=settings.get('mmap', True),
                require_hit=settings.get('require_hit', False),
            )
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copytree(entry / 'pipeline', directory / plane)
            cache_records.append({
                'plane': plane,
                'key': key,
                'hit': hit,
                'entry': str(entry),
                'format': signature['output_format'],
                'projections': {
                    partition: str(
                        entry / partition
                        if signature['output_format'] == 'png'
                        else entry / f'{partition}.npy'
                    )
                    for partition in ['train', 'held_out']
                },
                'signature': signature,
            })
        else:
            arrays = build(directory / plane)

        outputs[0].append(arrays[0])

        if len(other):
            outputs[1].append(arrays[1])

    if use_cache:
        with (directory / 'cache.json').open('x') as stream:
            json.dump({'planes': cache_records}, stream, indent=2)

    return outputs


def nested_protocol(cfg):
    if cfg.experiment.model not in {
        'MultiBranchAttention', 'MultiBranchMean', 'MultiBranchConcat', 'SingleBranch',
    }:
        raise ValueError('Unsupported model in the raw nested runner')

    planes = [plane for plane, _, _ in projection_layout(cfg)]

    if not cfg.dataset.root or not cfg.data.manifest:
        raise ValueError('Set dataset.root (or MRI_DATASET_ROOT) and data.manifest')

    root = Path(cfg.dataset.root).resolve()
    records = read_manifest(cfg.data.manifest, root)

    if 'sources' in cfg.dataset:
        enabled = {
            name: source.label
            for name, source in cfg.dataset.sources.items()
            if source.enabled
        }

        for row in records:
            if (
                row.get('source_folder') not in enabled
                or int(row['class']) != enabled[row['source_folder']]
            ):
                raise ValueError('Manifest source/class does not match the dataset config')

    labels = np.array([int(row['class']) for row in records])
    groups = np.array([row['patient_id'] for row in records])

    if set(labels) != set(range(6)):
        raise ValueError('Expected the six documented classes')

    folds = nested_group_folds(labels, groups, cfg.cv.folds, cfg.cv.inner_folds, cfg.seed)
    return root, records, labels, folds, planes


def split_record(records, fold):
    return {
        'train': [records[index]['sample_id'] for index in fold['train']],
        'test': [records[index]['sample_id'] for index in fold['test']],
        'inner': [
            {
                'train': [records[index]['sample_id'] for index in train],
                'validation': [records[index]['sample_id'] for index in valid],
            }
            for train, valid in fold['inner']
        ],
    }


def run_prepare(cfg, output):
    if not cfg.preprocessing.get('cache', {}).get('enabled', False):
        raise ValueError('Preparation requires preprocessing.cache.enabled=true')

    root, records, _, folds, planes = nested_protocol(cfg)
    cache_references = []

    for fold_index, fold in enumerate(folds):
        fold_dir = output / f'fold_{fold_index}'
        fold_dir.mkdir(exist_ok=False)
        (fold_dir / 'split.json').write_text(json.dumps(split_record(records, fold), indent=2))
        partitions = [
            (f'inner_{index}', train, valid)
            for index, (train, valid) in enumerate(fold['inner'])
        ]
        partitions.append(('outer', fold['train'], fold['test']))

        for name, train, held_out in partitions:
            directory = fold_dir / name
            projections = project(train, held_out, records, root, cfg, directory)
            del projections
            gc.collect()
            cache_references.append(str(directory / 'cache.json'))

    entries = [
        plane
        for path in cache_references
        for plane in json.loads(Path(path).read_text())['planes']
    ]
    protocol = {
        'purpose': 'Fold-specific train-only preprocessing without classifier training',
        'performance_claims': False,
        'examinations': len(records),
        'projection_planes': planes,
        'cache_references': cache_references,
        'limitation': 'Initial preparation retains the existing in-memory raw MRI and MPCA implementation',
    }
    (output / 'prepare_protocol.json').write_text(json.dumps(protocol, indent=2))
    (output / 'metrics.json').write_text(json.dumps({
        'prepared_splits': len(cache_references),
        'projection_entries': len(entries),
        'cache_hits': sum(entry['hit'] for entry in entries),
        'cache_misses': sum(not entry['hit'] for entry in entries),
    }, indent=2))
    mlflow.log_artifact(str(output / 'prepare_protocol.json'))


def run_nested(cfg, output):
    root, records, labels, folds, planes = nested_protocol(cfg)
    plane_count = len(planes)
    params = OmegaConf.to_container(cfg.experiment.params, resolve=True)
    predictions = []

    for fold_index, fold in enumerate(folds):
        fold_dir = output / f'fold_{fold_index}'
        fold_dir.mkdir(exist_ok=False)

        split = split_record(records, fold)

        learning_rates = list(cfg.tuning.learning_rates) if 'tuning' in cfg else [cfg.lr]
        if not learning_rates or any(lr <= 0 for lr in learning_rates):
            raise ValueError('Expected positive learning-rate candidates')

        scores = {lr: [] for lr in learning_rates}
        selected_epochs = {lr: [] for lr in learning_rates}

        for inner_index, (train, valid) in enumerate(fold['inner']):
            train_x, valid_x = project(
                train,
                valid,
                records,
                root,
                cfg,
                fold_dir / f'inner_{inner_index}',
            )

            train_ds = ProjectionDataset(train_x, labels[train], cfg.preprocessing.augment, plane_count=plane_count)
            valid_ds = ProjectionDataset(valid_x, labels[valid], plane_count=plane_count)

            for lr in learning_rates:
                model, score, epoch = fit(train_ds, valid_ds, params, cfg, cfg.cv.epochs, lr, 6)

                scores[lr].append(score)
                selected_epochs[lr].append(epoch)

                del model

            del train_x, valid_x, train_ds, valid_ds
            gc.collect()

        selected_lr = max(learning_rates, key=lambda lr: np.mean(scores[lr]))
        epochs = max(1, int(np.median(selected_epochs[selected_lr])))

        (fold_dir / 'split.json').write_text(json.dumps(split, indent=2))

        train_x, test_x = project(
            fold['train'],
            fold['test'],
            records,
            root,
            cfg,
            fold_dir / 'outer',
        )

        train_ds = ProjectionDataset(
            train_x, labels[fold['train']], cfg.preprocessing.augment, plane_count=plane_count,
        )
        test_ds = ProjectionDataset(test_x, labels[fold['test']], plane_count=plane_count)

        # Outer refit uses a fixed epoch budget selected inside the training partition.
        model, _, _ = fit(train_ds, None, params, cfg, epochs, selected_lr, 6)
        predicted, probabilities = predict(model, test_ds, cfg)

        torch.save(model.state_dict(), fold_dir / 'classifier.pt')

        mlflow.log_metric(
            'outer_macro_f1',
            oof_metrics(test_ds.labels, predicted, list(range(6)))['macro_f1'],
            step=fold_index,
        )

        (fold_dir / 'selection.json').write_text(
            json.dumps({
                'lr': selected_lr,
                'inner_scores': scores,
                'inner_epochs': selected_epochs,
                'refit_epochs': epochs,
            })
        )

        for index, prediction, probability in zip(fold['test'], predicted, probabilities):
            predictions.append({
                'sample_id': records[index]['sample_id'],
                'fold': fold_index,
                'truth': int(labels[index]),
                'prediction': int(prediction),
                'probabilities': probability.tolist(),
                'patient_id': records[index]['patient_id'],
                'source': records[index].get('source', ''),
                'source_folder': records[index].get('source_folder', ''),
                'seed': cfg.seed,
                'model': cfg.experiment.model,
                'projection_planes': planes,
            })

        (output / 'oof_partial.json').write_text(json.dumps(predictions, indent=2))

        del model, train_x, test_x, train_ds, test_ds
        gc.collect()

    if (
        len(predictions) != len(records)
        or len({row['sample_id'] for row in predictions}) != len(records)
    ):
        raise ValueError('Incomplete or duplicate OOF predictions')

    (output / 'oof.json').write_text(json.dumps(predictions, indent=2))

    metrics = oof_metrics(
        [r['truth'] for r in predictions],
        [r['prediction'] for r in predictions],
        list(range(6)),
    )

    if 'uncertainty' in cfg:
        metrics['macro_f1_interval'] = patient_bootstrap_interval(
            [r['truth'] for r in predictions],
            [r['prediction'] for r in predictions],
            [r['patient_id'] for r in predictions],
            list(range(6)),
            cfg.uncertainty.bootstrap_samples,
            cfg.seed,
        )

    metrics['controls_by_folder'] = {}
    for folder in sorted({r['source_folder'] for r in predictions if r['truth'] == 0}):
        controls = [
            row for row in predictions
            if row['truth'] == 0 and row['source_folder'] == folder
        ]

        metrics['controls_by_folder'][folder] = {
            'scans': len(controls),
            'patients': len({r['patient_id'] for r in controls}),
            'correct': sum(r['prediction'] == 0 for r in controls),
        }

    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    mlflow.log_metrics({key: metrics[key] for key in ['accuracy', 'macro_f1']})
    mlflow.log_artifact(str(output / 'metrics.json'))
    mlflow.log_artifact(str(output / 'oof.json'))
