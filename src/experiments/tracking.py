from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def flatten(value, prefix=''):
    result = {}

    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key

        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = str(item)

    return result


@contextmanager
def tracked_run(cfg, project=None):
    project = Path(project) if project is not None else Path(HydraConfig.get().runtime.cwd)
    output = Path(HydraConfig.get().runtime.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = output / 'config.yaml'
    config.write_text(OmegaConf.to_yaml(cfg, resolve=True))

    root = project / 'logs'
    root.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f'sqlite:///{root / "mlflow.db"}')

    name = cfg.tracking.experiment
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(
            name,
            artifact_location=(root / 'artifacts').as_uri(),
        )

    mlflow.set_experiment(name)

    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=project,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=no'],
        cwd=project,
        capture_output=True,
        text=True,
    )

    source_files = list((project / 'src').rglob('*.py'))
    config_files = list((project / 'configs').rglob('*.yaml'))

    provenance = {
        'python': sys.version,
        'executable': sys.executable,
        'git_commit': commit.stdout.strip() if commit.returncode == 0 else None,
        'git_dirty': bool(status.stdout.strip()) if status.returncode == 0 else None,
        'source_sha256': {
            str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source_files + config_files)
        },
        'packages': {
            name: importlib.metadata.version(name)
            for name in ['torch', 'torchvision', 'numpy', 'hydra-core', 'mlflow']
        },
    }
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))

    with mlflow.start_run(run_name=output.name) as run:
        (output / 'mlflow_run_id.txt').write_text(run.info.run_id)

        mlflow.log_params(flatten(OmegaConf.to_container(cfg, resolve=True)))
        protocol = {
            'smoke': 'engineering_train_only',
            'prepare': 'train_only_fold_preprocessing_no_classifier',
        }.get(cfg.get('mode'), 'raw_mri_patient_grouped_epoch_selection')
        mlflow.set_tag('protocol', protocol)

        for filename in ('pyproject.toml', 'uv.lock'):
            source = project / filename

            if source.exists():
                snapshot = output / filename
                shutil.copyfile(source, snapshot)
                mlflow.log_artifact(str(snapshot))

        mlflow.log_artifact(str(config))
        mlflow.log_artifact(str(output / 'provenance.json'))

        try:
            yield run

            results = output / 'metrics.json'
            if results.exists():
                values = json.loads(results.read_text())

                for key, value in values.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)

                mlflow.log_artifact(str(results))

            (output / 'status.json').write_text(json.dumps({'status': 'FINISHED'}))

        except BaseException:
            (output / 'status.json').write_text(json.dumps({'status': 'FAILED'}))
            (output / 'error.txt').write_text(traceback.format_exc())

            mlflow.log_artifact(str(output / 'error.txt'))
            raise

        finally:
            mlflow.log_artifact(str(output / 'status.json'))
