from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'configs/dataset/mri.yaml'


@dataclass(frozen=True)
class Source:
    name: str
    path: Path
    label: int


def dataset_sources(config_path=None, root=None):
    config = OmegaConf.load(config_path or DEFAULT_CONFIG)
    selected_root = root if root is not None else config.root

    if not selected_root:
        raise ValueError('Set MRI_DATASET_ROOT or provide a dataset root explicitly')

    root_path = Path(selected_root).expanduser().resolve()
    result = []

    for name, settings in config.sources.items():
        if not settings.enabled:
            continue

        if Path(name).name != name or name.startswith('.') or name.startswith('_'):
            raise ValueError('Expected a single active source folder name')

        if type(settings.label) is not int or not 0 <= settings.label <= 5:
            raise ValueError('Class labels must be integers between 0 and 5')

        path = root_path / name

        if not path.is_dir():
            raise FileNotFoundError(path)

        result.append(Source(name, path, settings.label))

    if not result:
        raise ValueError('No dataset sources enabled')

    return result
