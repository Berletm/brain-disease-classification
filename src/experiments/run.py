from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __package__ in (None, ''):
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


@hydra.main(
    config_path=str(PROJECT_ROOT / "configs"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    from src.experiments.tracking import tracked_run
    from src.experiments.nested import run_nested, run_prepare
    from src.experiments.smoke import run_smoke

    with tracked_run(cfg, project=PROJECT_ROOT):
        runners = {
            'nested': run_nested,
            'smoke': run_smoke,
            'prepare': run_prepare,
        }
        runner = runners[cfg.mode]
        runner(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == '__main__':
    main()
