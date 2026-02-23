"""
Entry point: ``python scripts/train.py``

Uses Hydra for config management and CLI overrides.

Usage::

    # Default: vision-only with base config
    python scripts/train.py

    # Select experiment override
    python scripts/train.py +experiment=baseline_swin3d

    # Override any config value from CLI
    python scripts/train.py vision.variant=slice_wise mode=multimodal

    # Enable W&B logging
    python scripts/train.py wandb.enabled=true

    # Smoke test (5 steps)
    python scripts/train.py max_steps=5

    # Enable data augmentation
    python scripts/train.py +experiment=brats_os_multitask augmentation=true

    # Enable cross-validation
    python scripts/train.py +experiment=brats_os_multitask cv.enabled=true cv.n_folds=5

    # Combine experiment + overrides
    python scripts/train.py +experiment=baseline_swin3d max_steps=10 wandb.enabled=true
"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from avlt.train.engine import train_loop


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    # Handle +experiment= by loading the experiment YAML and merging it
    experiment = cfg.get("experiment")
    if experiment is not None:
        exp_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "experiments", f"{experiment}.yaml"
        )
        exp_cfg = OmegaConf.load(exp_path)
        # Strip any Hydra comments/directives from loaded config
        if "_target_" in exp_cfg:
            del exp_cfg["_target_"]
        cfg = OmegaConf.merge(cfg, exp_cfg)

    # Dispatch to cross-validation or standard training
    cv_enabled = OmegaConf.select(cfg, "cv.enabled", default=False)
    if cv_enabled:
        from avlt.train.cross_validation import run_cross_validation
        run_cross_validation(cfg)
    else:
        train_loop(cfg)


if __name__ == "__main__":
    main()
