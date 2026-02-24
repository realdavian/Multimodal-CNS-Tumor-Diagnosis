"""
W&B Sweep agent entry point.

This script is called by ``wandb agent`` for each sweep trial.
It reads the sweep parameters from ``wandb.config``, builds a
human-readable display name, and launches the existing training pipeline.

Usage:
    # Step 1: Create the sweep (one-time)
    source activate.sh && wandb sweep configs/sweep.yaml

    # Step 2: Launch the agent inside tmux so you can walk away
    tmux new -s sweep
    source activate.sh && wandb agent <ENTITY/PROJECT/SWEEP_ID>
    # Press Ctrl+B then D to detach from tmux

    # To re-attach later:
    tmux attach -t sweep
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import wandb
from omegaconf import OmegaConf

from avlt.train.engine import train_loop


def main():
    # W&B agent initializes the run and populates wandb.config
    run = wandb.init()

    # Read sweep parameters
    sweep_epochs = wandb.config.epochs
    sweep_lr = wandb.config.lr
    sweep_augmentation = wandb.config.augmentation

    # Build human-readable display name
    aug_tag = "augmented" if sweep_augmentation else "no-aug"
    lr_str = f"{sweep_lr:.10f}".rstrip("0")  # e.g. 0.00005
    display_name = f"Swin-UNET-{sweep_epochs}-Epoch-{aug_tag}-lr{lr_str}"

    # Set the DISPLAY name (not the run ID)
    wandb.run.name = display_name

    print(f"[Sweep] Display name: {display_name}")
    print(f"[Sweep] epochs={sweep_epochs}, lr={sweep_lr}, augmentation={sweep_augmentation}")

    # Load base + experiment config, then overlay sweep parameters
    base_cfg = OmegaConf.load("configs/base.yaml")
    exp_cfg = OmegaConf.load("configs/experiments/brats_os_multitask.yaml")
    cfg = OmegaConf.merge(base_cfg, exp_cfg)

    # Override with sweep values
    # wandb.enabled=true tells engine to USE the already-active run
    overrides = OmegaConf.create({
        "epochs": sweep_epochs,
        "lr": sweep_lr,
        "augmentation": sweep_augmentation,
        "wandb": {
            "enabled": True,
            "display_name": display_name,
        },
    })
    cfg = OmegaConf.merge(cfg, overrides)

    # Run training through the standard pipeline
    train_loop(cfg)

    wandb.finish()


if __name__ == "__main__":
    main()
