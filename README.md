# Adaptive Vision–Language Transformer (AVLT) — CNS Tumor Diagnosis

This repository provides a reproducible PyTorch implementation of the **Adaptive Vision–Language Transformer (AVLT)**
for multimodal CNS tumor diagnosis using MRI (T1/T1ce/T2/FLAIR) and clinical/genomic text.

It includes:
- Vision encoder (CNN stem + ViT via `timm`) with multiple encoder variants
- Language encoder (ClinicalBERT via `transformers`)
- Cross-attention + adaptive gating fusion
- Student–teacher self-distillation via EMA
- **Hydra** config management with per-experiment overrides
- **Weights & Biases** experiment tracking
- Synthetic data generator for smoke tests (no real data needed to get started)

---

## Setup

1. **Clone with submodules** (required for `Swin-UMamba` encoder):
```bash
git submodule update --init --recursive
```

2. **Create and activate environment (using your system's package repo)**:
```bash
conda create -n davian-py3110 python=3.11 -y
conda activate davian-py3110
```

3. **Install `uv` (if not already installed)**:
```bash
pip install uv
```

4. **Install dependencies**:
```bash
# 1. Install Swin-UMamba from git submodule
uv pip install -e lib/Swin-UMamba/swin_umamba
# 2. Install local project
uv pip install -e .
```

For W&B tracking, authenticate once:
```bash
wandb login
```

---

## Quick Start

`dataset` defaults to `synthetic` in `base.yaml` — no extra flag needed for smoke tests.

```bash
# Smoke test: vision-only, 5 steps
python scripts/train.py max_steps=5

# Evaluate latest checkpoint
python scripts/eval.py ckpt=outputs/avlt_vision_only.pt
```

---

## Training

Hydra handles config composition and CLI overrides:

```bash
# Default: vision-only, fixed encoder
python scripts/train.py

# Use an experiment preset
python scripts/train.py +experiment=baseline_slice_wise

# Override any config value from CLI
python scripts/train.py vision.variant=slice_wise mode=multimodal batch_size=4

# Enable W&B logging
python scripts/train.py wandb.enabled=true wandb.project=avlt

# Smoke test (5 steps, fast with cached data)
python scripts/train.py max_steps=5
```

### Experiment Presets

Located in `configs/experiments/`. Select one with `+experiment=<name>`:

| Name | Command | Description |
|------|---------|-------------|
| CNN + ViT | `+experiment=baseline_cnn_vit` | 2D encoder — CNN stem + ViT with MaxPool (default) |
| Slice-wise | `+experiment=baseline_slice_wise` | 2.5D encoder — per-slice ViT + slice attention |
| Swin3D | `+experiment=baseline_swin3d` | 3D encoder — MONAI SwinUNETR (batch_size=2) |

### Key Config Fields

All fields in `configs/base.yaml` are overridable from the CLI.

| Field | Default | Description |
|-------|---------|-------------|
| `mode` | `vision_only` | `vision_only` or `multimodal` |
| `vision.variant` | `fixed` | `fixed`, `no_pool`, `slice_wise`, `swin3d`, `original` |
| `dataset` | `synthetic` | Dataset to use (`synthetic` or a real dataset key) |
| `batch_size` | `8` | Training batch size |
| `epochs` | `3` | Number of training epochs |
| `lr` | `1e-4` | Learning rate |
| `self_distillation` | `true` | Enable EMA student-teacher distillation |
| `max_steps` | `null` | Hard stop after N steps (for smoke tests) |
| `wandb.enabled` | `false` | Enable W&B logging |

---

## Evaluation & Inference

```bash
# Evaluate a checkpoint
python scripts/eval.py ckpt=outputs/avlt_vision_only.pt

# Evaluate a multimodal checkpoint
python scripts/eval.py ckpt=outputs/avlt_multimodal.pt mode=multimodal

# Single-case inference (random volume if no image path given)
python scripts/infer.py ckpt=outputs/avlt_vision_only.pt
python scripts/infer.py ckpt=outputs/avlt_vision_only.pt image=path/to/volume.npy
```

---

## Real Data Usage

> **Not yet implemented**: `engine.py` raises `NotImplementedError` for any `dataset` other than `synthetic`.
> When integrating real data, implement a new `Dataset` class in `src/avlt/data/` and register it in `create_dataloaders()`.

Expected data layout:

```
data/
  brats/
    imagesTr/    # NIfTI volumes (T1/T1ce/T2/FLAIR, preprocessed)
    labelsTr/    # CSV: {patient_id, label}
    textTr/      # CSV or per-patient .txt (clinical notes)
```

---

## Repository Layout

```
configs/
  base.yaml                    # Main config (Hydra entry point)
  experiments/                 # Per-experiment overrides (+experiment=<name>)
src/avlt/
  models/
    avlt.py                    # Multimodal AVLT (vision + text + fusion)
    avlt_vision_only.py        # Vision-only AVLT variant
    encoders/                  # Vision encoder variants (fixed, slice_wise, swin3d, …)
    fusion.py                  # Cross-attention + adaptive gating
    layers.py                  # SliceAttention
  data/
    dataset.py                 # SyntheticDataset (cached); add real datasets here
  train/
    engine.py                  # Train loop (AMP, DataParallel, W&B)
    losses.py                  # Classification + distillation + alignment losses
    distillation.py            # EMA student-teacher module
  utils/
    metrics.py                 # ACC, F1, AUC
    loggers.py                 # Logging setup
  viz/
    plots.py                   # ROC curve, confusion matrix
scripts/
  train.py                     # Training entry (Hydra)
  eval.py                      # Evaluation entry (Hydra)
  infer.py                     # Single-case inference (Hydra)
```

---

## License
Apache-2.0
