# AVLT Developer Guide: Architecture & Experimentation

Welcome to the `avlt` project. This guide details the coding standards and structural design of the framework. The project is designed to enable researchers to rapidly test new ideas—such as custom neural backbones, new loss functions, or novel datasets—without directly hacking the core training loops (`engine.py`).

The overarching philosophy is **Config-Driven Plugin Architecture**. We separate the *definition of a component* from its *instantiation*, heavily relying on factory patterns and [Hydra](https://hydra.cc/docs/intro/) for configuration.

---

## 1. Core Architecture Pattern

To prevent overhead and spaghetti code when experimenting, the framework enforces a strict separation of concerns:

- **Config (`configs/`)**: The source of truth. Every experiment variant is defined in YAML.
- **Components (`src/avlt/`)**: Contains models, data loaders, and losses. These are pure PyTorch modules.
- **Factories**: Act as glue. The config strings (e.g., `vision_variant: swin3d`) tell the factory which Python class to instantiate.
- **Engine (`src/avlt/train/engine.py`)**: The runner. It asks the factories for the objects it needs and executes the training loop.

By following this standard, you guarantee that a new component will interact correctly with W&B logging, self-distillation, cross-validation, and AMP (Automatic Mixed Precision) without you having to write that boilerplate.

---

## 2. Implementing New Components

### A. Adding a New Vision Encoder
Encoders map raw 3D inputs to feature vectors (`f_v`). They are managed by a registry in `src/avlt/models/encoders/__init__.py`.

**Step 1:** Create your module (e.g., `src/avlt/models/encoders/my_new_encoder.py`)
```python
import torch.nn as nn

class MyNewEncoder(nn.Module):
    def __init__(self, in_channels=4, out_dim=768, **kwargs):
        super().__init__()
        # Define architecture...
        pass
        
    def forward(self, x):
        # Must return (features, [optional auxiliary outputs])
        # For vision-only/multimodal: returns (f_v)
        # For multitask: returns (f_v, seg_logits)
        pass
```

**Step 2:** Register it in `src/avlt/models/encoders/__init__.py`
```python
from .my_new_encoder import MyNewEncoder

def create_vision_encoder(variant='fixed', **kwargs):
    variants = {
        'swin3d': Swin3DVisionEncoder,
        'my_new_variant': MyNewEncoder,  # Add your key here
    }
    return variants[variant](**kwargs)
```

**Step 3:** Use it via config
```yaml
vision:
  variant: my_new_variant
```

### B. Adding a New Loss Function
Loss functions use a factory pattern based in `src/avlt/train/losses.py`.

**Step 1:** Define the PyTorch loss wrapper or use a library (e.g., MONAI).
**Step 2:** Add it to the registry inside `_build_cls_loss` (for classification) or `_build_seg_loss` (for segmentation).

```python
def _build_cls_loss(name: str, **kwargs):
    registry = {
        "cross_entropy": lambda: nn.CrossEntropyLoss(),
        "my_focal_loss": lambda: MyFocalLossWrapper(), # Your new loss
    }
    return registry[name]()
```

**Step 3:** Tell the experiment to use it via config:
```yaml
loss:
  classification: my_focal_loss
```
The overarching `MultitaskLoss`, `VisionOnlyLoss`, etc., will automatically pick it up and wire it into the total loss calculation.

### C. Adding a New Dataset
Datasets use a factory pattern based in `src/avlt/data/__init__.py`. The engine dynamically creates datasets without needing hardcoded imports.

**Step 1:** Create your PyTorch Dataset in `src/avlt/data/my_new_dataset.py`.
**Step 2:** Hook it into the registry in `src/avlt/data/__init__.py`:
```python
from .my_new_dataset import MyNewDataset

def create_dataset(dataset_name: str, **kwargs):
    registry = {
        "brats_peds": BraTSDataset,
        "my_new_data": MyNewDataset, # <-- Add your hook here
    }
    return registry[dataset_name](**kwargs)
```

**Step 3:** Tell the experiment to use it via config:
```yaml
dataset: my_new_data
```

### D. Adding a New Top-Level Model Wrapper
Top-level models (`AVLT`, `AVLTVisionMultitask`) compile encoders, fusion blocks, and heads together. They are managed by `src/avlt/models/__init__.py`.

**Crucial Standard - Dictionary Outputs:**
To prevent messy `if/else` tuple unpacking chains in the training loop, **all top-level models MUST return a standard Python Dictionary**.
Example dictionary output:
```python
return {
    "os_logits": os_logits,         # (B, num_classes)
    "seg_logits": seg_logits,       # (B, num_seg_classes, D, H, W)
    "f_v": f_v,                     # (B, 768)
    "f_t": f_t                      # (B, 768)
}
```
The loss subclasses (`MultitaskLoss`, etc.) will cleanly pull exactly the keys they need from this dictionary.

**Step 1:** Create your module (e.g., `src/avlt/models/my_new_wrapper.py`)
**Step 2:** Register it in `src/avlt/models/__init__.py` inside `create_model()`.

---

## 3. Running Experiments

Experiments rely on overriding `configs/base.yaml` using Hydra. **Do not modify `base.yaml` for a specific experiment runs.** Instead, formulate a new layout.

### A. Creating an Experiment Definition
In `configs/experiments/`, create a YAML file representing your hypothesis (e.g., `configs/experiments/my_idea.yaml`).

```yaml
# configs/experiments/my_idea.yaml
# We are testing if MyNewEncoder improves OS Accuracy on the multitask dataset

dataset: brats_multitask
mode: multitask       # Tells engine to use AVLTVisionMultitask & MultitaskLoss
vision:
  variant: my_new_variant

loss:
  segmentation: dice_focal
  w_seg: 0.3          # Re-weight segmentation 

epochs: 30            # Override base epochs
augmentation: true
```

### B. Running Locally
Use the plus (`+`) operator to tell Hydra to merge your experiment YAML over the base config.
```bash
source activate.sh
python scripts/train.py +experiment=my_idea
```
You can also override fields directly from the CLI without changing the file:
```bash
python scripts/train.py +experiment=my_idea lr=1e-4 batch_size=16
```

### C. Sweeps for Hyperparameter Tuning
We use W&B Sweeps to automate finding optimal hyperparameter parameters.

Create a sweep config (e.g., `configs/sweep_my_idea.yaml`):
```yaml
program: scripts/run_sweep.py
method: grid
project: avlt

parameters:
  epochs:
    values: [20, 50]
  loss.w_seg:
    values: [0.1, 0.5, 1.0]
```
Note: Ensure you update `scripts/run_sweep.py` to point to your `my_idea.yaml` experiment config.

**Launch:**
1. `wandb sweep configs/sweep_my_idea.yaml`
2. `wandb agent <sweep_id>`

---

## 4. Coding Golden Rules
- **No implicit arguments, no tuple unpacking in chains:** The engine and loss refactoring ensures that top-level model outputs are dictionaries (`outputs_s = model(imgs)`). Do not write `if mode == "multitask": logits, seg_logits = model(imgs)`. Let the loss subclasses gracefully extract what they need from the dictionary.
- **Hydra Configs over Argparse:** Everything that controls model shape, training loop lengths, or tensor shapes belongs in YAML. 
- **Optional Components:** Components like Self-Distillation should gracefully turn off if `self_distillation: false` is in the config, skipping heavy memory allocation.
- **Avoid Model Side-Effects:** Models (`nn.Module`) should only take tensors and return tensors (or dictionaries of tensors). They should not calculate their own loss, touch the W&B logger, or read directly from Pandas CSVs.
