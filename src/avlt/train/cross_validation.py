"""
Stratified K-Fold Cross-Validation runner.

Wraps the existing ``train_loop`` to run multiple folds, each with 
its own unique output directory. Aggregates per-fold metrics into a
summary ``cv_results.json``.

Toggled via config:
    cv.enabled: true
    cv.n_folds: 5
"""

import os
import json
import datetime
import numpy as np

from sklearn.model_selection import StratifiedKFold

from .engine import train_loop, create_dataloaders, _cfg_get, _cfg_to_dict
from ..utils.loggers import logger


def _extract_labels(dataset):
    """Extract OS class labels from the full dataset for stratification.
    
    Works with both raw Dataset objects and their underlying .data list.
    """
    labels = []
    if hasattr(dataset, "data"):
        # BraTSDataset / BraTSMultitaskDataset store items in self.data
        for item in dataset.data:
            labels.append(item["label"])
    else:
        # Fallback: iterate through the dataset
        for i in range(len(dataset)):
            sample = dataset[i]
            labels.append(sample["label"].item() if hasattr(sample["label"], "item") else sample["label"])
    return np.array(labels)


def _build_full_dataset(cfg):
    """Build the full (unsplit) dataset based on config.
    
    Returns the dataset object before any train/val/test splitting.
    """
    dataset_name = _cfg_get(cfg, "dataset")
    mode = _cfg_get(cfg, "mode", "vision_only")
    augment = _cfg_get(cfg, "augmentation", False)

    if dataset_name == "brats_peds":
        from ..data.brats import BraTSDataset
        return BraTSDataset(
            data_root=_cfg_get(cfg, "data_root"),
            cohort_csv=_cfg_get(cfg, "cohort_csv"),
            split="train",
            image_size=_cfg_get(cfg, "image_size", 224),
            num_slices=_cfg_get(cfg, "num_slices", 128),
            mode=mode,
            augment=augment,
        )
    elif dataset_name == "brats_multitask":
        from ..data.brats_multitask import BraTSMultitaskDataset
        return BraTSMultitaskDataset(
            data_root=_cfg_get(cfg, "data_root"),
            cohort_csv=_cfg_get(cfg, "cohort_csv"),
            split="train",
            image_size=_cfg_get(cfg, "image_size", 224),
            num_slices=_cfg_get(cfg, "num_slices", 128),
            mode=mode,
            augment=augment,
        )
    else:
        raise NotImplementedError(f"Cross-validation not supported for dataset: {dataset_name}")


def run_cross_validation(cfg, device=None):
    """Execute stratified k-fold cross-validation.

    Each fold calls ``train_loop`` with the fold's train/val indices injected
    into the dataloader creation. Results are aggregated and saved.

    Args:
        cfg:     Dict or OmegaConf config.
        device:  ``'cuda'`` / ``'cpu'`` (auto-detected if None).
    """
    n_folds = _cfg_get(cfg, "cv.n_folds", 5)
    seed = _cfg_get(cfg, "seed", 42)
    out_dir_base = _cfg_get(cfg, "outputs", "outputs")

    # Generate a unique CV run directory
    cv_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_cv"
    cv_out_dir = os.path.join(out_dir_base, cv_run_id)
    os.makedirs(cv_out_dir, exist_ok=True)

    logger.info(f"Starting {n_folds}-fold cross-validation")
    logger.info(f"CV outputs: {cv_out_dir}")

    # Build full dataset and extract labels for stratification
    full_ds = _build_full_dataset(cfg)
    labels = _extract_labels(full_ds)
    indices = np.arange(len(full_ds))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        logger.info(f"--- Fold {fold_idx + 1}/{n_folds} ---")
        logger.info(f"  Train: {len(train_idx)} samples | Val: {len(val_idx)} samples")

        # Run training for this fold, passing fold indices
        fold_report = train_loop(
            cfg,
            device=device,
            fold_train_indices=train_idx.tolist(),
            fold_val_indices=val_idx.tolist(),
            fold_label=f"fold_{fold_idx + 1}",
        )

        fold_results.append({
            "fold": fold_idx + 1,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "metrics": fold_report,
        })

    # Aggregate results across folds
    summary = _aggregate_fold_results(fold_results)
    summary["n_folds"] = n_folds
    summary["total_samples"] = len(full_ds)
    summary["folds"] = fold_results

    # Save CV summary
    cv_path = os.path.join(cv_out_dir, "cv_results.json")
    with open(cv_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"CV results saved to {cv_path}")
    logger.info(f"CV Summary: {summary['aggregated']}")

    return summary


def _aggregate_fold_results(fold_results):
    """Compute mean and std of numeric metrics across folds.
    
    Handles nested 'validation' and 'test' keys in fold metrics.
    """
    aggregated = {}

    # Collect all numeric metrics from all folds
    for split_name in ["validation", "test"]:
        split_metrics = {}
        for fold in fold_results:
            metrics = fold["metrics"]
            if split_name not in metrics:
                continue
            for key, val in metrics[split_name].items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    if key not in split_metrics:
                        split_metrics[key] = []
                    split_metrics[key].append(val)

        if split_metrics:
            aggregated[split_name] = {}
            for key, values in split_metrics.items():
                arr = np.array(values)
                # Filter out inf/nan for robust aggregation
                valid = arr[np.isfinite(arr)]
                if len(valid) > 0:
                    aggregated[split_name][f"{key}_mean"] = float(np.mean(valid))
                    aggregated[split_name][f"{key}_std"] = float(np.std(valid))
                else:
                    aggregated[split_name][f"{key}_mean"] = float("nan")
                    aggregated[split_name][f"{key}_std"] = float("nan")

    return {"aggregated": aggregated}
