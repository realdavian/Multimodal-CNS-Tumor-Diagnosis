"""
Unified training engine for both vision-only and multimodal AVLT.

Mode is read from ``cfg["mode"]`` (or ``cfg.mode`` with OmegaConf):
  - ``"vision_only"`` → AVLTVisionOnly, no text data, no alignment loss
  - ``"multimodal"``  → AVLT, text data + cross-attention + alignment loss

Self-distillation via EMA teacher is handled by the ``SelfDistillation``
module and can be toggled with ``cfg["self_distillation"]`` (default: True).

W&B logging is enabled when ``cfg.wandb.enabled`` is True.
"""

import os
import json
import time
import datetime

import wandb
import torch
from torch.utils.data import DataLoader, random_split, Subset
from omegaconf import OmegaConf, DictConfig

from ..models import create_model
from ..data import create_dataset
from .losses import build_loss
from .distillation import SelfDistillation
from ..utils.metrics import MetricTracker
from ..utils.loggers import logger
from ..viz.plots import save_confusion, save_roc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg_get(cfg, key, default=None):
    """Access config value from dict or OmegaConf DictConfig."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.select(cfg, key, default=default)
    # Plain dict — support dotted keys
    keys = key.split(".")
    val = cfg
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
        if val is None:
            return default
    return val


def _cfg_to_dict(cfg):
    """Convert OmegaConf DictConfig to plain dict (for JSON serialization etc.)."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


# ---------------------------------------------------------------------------
# W&B wrappers
# ---------------------------------------------------------------------------

def _wandb_init(cfg):
    """Initialize W&B run if enabled.
    
    If a run is already active (e.g. from a W&B sweep agent), reuse it
    and upload the full training config. Supports an optional display name
    via ``wandb.display_name`` config key.

    Returns the run object or None.
    """
    if not _cfg_get(cfg, "wandb.enabled", False):
        return None
    try:
        cfg_dict = _cfg_to_dict(cfg)

        def code_filter(path: str) -> bool:
            # Explicitly ignore massive non-project code directories
            ignores = [".venv", "lib", "outputs", "wandb", ".git", "__pycache__", "paper_parsing", "tests"]
            if any(part in path.split(os.sep) for part in ignores):
                return False
            return path.endswith(".py") or path.endswith(".yaml") or path.endswith(".yml")
        
        # If a sweep agent already initialized a run, reuse it
        if wandb.run is not None:
            run = wandb.run
            # Upload the full training config so params appear in the W&B UI
            run.config.update(cfg_dict, allow_val_change=True)
            logger.info(f"Reusing existing W&B run: {run.name} ({run.url})")
            
        else:
            run = wandb.init(
                project=_cfg_get(cfg, "wandb.project", "avlt"),
                entity=_cfg_get(cfg, "wandb.entity"),
                config=cfg_dict,
                tags=list(_cfg_get(cfg, "wandb.tags", [])),
                notes=_cfg_get(cfg, "wandb.notes", ""),
                reinit=True,
            )
            logger.info(f"W&B run: {run.url}")
        
        # Upload codebase (scripts + yaml configs)
        run.log_code(".", include_fn=code_filter)

        # Set display name if provided
        display_name = _cfg_get(cfg, "wandb.display_name")
        if display_name:
            run.name = display_name
            logger.info(f"W&B display name: {display_name}")

        return run
    except Exception as e:
        logger.warning(f"W&B init failed ({e}). Continuing without W&B.")
        return None


def _wandb_log(run, data: dict, step: int = None):
    """Log to W&B if run is active."""
    if run is None:
        return
    wandb.log(data, step=step)


def _wandb_finish(run):
    """Finish W&B run if active."""
    if run is None:
        return
    wandb.finish()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def create_dataloaders(cfg, fold_train_indices=None, fold_val_indices=None):
    """Build train / val DataLoaders from config.
    
    Args:
        cfg:                Config dict or OmegaConf.
        fold_train_indices:  Optional list of indices for CV fold training subset.
        fold_val_indices:    Optional list of indices for CV fold validation subset.
    """
    dataset_name = _cfg_get(cfg, "dataset")
    mode = _cfg_get(cfg, "mode", "vision_only")
    augment = _cfg_get(cfg, "augmentation", False)
    
    logger.debug(f"Creating {dataset_name} dataset (mode={mode})")
    ds = create_dataset(
        dataset_name,
        # pass all possible args, individual datasets **kwargs ignore what they don't need
        n=256,
        num_classes=_cfg_get(cfg, "num_classes"),
        image_size=_cfg_get(cfg, "image_size", 224),
        num_slices=_cfg_get(cfg, "num_slices", 16 if dataset_name == "synthetic" else 128),
        split="train",
        mode=mode,
        text_model=_cfg_get(cfg, "text.model_name", "emilyalsentzer/Bio_ClinicalBERT"),
        text_maxlen=_cfg_get(cfg, "text_maxlen", 128),
        data_root=_cfg_get(cfg, "data_root"),
        cohort_csv=_cfg_get(cfg, "cohort_csv"),
        augment=augment,
    )
    
    n_train = int(0.8 * len(ds))
    # Synthetic has no test set logic currently
    if dataset_name == "synthetic":
        n_val = len(ds) - n_train
        n_test = 0
        train_ds, val_ds = random_split(
            ds, [n_train, n_val], 
            generator=torch.Generator().manual_seed(_cfg_get(cfg, "seed", 42))
        )
        test_ds = None
    else:
        n_val = int(0.1 * len(ds))
        n_test = len(ds) - n_train - n_val
        train_ds, val_ds, test_ds = random_split(
            ds, 
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(_cfg_get(cfg, "seed", 42))
        )
        
    logger.info(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds) if test_ds else 0}")

    # If CV fold indices are provided, use them instead of random_split
    if fold_train_indices is not None and fold_val_indices is not None:
        train_ds = Subset(ds, fold_train_indices)
        val_ds = Subset(ds, fold_val_indices)
        test_ds = None
        logger.info(f"CV fold split: train={len(train_ds)}, val={len(val_ds)}")

    train_dl = DataLoader(
        train_ds,
        batch_size=_cfg_get(cfg, "batch_size"),
        shuffle=True,
        num_workers=_cfg_get(cfg, "num_workers"),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=_cfg_get(cfg, "batch_size"),
        shuffle=False,
        num_workers=_cfg_get(cfg, "num_workers"),
    )
    test_dl = None
    if test_ds is not None:
        test_dl = DataLoader(
            test_ds,
            batch_size=_cfg_get(cfg, "batch_size"),
            shuffle=False,
            num_workers=_cfg_get(cfg, "num_workers"),
        )
    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _unwrap(model):
    """Return the underlying module (handles DataParallel)."""
    return model.module if hasattr(model, "module") else model


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def train_loop(cfg, device=None, fold_train_indices=None, fold_val_indices=None, fold_label=None):
    """Full train -> eval pipeline.

    Args:
        cfg:                Dict or OmegaConf DictConfig from YAML config.
        device:             ``'cuda'`` / ``'cpu'`` (auto-detected if None).
        fold_train_indices: Optional list of indices for CV fold training subset.
        fold_val_indices:   Optional list of indices for CV fold validation subset.
        fold_label:         Optional string label for the fold (e.g. "fold_1").
    
    Returns:
        dict with validation (and optionally test) metrics.
    """
    mode = _cfg_get(cfg, "mode", "vision_only")
    max_steps = _cfg_get(cfg, "max_steps")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Mode: {mode} | Device: {device}")

    out_dir_base = _cfg_get(cfg, "outputs", "outputs")

    # ---- W&B ----
    run = _wandb_init(cfg)
    
    # Generate unique run ID
    if run is not None:
        run_id = run.id
    else:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Append fold label if running CV
    if fold_label:
        run_id = f"{run_id}_{fold_label}"
        
    out_dir = os.path.join(out_dir_base, run_id)
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Outputs will be saved to {out_dir}")

    # ---- data ----
    t0 = time.time()
    train_dl, val_dl, test_dl = create_dataloaders(cfg, fold_train_indices, fold_val_indices)
    logger.debug(f"Dataloaders created in {time.time() - t0:.2f}s")
    test_size = len(test_dl.dataset) if test_dl is not None else 0
    logger.info(f"Train: {len(train_dl.dataset)} | Val: {len(val_dl.dataset)} | Test: {test_size}")

    # ---- student model ----
    model_s = create_model(cfg, device)

    # ---- self-distillation (EMA teacher) ----
    use_sd = _cfg_get(cfg, "self_distillation", True)
    distiller = SelfDistillation(
        student=model_s,
        momentum=_cfg_get(cfg, "ema_momentum", 0.999),
        device=device,
        enabled=use_sd,
    )

    # ---- multi-GPU ----
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        logger.info(f"Using {n_gpus} GPUs via DataParallel")
        model_s = torch.nn.DataParallel(model_s)
        distiller.wrap_parallel()

    # ---- optimizer / scaler / losses ----
    opt = torch.optim.AdamW(
        model_s.parameters(),
        lr=float(_cfg_get(cfg, "lr")),
        weight_decay=float(_cfg_get(cfg, "weight_decay")),
    )
    use_amp = _cfg_get(cfg, "trainer.mixed_precision", True) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loss_fn = build_loss(cfg)

    log_every = _cfg_get(cfg, "trainer.log_every", 10)
    grad_clip = _cfg_get(cfg, "trainer.grad_clip", 1.0)

    # ---- training ----
    step = 0
    for epoch in range(_cfg_get(cfg, "epochs", 3)):
        model_s.train()
        for batch in train_dl:
            step += 1
            imgs = batch["image"].to(device)
            y = batch["label"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                if mode == "multimodal":
                    batch["input_ids"] = batch["input_ids"].to(device)
                    batch["attention_mask"] = batch["attention_mask"].to(device)
                    outputs_s = model_s(imgs, batch["input_ids"], batch["attention_mask"])
                else:
                    outputs_s = model_s(imgs)

                outputs_t = None
                if use_sd:
                    if mode == "multimodal":
                        outputs_t = distiller.forward(imgs, batch["input_ids"], batch["attention_mask"])
                    else:
                        outputs_t = distiller.forward(imgs)

                # Loss function blindly extracts what it needs
                loss, parts = loss_fn.total(batch, outputs_s, outputs_t)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            # EMA update
            if use_sd:
                distiller.update(model_s)

            if step % log_every == 0:
                parts_str = "  ".join(f"{k} {v:.3f}" for k, v in parts.items())
                logger.info(f"epoch {epoch}  step {step}  loss {loss.item():.4f}  |  {parts_str}")

            # W&B step logging
            _wandb_log(run, {"train/loss": loss.item(), **{f"train/{k}": v for k, v in parts.items()}}, step=step)

            if max_steps and step >= max_steps:
                break
                
            # Periodic evaluation
            val_every_steps = _cfg_get(cfg, "trainer.val_every", 1000)
            if step % val_every_steps == 0:
                logger.info(f"--- Step {step} Validation ---")
                val_report = _evaluate(model_s, val_dl, cfg, device, mode, out_dir, prefix="val")
                metrics_tracker_log = {f"val/{k}": v for k, v in val_report.items() if isinstance(v, (int, float))}
                _wandb_log(run, metrics_tracker_log, step=step)
                
                # Evaluation sets the model to eval(), we must set it back to train
                model_s.train()
                
                # Save periodic checkpoint
                step_ckpt_name = f"avlt_{mode}_step{step}.pt"
                step_ckpt_path = os.path.join(out_dir, step_ckpt_name)
                torch.save(_unwrap(model_s).state_dict(), step_ckpt_path)

        if max_steps and step >= max_steps:
            break

    # ---- save final checkpoint ----
    ckpt_name = "avlt_multimodal.pt" if mode == "multimodal" else "avlt_vision_only.pt"
    ckpt_path = os.path.join(out_dir, ckpt_name)
    torch.save(_unwrap(model_s).state_dict(), ckpt_path)
    logger.info(f"Saved checkpoint: {ckpt_path}")

    # ---- evaluation ----
    logger.info("Running evaluation on validation set...")
    val_report = _evaluate(model_s, val_dl, cfg, device, mode, out_dir, prefix="val")
    final_report = {"validation": val_report}
    _wandb_log(run, {f"val/{k}": v for k, v in val_report.items() if isinstance(v, (int, float))})

    if test_dl is not None:
        logger.info("Running evaluation on test set...")
        test_report = _evaluate(model_s, test_dl, cfg, device, mode, out_dir, prefix="test")
        final_report["test"] = test_report
        _wandb_log(run, {f"test/{k}": v for k, v in test_report.items() if isinstance(v, (int, float))})

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_report, f, indent=2)
    logger.info(f"Final Metrics: {final_report}")

    _wandb_finish(run)

    return final_report


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(model, dl, cfg, device, mode, out_dir, prefix="val"):
    """Run evaluation and save metrics + plots. Returns the metrics dict."""
    metrics_tracker = MetricTracker(_cfg_get(cfg, "num_classes"))
    model.eval()
    y_true, y_prob = [], []
    
    # Optional Seg Metrics
    dice_metric = None
    hd95_metric = None
    sd_metric = None
    iou_metric = None
    if mode == "multitask":
        from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric, MeanIoU
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        sd_metric = SurfaceDistanceMetric(include_background=False, reduction="mean")
        iou_metric = MeanIoU(include_background=False, reduction="mean")

    with torch.no_grad():
        for batch in dl:
            imgs = batch["image"].to(device)
            y = batch["label"].to(device)

            if mode == "multimodal":
                batch["input_ids"] = batch["input_ids"].to(device)
                batch["attention_mask"] = batch["attention_mask"].to(device)
                outputs = model(imgs, batch["input_ids"], batch["attention_mask"])
            else:
                outputs = model(imgs)

            logits = outputs["os_logits"]

            if mode == "multitask":
                seg_mask = batch["seg_mask"].to(device)
                seg_logits = outputs["seg_logits"]
                
                # Compute Dice
                import monai.transforms as mt
                # We need discrete predictions for Dice
                val_outputs_discrete = [mt.AsDiscrete(argmax=True, to_onehot=4)(i) for i in seg_logits]
                val_labels_discrete = [mt.AsDiscrete(to_onehot=4)(i.unsqueeze(0)) for i in seg_mask]
                
                # Stack back
                val_outputs_discrete = torch.stack(val_outputs_discrete, dim=0)
                val_labels_discrete = torch.stack(val_labels_discrete, dim=0)
                dice_metric(y_pred=val_outputs_discrete, y=val_labels_discrete)
                hd95_metric(y_pred=val_outputs_discrete, y=val_labels_discrete)
                sd_metric(y_pred=val_outputs_discrete, y=val_labels_discrete)
                iou_metric(y_pred=val_outputs_discrete, y=val_labels_discrete)

            metrics_tracker.update(logits, y)
            y_true.append(y.cpu())
            y_prob.append(torch.softmax(logits, dim=1).cpu())

    y_true = torch.cat(y_true).numpy()
    y_prob = torch.cat(y_prob).numpy()

    report = metrics_tracker.report()
    
    if mode == "multitask":
        mean_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        report["mean_dice"] = mean_dice
        
        # Aggregate Hausdorff Distance
        hd95_aggr = hd95_metric.aggregate()
        report["mean_hd95"] = hd95_aggr.nanmean().item() if hd95_aggr is not None else float("inf")
        hd95_metric.reset()
        
        # Aggregate Surface Distance
        sd_aggr = sd_metric.aggregate()
        report["mean_surface_distance"] = sd_aggr.nanmean().item() if sd_aggr is not None else float("inf")
        sd_metric.reset()
        
        # Aggregate Mean IoU
        iou_aggr = iou_metric.aggregate()
        report["mean_iou"] = iou_aggr.nanmean().item() if iou_aggr is not None else float("nan")
        iou_metric.reset()
        
    save_confusion(y_true, y_prob.argmax(1), os.path.join(out_dir, f"{prefix}_confusion.png"))
    save_roc(y_true, y_prob, os.path.join(out_dir, f"{prefix}_roc.png"))

    return report
