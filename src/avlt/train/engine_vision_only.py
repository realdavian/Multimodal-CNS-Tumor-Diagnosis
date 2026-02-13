"""
Training engine for the vision-only pipeline.

Mirrors the multimodal engine but strips the text data path,
cross-attention fusion, and alignment loss.  Keeps:
  - Teacher-student self-distillation via EMA
  - Mixed-precision (AMP)
  - DataParallel multi-GPU
  - Gradient clipping
  - Validation + metric / plot generation
"""

import os
import json
import time

import torch
from torch.utils.data import DataLoader, random_split

from ..models.avlt_vision_only import AVLTVisionOnly
from ..data.dataset_vision_only import SyntheticVisionDataset
from .losses_vision_only import VisionOnlyLosses
from ..utils.metrics import MetricTracker
from ..viz.plots import save_confusion, save_roc

from ..utils.loggers import logger


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def create_dataloaders(cfg):
    """Build train / val DataLoaders from config."""
    if cfg["dataset"] != "synthetic":
        raise NotImplementedError("Integrate your real imaging datasets here.")

    logger.debug("Creating synthetic dataset")
    ds = SyntheticVisionDataset(
        n=256,
        num_classes=cfg["num_classes"],
        image_size=cfg["image_size"],
        num_slices=cfg.get("num_slices", 16),
        split="train",
    )
    n_train = int(0.8 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    logger.debug("Creating dataloaders object")
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )
    return train_dl, val_dl


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _build_model(cfg, device):
    """Instantiate a single AVLTVisionOnly model on *device*."""
    return AVLTVisionOnly(
        num_classes=cfg["num_classes"],
        image_size=cfg["image_size"],
        backbone=cfg["vision"]["backbone"],
        dropout=cfg["dropout"],
        vision_variant=cfg["vision"].get("variant", "fixed"),
    ).to(device)


def _unwrap(model):
    """Return the underlying module (handles DataParallel)."""
    return model.module if hasattr(model, "module") else model


# ---------------------------------------------------------------------------
# EMA update
# ---------------------------------------------------------------------------

def _ema_update(model_s, model_t, momentum):
    """Exponential-moving-average update: teacher ← m·teacher + (1−m)·student."""
    s_state = _unwrap(model_s).state_dict()
    t_state = _unwrap(model_t).state_dict()
    with torch.no_grad():
        for key in t_state:
            t_state[key].copy_(momentum * t_state[key] + (1 - momentum) * s_state[key])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def train_loop(cfg, device=None, max_steps=None):
    """Full train → eval pipeline for the vision-only model.

    Args:
        cfg:        dict loaded from configs/vision_only.yaml
        device:     'cuda' / 'cpu' (auto-detected if None)
        max_steps:  optional early-stop for smoke tests
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f" Using device: {device}")
    os.makedirs(cfg["outputs"], exist_ok=True)

    # ---- data ----
    logger.debug("Creating dataloaders")
    start_time = time.time()
    train_dl, val_dl = create_dataloaders(cfg)
    end_time = time.time()
    logger.debug(f"Dataloaders created in {end_time - start_time:.2f} seconds")
    logger.info(f" Train samples: {len(train_dl.dataset)}")
    logger.info(f" Val samples: {len(val_dl.dataset)}")

    # ---- models (student + EMA teacher) ----
    model_s = _build_model(cfg, device)
    model_t = _build_model(cfg, device)
    model_t.load_state_dict(model_s.state_dict())
    for p in model_t.parameters():
        p.requires_grad = False

    n_gpus = torch.cuda.device_count()
    logger.info(f" Number of GPUs: {n_gpus}")
    if n_gpus > 1:
        logger.info(f" Using {n_gpus} GPUs via DataParallel")
        model_s = torch.nn.DataParallel(model_s)
        model_t = torch.nn.DataParallel(model_t)

    # ---- optimizer / scaler / losses ----
    opt = torch.optim.AdamW(
        model_s.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    use_amp = cfg["trainer"]["mixed_precision"] and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    losses = VisionOnlyLosses(w_sd=cfg["w_sd"])

    # ---- training ----
    step = 0
    for epoch in range(cfg["epochs"]):
        model_s.train()
        for batch in train_dl:
            step += 1
            imgs = batch["image"].to(device)
            y = batch["label"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits_s, _ = model_s(imgs)
                with torch.no_grad():
                    logits_t, _ = model_t(imgs)
                loss, parts = losses.total(logits_s, logits_t, y)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                model_s.parameters(), cfg["trainer"]["grad_clip"]
            )
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            _ema_update(model_s, model_t, cfg["ema_momentum"])

            if step % cfg["trainer"]["log_every"] == 0:
                logger.info(
                    f" epoch {epoch}  step {step}  "
                    f"loss {loss.item():.4f}  |  "
                    f"cls {parts['cls']:.3f}  sd {parts['sd']:.3f}"
                )

            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    # ---- save checkpoint ----
    ckpt_path = os.path.join(cfg["outputs"], "vision_only.pt")
    torch.save(_unwrap(model_s).state_dict(), ckpt_path)
    logger.info(f" Saved checkpoint: {ckpt_path}")

    # ---- evaluation ----
    metrics = MetricTracker(cfg["num_classes"])
    model_s.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for batch in val_dl:
            imgs = batch["image"].to(device)
            y = batch["label"].to(device)
            logits, _ = model_s(imgs)
            metrics.update(logits, y)
            y_true.append(y.cpu())
            y_prob.append(torch.softmax(logits, dim=1).cpu())

    y_true = torch.cat(y_true).numpy()
    y_prob = torch.cat(y_prob).numpy()

    report = metrics.report()
    metrics_path = os.path.join(cfg["outputs"], "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f" Metrics: {report}")

    save_confusion(y_true, y_prob.argmax(1), os.path.join(cfg["outputs"], "confusion.png"))
    save_roc(y_true, y_prob, os.path.join(cfg["outputs"], "roc.png"))
