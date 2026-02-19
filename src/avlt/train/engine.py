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

import torch
from torch.utils.data import DataLoader, random_split
from omegaconf import OmegaConf, DictConfig

from ..models.avlt import AVLT
from ..models.avlt_vision_only import AVLTVisionOnly
from ..data.dataset import SyntheticDataset
from .losses import Losses
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
    """Initialize W&B run if enabled. Returns the run object or None."""
    if not _cfg_get(cfg, "wandb.enabled", False):
        return None
    try:
        import wandb
        run = wandb.init(
            project=_cfg_get(cfg, "wandb.project", "avlt"),
            entity=_cfg_get(cfg, "wandb.entity"),
            config=_cfg_to_dict(cfg),
            tags=list(_cfg_get(cfg, "wandb.tags", [])),
            notes=_cfg_get(cfg, "wandb.notes", ""),
            reinit=True,
        )
        logger.info(f"W&B run: {run.url}")
        return run
    except Exception as e:
        logger.warning(f"W&B init failed ({e}). Continuing without W&B.")
        return None


def _wandb_log(run, data: dict, step: int = None):
    """Log to W&B if run is active."""
    if run is None:
        return
    import wandb
    wandb.log(data, step=step)


def _wandb_finish(run):
    """Finish W&B run if active."""
    if run is None:
        return
    import wandb
    wandb.finish()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def create_dataloaders(cfg):
    """Build train / val DataLoaders from config."""
    if _cfg_get(cfg, "dataset") != "synthetic":
        raise NotImplementedError("Integrate your real imaging datasets here.")

    mode = _cfg_get(cfg, "mode", "vision_only")
    logger.debug(f"Creating synthetic dataset (mode={mode})")

    ds = SyntheticDataset(
        n=256,
        num_classes=_cfg_get(cfg, "num_classes"),
        image_size=_cfg_get(cfg, "image_size"),
        num_slices=_cfg_get(cfg, "num_slices", 16),
        split="train",
        mode=mode,
        text_model=_cfg_get(cfg, "text.model_name", "emilyalsentzer/Bio_ClinicalBERT"),
        text_maxlen=_cfg_get(cfg, "text_maxlen", 128),
    )
    n_train = int(0.8 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])

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
    return train_dl, val_dl


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _build_model(cfg, device):
    """Instantiate the correct AVLT variant based on mode."""
    mode = _cfg_get(cfg, "mode", "vision_only")
    if mode == "multimodal":
        return AVLT(
            num_classes=_cfg_get(cfg, "num_classes"),
            image_size=_cfg_get(cfg, "image_size"),
            backbone=_cfg_get(cfg, "vision.backbone"),
            text_model=_cfg_get(cfg, "text.model_name"),
            dropout=_cfg_get(cfg, "dropout"),
            vision_variant=_cfg_get(cfg, "vision.variant", "fixed"),
        ).to(device)
    else:
        return AVLTVisionOnly(
            num_classes=_cfg_get(cfg, "num_classes"),
            image_size=_cfg_get(cfg, "image_size"),
            backbone=_cfg_get(cfg, "vision.backbone"),
            dropout=_cfg_get(cfg, "dropout"),
            vision_variant=_cfg_get(cfg, "vision.variant", "fixed"),
        ).to(device)


def _unwrap(model):
    """Return the underlying module (handles DataParallel)."""
    return model.module if hasattr(model, "module") else model


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def train_loop(cfg, device=None):
    """Full train → eval pipeline.

    Args:
        cfg:        Dict or OmegaConf DictConfig from YAML config.
        device:     ``'cuda'`` / ``'cpu'`` (auto-detected if None).
    """
    mode = _cfg_get(cfg, "mode", "vision_only")
    max_steps = _cfg_get(cfg, "max_steps")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Mode: {mode} | Device: {device}")

    out_dir = _cfg_get(cfg, "outputs", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ---- W&B ----
    run = _wandb_init(cfg)

    # ---- data ----
    t0 = time.time()
    train_dl, val_dl = create_dataloaders(cfg)
    logger.debug(f"Dataloaders created in {time.time() - t0:.2f}s")
    logger.info(f"Train: {len(train_dl.dataset)} | Val: {len(val_dl.dataset)}")

    # ---- student model ----
    model_s = _build_model(cfg, device)

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

    losses = Losses(
        w_align=_cfg_get(cfg, "w_align", 0.0),
        w_sd=_cfg_get(cfg, "w_sd", 0.5),
    )

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
                    ids = batch["input_ids"].to(device)
                    attn = batch["attention_mask"].to(device)
                    logits_s, f_v, f_t, f_fused, alpha, beta = model_s(imgs, ids, attn)
                    # Teacher forward
                    logits_t = None
                    if use_sd:
                        teacher_out = distiller.forward(imgs, ids, attn)
                        logits_t = teacher_out[0] if teacher_out is not None else None
                    loss, parts = losses.total(logits_s, logits_t, y, f_v, f_t)
                else:
                    logits_s, f_v = model_s(imgs)
                    # Teacher forward
                    logits_t = None
                    if use_sd:
                        teacher_out = distiller.forward(imgs)
                        logits_t = teacher_out[0] if teacher_out is not None else None
                    loss, parts = losses.total(logits_s, logits_t, y)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            # EMA update
            distiller.update(model_s)

            if step % log_every == 0:
                parts_str = "  ".join(f"{k} {v:.3f}" for k, v in parts.items())
                logger.info(f"epoch {epoch}  step {step}  loss {loss.item():.4f}  |  {parts_str}")

            # W&B step logging
            _wandb_log(run, {"train/loss": loss.item(), **{f"train/{k}": v for k, v in parts.items()}}, step=step)

            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    # ---- save checkpoint ----
    ckpt_name = "avlt_multimodal.pt" if mode == "multimodal" else "avlt_vision_only.pt"
    ckpt_path = os.path.join(out_dir, ckpt_name)
    torch.save(_unwrap(model_s).state_dict(), ckpt_path)
    logger.info(f"Saved checkpoint: {ckpt_path}")

    # ---- evaluation ----
    eval_report = _evaluate(model_s, val_dl, cfg, device, mode)
    _wandb_log(run, {f"eval/{k}": v for k, v in eval_report.items() if isinstance(v, (int, float))})
    _wandb_finish(run)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(model, val_dl, cfg, device, mode):
    """Run evaluation and save metrics + plots. Returns the metrics dict."""
    metrics = MetricTracker(_cfg_get(cfg, "num_classes"))
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for batch in val_dl:
            imgs = batch["image"].to(device)
            y = batch["label"].to(device)

            if mode == "multimodal":
                ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                logits, *_ = model(imgs, ids, attn)
            else:
                logits, *_ = model(imgs)

            metrics.update(logits, y)
            y_true.append(y.cpu())
            y_prob.append(torch.softmax(logits, dim=1).cpu())

    y_true = torch.cat(y_true).numpy()
    y_prob = torch.cat(y_prob).numpy()

    report = metrics.report()
    out_dir = _cfg_get(cfg, "outputs", "outputs")
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Metrics: {report}")

    save_confusion(y_true, y_prob.argmax(1), os.path.join(out_dir, "confusion.png"))
    save_roc(y_true, y_prob, os.path.join(out_dir, "roc.png"))

    return report
