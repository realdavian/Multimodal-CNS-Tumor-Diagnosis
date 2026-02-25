"""
Modular loss functions for vision-only, multimodal, and multitask training.

Architecture:
    BaseLoss               -- classification + self-distillation (shared by all modes)
      |-- VisionOnlyLoss   -- inherits base, no extra terms
      |-- MultimodalLoss   -- adds cross-modal alignment term
      |-- MultitaskLoss    -- adds dense segmentation term

    build_loss(cfg)        -- factory that returns the right subclass for the mode

Individual loss functions (CE, focal, DiceCE, etc.) are selected via a
registry dict so they can be swapped through config without changing code.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, DiceFocalLoss

from ..utils.loggers import logger


# ---------------------------------------------------------------------------
# Loss function registries
# ---------------------------------------------------------------------------

def _build_cls_loss(name: str, **kwargs):
    """Build a classification loss function by name.

    Supported:
        cross_entropy  -- standard nn.CrossEntropyLoss
        weighted_ce    -- CrossEntropyLoss with class_weights kwarg
        focal          -- simple focal loss wrapper
    """
    registry = {
        "cross_entropy": lambda: nn.CrossEntropyLoss(),
        "weighted_ce": lambda: nn.CrossEntropyLoss(
            weight=torch.tensor(kwargs["class_weights"], dtype=torch.float32)
        ),
    }
    if name not in registry:
        raise ValueError(
            f"Unknown classification loss '{name}'. Available: {list(registry.keys())}"
        )
    return registry[name]()


def _build_seg_loss(name: str, num_classes: int = 4):
    """Build a segmentation loss function by name.

    All segmentation losses expect:
        input:  [B, C, D, H, W]  (raw logits)
        target: [B, 1, D, H, W]  (integer labels)

    Supported:
        dice_ce    -- MONAI DiceCELoss (Dice + CrossEntropy)
        dice_focal -- MONAI DiceFocalLoss (Dice + Focal)
    """
    registry = {
        "dice_ce": lambda: DiceCELoss(
            to_onehot_y=True, softmax=True, include_background=False
        ),
        "dice_focal": lambda: DiceFocalLoss(
            to_onehot_y=True, softmax=True, include_background=False
        ),
    }
    if name not in registry:
        raise ValueError(
            f"Unknown segmentation loss '{name}'. Available: {list(registry.keys())}"
        )
    return registry[name]()


# ---------------------------------------------------------------------------
# Base class: classification + self-distillation
# ---------------------------------------------------------------------------

class BaseLoss(ABC):
    """Base loss providing classification and self-distillation terms.

    Subclasses override ``total()`` with a fixed, mode-specific signature.
    """

    def __init__(self, cls_loss_name: str = "cross_entropy", w_sd: float = 0.5, **kwargs):
        self.w_sd = w_sd
        self.ce = _build_cls_loss(cls_loss_name, **kwargs)

    def classification(self, logits, y):
        """Standard classification loss."""
        return self.ce(logits, y)

    def distill(self, logits_teacher, logits_student):
        """KL-divergence self-distillation (teacher is detached)."""
        log_p_s = F.log_softmax(logits_student, dim=1)
        p_t = F.softmax(logits_teacher.detach(), dim=1)
        return F.kl_div(log_p_s, p_t, reduction="batchmean")

    @abstractmethod
    def total(self, *args, **kwargs):
        """Compute total loss. Signature varies by subclass."""
        ...


# ---------------------------------------------------------------------------
# Vision-only: cls + distillation
# ---------------------------------------------------------------------------

class VisionOnlyLoss(BaseLoss):
    """L = L_cls [+ w_sd * L_sd]"""

    def total(self, batch, outputs_s, outputs_t=None):
        """
        Args:
            batch:      Standard data batch from DataLoader.
            outputs_s:  Student model output dict.
            outputs_t:  Teacher model output dict or None.

        Returns:
            (loss, parts_dict)
        """
        logits_s = outputs_s["os_logits"]
        logits_t = outputs_t["os_logits"] if outputs_t else None
        y = batch["label"].to(logits_s.device)
        l_cls = self.classification(logits_s, y)
        parts = {"cls": l_cls.item()}
        total = l_cls

        if logits_t is not None and self.w_sd > 0:
            l_sd = self.distill(logits_t, logits_s)
            total = total + self.w_sd * l_sd
            parts["sd"] = l_sd.item()

        return total, parts


# ---------------------------------------------------------------------------
# Multimodal: cls + distillation + alignment
# ---------------------------------------------------------------------------

class MultimodalLoss(BaseLoss):
    """L = L_cls [+ w_sd * L_sd] + w_align * L_align"""

    def __init__(self, w_align: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.w_align = w_align

    def align(self, f_v, f_t):
        """Cosine alignment between vision and text features."""
        f_vn = F.normalize(f_v, dim=1)
        f_tn = F.normalize(f_t, dim=1)
        return 1.0 - (f_vn * f_tn).sum(dim=1).mean()

    def total(self, batch, outputs_s, outputs_t=None):
        """
        Args:
            batch:      Standard data batch from DataLoader.
            outputs_s:  Student model output dict.
            outputs_t:  Teacher model output dict or None.

        Returns:
            (loss, parts_dict)
        """
        logits_s = outputs_s["os_logits"]
        logits_t = outputs_t["os_logits"] if outputs_t else None
        y = batch["label"].to(logits_s.device)
        f_v = outputs_s["f_v"]
        f_t = outputs_s["f_t"]
        l_cls = self.classification(logits_s, y)
        parts = {"cls": l_cls.item()}
        total = l_cls

        if logits_t is not None and self.w_sd > 0:
            l_sd = self.distill(logits_t, logits_s)
            total = total + self.w_sd * l_sd
            parts["sd"] = l_sd.item()

        if self.w_align > 0:
            l_align = self.align(f_v, f_t)
            total = total + self.w_align * l_align
            parts["align"] = l_align.item()

        return total, parts


# ---------------------------------------------------------------------------
# Multitask: cls + distillation + segmentation
# ---------------------------------------------------------------------------

class MultitaskLoss(BaseLoss):
    """L = L_cls [+ w_sd * L_sd] + w_seg * L_seg"""

    def __init__(self, w_seg: float = 1.0, seg_loss_name: str = "dice_ce", **kwargs):
        super().__init__(**kwargs)
        self.w_seg = w_seg
        self.seg_loss = _build_seg_loss(seg_loss_name)

    def segmentation(self, seg_logits, seg_mask):
        """Dense segmentation loss.

        Args:
            seg_logits: Model predictions  [B, C, D, H, W].
            seg_mask:   Ground-truth masks [B, D, H, W].
        """
        # MONAI expects target with channel dim: [B, 1, D, H, W]
        seg_mask = seg_mask.unsqueeze(1)
        return self.seg_loss(seg_logits, seg_mask)

    def total(self, batch, outputs_s, outputs_t=None):
        """
        Args:
            batch:      Standard data batch from DataLoader.
            outputs_s:  Student model output dict.
            outputs_t:  Teacher model output dict or None.

        Returns:
            (loss, parts_dict)
        """
        logits_s = outputs_s["os_logits"]
        logits_t = outputs_t["os_logits"] if outputs_t else None
        y = batch["label"].to(logits_s.device)
        seg_logits = outputs_s["seg_logits"]
        seg_mask = batch["seg_mask"].to(logits_s.device)
        l_cls = self.classification(logits_s, y)
        parts = {"cls": l_cls.item()}
        total = l_cls

        if logits_t is not None and self.w_sd > 0:
            l_sd = self.distill(logits_t, logits_s)
            total = total + self.w_sd * l_sd
            parts["sd"] = l_sd.item()

        if self.w_seg > 0:
            l_seg = self.segmentation(seg_logits, seg_mask)
            total = total + self.w_seg * l_seg
            parts["seg"] = l_seg.item()

        return total, parts


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Helper to safely read config values (same pattern as engine._cfg_get)
def _loss_cfg_get(cfg, key, default=None):
    """Read a nested key from OmegaConf or dict config."""
    try:
        from omegaconf import OmegaConf
        return OmegaConf.select(cfg, key, default=default)
    except (ImportError, AttributeError):
        keys = key.split(".")
        val = cfg
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val


def build_loss(cfg):
    """Build the appropriate loss object based on the training mode.

    Reads from the ``loss`` config block:
        loss.classification:  "cross_entropy" | "weighted_ce"
        loss.segmentation:    "dice_ce" | "dice_focal"
        loss.w_sd:            float  (self-distillation weight)
        loss.w_seg:           float  (segmentation weight, multitask only)
        loss.w_align:         float  (alignment weight, multimodal only)
        loss.class_weights:   list[float]  (for weighted_ce only)

    Falls back to top-level keys (``w_sd``, ``w_align``) for backward
    compatibility with existing configs.

    Args:
        cfg: OmegaConf or dict config.

    Returns:
        An instance of VisionOnlyLoss, MultimodalLoss, or MultitaskLoss.
    """
    mode = _loss_cfg_get(cfg, "mode", "vision_only")

    # Read from loss.* block, falling back to top-level keys for compatibility
    cls_loss_name = _loss_cfg_get(cfg, "loss.classification", "cross_entropy")
    w_sd = float(_loss_cfg_get(cfg, "loss.w_sd", _loss_cfg_get(cfg, "w_sd", 0.5)))

    # Optional class weights (for weighted_ce)
    class_weights = _loss_cfg_get(cfg, "loss.class_weights")

    # Shared kwargs for BaseLoss.__init__
    base_kwargs = {
        "cls_loss_name": cls_loss_name,
        "w_sd": w_sd,
    }
    if class_weights is not None:
        base_kwargs["class_weights"] = list(class_weights)

    if mode == "multimodal":
        w_align = float(_loss_cfg_get(cfg, "loss.w_align", _loss_cfg_get(cfg, "w_align", 1.0)))
        loss_obj = MultimodalLoss(w_align=w_align, **base_kwargs)

    elif mode == "multitask":
        seg_loss_name = _loss_cfg_get(cfg, "loss.segmentation", "dice_ce")
        w_seg = float(_loss_cfg_get(cfg, "loss.w_seg", _loss_cfg_get(cfg, "w_seg", 1.0)))
        loss_obj = MultitaskLoss(w_seg=w_seg, seg_loss_name=seg_loss_name, **base_kwargs)

    else:
        # vision_only or any unknown mode
        loss_obj = VisionOnlyLoss(**base_kwargs)

    logger.info(
        f"Loss: {type(loss_obj).__name__} | cls={cls_loss_name} | w_sd={w_sd}"
    )
    return loss_obj
