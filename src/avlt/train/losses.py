"""
Unified loss functions for both vision-only and multimodal training.

- Classification: cross-entropy
- Self-distillation: KL-divergence (student → teacher)
- Alignment: cosine similarity between vision and text features (multimodal only)

The alignment loss is automatically skipped when text features are not provided
(i.e., vision-only mode), so the same class works for both pipelines.
"""

import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss

class Losses:
    """Composite loss: L = L_cls + w_sd * L_sd [+ w_align * L_align] [+ w_seg * L_seg].

    Args:
        w_align: Weight for cross-modal alignment loss (0.0 disables it).
        w_sd: Weight for self-distillation KL loss.
        w_seg: Weight for volumetric semantic segmentation loss.
    """

    def __init__(self, w_align: float = 0.0, w_sd: float = 0.5, w_seg: float = 1.0):
        self.w_align = w_align
        self.w_sd = w_sd
        self.w_seg = w_seg
        self.ce = nn.CrossEntropyLoss()
        
        # DiceCELoss handles dense mask segmentation directly:
        # include_background=False ignores the background mask during Dice evaluation
        # to focus entirely on tumor subregions. to_onehot_y automatically manages the one-hot target.
        self.dice_ce = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)

    # ---- individual terms ------------------------------------------------

    def classification(self, logits, y):
        """Standard cross-entropy."""
        return self.ce(logits, y)

    def distill(self, logits_teacher, logits_student):
        """KL-divergence self-distillation (teacher is detached)."""
        log_p_s = F.log_softmax(logits_student, dim=1)
        p_t = F.softmax(logits_teacher.detach(), dim=1)
        return F.kl_div(log_p_s, p_t, reduction="batchmean")

    def align(self, f_v, f_t):
        """Cosine alignment between vision and text features."""
        f_vn = F.normalize(f_v, dim=1)
        f_tn = F.normalize(f_t, dim=1)
        return 1.0 - (f_vn * f_tn).sum(dim=1).mean()
        
    def segmentation(self, seg_logits, seg_mask):
        """Dice + Cross entropy segmentation loss."""
        # seg_logits: [B, C, D, H, W]
        # seg_mask:   [B, D, H, W] (needs channel dim added for MONAI)
        seg_mask = seg_mask.unsqueeze(1) # [B, 1, D, H, W]
        return self.dice_ce(seg_logits, seg_mask)

    # ---- composite -------------------------------------------------------

    def total(self, logits_s, logits_t, y, f_v=None, f_t=None, seg_logits_s=None, seg_mask=None):
        """Compute total loss.

        Args:
            logits_s: Student logits.
            logits_t: Teacher logits (or None if distillation disabled).
            y: Ground-truth OS labels.
            f_v: Vision features (optional, for alignment).
            f_t: Text features (optional, for alignment).
            seg_logits_s: Dense Multitask segmentation predictions [B, C, D, H, W]
            seg_mask: Ground-truth semantic masks [B, D, H, W]

        Returns:
            loss: Scalar tensor.
            parts: Dict of individual loss values for logging.
        """
        l_cls = self.classification(logits_s, y)
        parts = {"cls": l_cls.item()}
        total = l_cls

        # Self-distillation
        if logits_t is not None and self.w_sd > 0:
            l_sd = self.distill(logits_t, logits_s)
            total = total + self.w_sd * l_sd
            parts["sd"] = l_sd.item()
            
        # Multitask Dense Segmentation
        if seg_logits_s is not None and seg_mask is not None and self.w_seg > 0:
            l_seg = self.segmentation(seg_logits_s, seg_mask)
            total = total + self.w_seg * l_seg
            parts["seg"] = l_seg.item()

        # Cross-modal alignment (multimodal only)
        if f_v is not None and f_t is not None and self.w_align > 0:
            l_align = self.align(f_v, f_t)
            total = total + self.w_align * l_align
            parts["align"] = l_align.item()

        return total, parts
