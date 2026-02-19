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


class Losses:
    """Composite loss: L = L_cls + w_sd * L_sd [+ w_align * L_align].

    Args:
        w_align: Weight for cross-modal alignment loss (0.0 disables it).
        w_sd: Weight for self-distillation KL loss.
    """

    def __init__(self, w_align: float = 0.0, w_sd: float = 0.5):
        self.w_align = w_align
        self.w_sd = w_sd
        self.ce = nn.CrossEntropyLoss()

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

    # ---- composite -------------------------------------------------------

    def total(self, logits_s, logits_t, y, f_v=None, f_t=None):
        """Compute total loss.

        Args:
            logits_s: Student logits.
            logits_t: Teacher logits (or None if distillation disabled).
            y: Ground-truth labels.
            f_v: Vision features (optional, for alignment).
            f_t: Text features (optional, for alignment).

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

        # Cross-modal alignment (multimodal only)
        if f_v is not None and f_t is not None and self.w_align > 0:
            l_align = self.align(f_v, f_t)
            total = total + self.w_align * l_align
            parts["align"] = l_align.item()

        return total, parts
