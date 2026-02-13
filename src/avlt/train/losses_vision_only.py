"""
Loss functions for the vision-only pipeline.

Keeps classification (CE) and teacher→student self-distillation (KL-div).
Drops the cross-modal alignment loss (requires text features).
"""

import torch.nn as nn
import torch.nn.functional as F


class VisionOnlyLosses:
    """L_total = L_cls + w_sd * L_sd"""

    def __init__(self, w_sd: float = 0.5):
        self.w_sd = w_sd
        self.ce = nn.CrossEntropyLoss()

    # ---- individual terms ----

    def classification(self, logits, y):
        """Standard cross-entropy."""
        return self.ce(logits, y)

    def distill(self, logits_teacher, logits_student):
        """KL-divergence self-distillation (teacher is detached)."""
        log_p_s = F.log_softmax(logits_student, dim=1)
        p_t = F.softmax(logits_teacher.detach(), dim=1)
        return F.kl_div(log_p_s, p_t, reduction="batchmean")

    # ---- composite ----

    def total(self, logits_student, logits_teacher, y):
        """
        Returns:
            loss:  scalar tensor
            parts: dict of individual loss values (for logging)
        """
        l_cls = self.classification(logits_student, y)
        l_sd = self.distill(logits_teacher, logits_student)
        loss = l_cls + self.w_sd * l_sd
        return loss, {"cls": l_cls.item(), "sd": l_sd.item()}
