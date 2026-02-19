"""
Self-distillation via Exponential Moving Average (EMA) teacher.

Isolates the student-teacher training pattern so the engine stays clean.
Can be disabled by setting `enabled=False` or swapped for a different
training strategy without touching the main loop.
"""

import copy
import torch
import torch.nn as nn


class SelfDistillation:
    """Manages an EMA teacher model for self-distillation.

    Usage::

        distiller = SelfDistillation(student_model, momentum=0.999)
        for batch in dataloader:
            logits_s = student_model(x)
            logits_t = distiller.forward(x)       # teacher inference (no grad)
            loss = ...  # use logits_t for KL-div
            loss.backward(); opt.step()
            distiller.update()                     # EMA step

    Args:
        student: The student model (nn.Module).
        momentum: EMA decay factor (0.999 = slow teacher update).
        device: Device to place teacher on.
        enabled: If False, forward() returns None and update() is a no-op.
    """

    def __init__(
        self,
        student: nn.Module,
        momentum: float = 0.999,
        device: str = "cpu",
        enabled: bool = True,
    ):
        self.momentum = momentum
        self.enabled = enabled
        self.teacher = None

        if enabled:
            # Deep-copy student weights into teacher
            self.teacher = copy.deepcopy(self._unwrap(student)).to(device)
            for p in self.teacher.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        """Run teacher inference. Returns None if distillation is disabled."""
        if not self.enabled:
            return None
        self.teacher.eval()
        return self.teacher(*args, **kwargs)

    @torch.no_grad()
    def update(self, student: nn.Module):
        """EMA update: teacher ← m·teacher + (1−m)·student."""
        if not self.enabled:
            return
        m = self.momentum
        s_state = self._unwrap(student).state_dict()
        t_state = self._unwrap(self.teacher).state_dict()
        for key in t_state:
            t_state[key].copy_(m * t_state[key] + (1 - m) * s_state[key])

    def wrap_parallel(self, device_ids=None):
        """Wrap teacher in DataParallel if needed."""
        if self.enabled and self.teacher is not None:
            self.teacher = nn.DataParallel(self.teacher, device_ids=device_ids)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(model):
        """Handle DataParallel wrapper."""
        return model.module if hasattr(model, "module") else model
