"""
Unit tests for the modular loss module.

Tests cover:
    - Factory instantiation for all modes
    - Forward pass and gradient flow for each loss subclass
    - Loss function registries (cross_entropy, dice_ce, dice_focal)
    - Backward compatibility with top-level config keys
    - Error handling for unknown loss names
"""

import pytest
import torch
from omegaconf import OmegaConf

from avlt.train.losses import (
    BaseLoss,
    VisionOnlyLoss,
    MultimodalLoss,
    MultitaskLoss,
    build_loss,
    _build_cls_loss,
    _build_seg_loss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch():
    """Minimal batch tensors for testing."""
    return {
        "label": torch.tensor([0, 2]),
        "seg_mask": torch.zeros(2, 8, 8, 8, dtype=torch.long),
    }

@pytest.fixture
def outputs_s():
    """Simulated student output dict."""
    return {
        "os_logits": torch.randn(2, 3, requires_grad=True),
        "f_v": torch.randn(2, 768),
        "f_t": torch.randn(2, 768),
        "seg_logits": torch.randn(2, 4, 8, 8, 8, requires_grad=True),
    }

@pytest.fixture
def outputs_t():
    """Simulated teacher output dict."""
    return {
        "os_logits": torch.randn(2, 3),
    }

@pytest.fixture
def batch_with_seg(batch):
    """Seg mask with realistic label distribution (non-trivial)."""
    mask = batch["seg_mask"].clone()
    mask[:, 3:5, 3:5, 3:5] = 1  # Necrotic
    mask[:, 2:4, 2:4, 2:4] = 2  # Edema
    mask[:, 6:7, 6:7, 6:7] = 3  # Enhancing
    batch["seg_mask"] = mask
    return batch


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistries:
    """Test individual loss function registries."""

    def test_build_cls_cross_entropy(self):
        loss_fn = _build_cls_loss("cross_entropy")
        assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)

    def test_build_cls_weighted_ce(self):
        loss_fn = _build_cls_loss("weighted_ce", class_weights=[1.0, 1.2, 0.8])
        assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)
        assert loss_fn.weight is not None
        assert loss_fn.weight.shape == (3,)

    def test_build_cls_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown classification loss"):
            _build_cls_loss("nonexistent")

    def test_build_seg_dice_ce(self):
        from monai.losses import DiceCELoss
        loss_fn = _build_seg_loss("dice_ce")
        assert isinstance(loss_fn, DiceCELoss)

    def test_build_seg_dice_focal(self):
        from monai.losses import DiceFocalLoss
        loss_fn = _build_seg_loss("dice_focal")
        assert isinstance(loss_fn, DiceFocalLoss)

    def test_build_seg_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown segmentation loss"):
            _build_seg_loss("nonexistent")


# ---------------------------------------------------------------------------
# VisionOnlyLoss tests
# ---------------------------------------------------------------------------

class TestVisionOnlyLoss:

    def test_instantiation(self):
        loss = VisionOnlyLoss(cls_loss_name="cross_entropy", w_sd=0.5)
        assert loss.w_sd == 0.5

    def test_total_with_distillation(self, batch, outputs_s, outputs_t):
        loss = VisionOnlyLoss(cls_loss_name="cross_entropy", w_sd=0.5)
        total, parts = loss.total(batch, outputs_s, outputs_t)

        assert "cls" in parts
        assert "sd" in parts
        assert total.requires_grad

    def test_total_without_distillation(self, batch, outputs_s):
        """When logits_t is None, only classification loss should be present."""
        loss = VisionOnlyLoss(cls_loss_name="cross_entropy", w_sd=0.5)
        total, parts = loss.total(batch, outputs_s, None)

        assert "cls" in parts
        assert "sd" not in parts

    def test_total_sd_disabled(self, batch, outputs_s, outputs_t):
        """When w_sd=0, distillation term should be skipped."""
        loss = VisionOnlyLoss(cls_loss_name="cross_entropy", w_sd=0.0)
        total, parts = loss.total(batch, outputs_s, outputs_t)

        assert "sd" not in parts

    def test_gradient_flow(self, batch, outputs_s, outputs_t):
        loss = VisionOnlyLoss(cls_loss_name="cross_entropy", w_sd=0.5)
        total, _ = loss.total(batch, outputs_s, outputs_t)
        total.backward()
        assert outputs_s["os_logits"].grad is not None


# ---------------------------------------------------------------------------
# MultimodalLoss tests
# ---------------------------------------------------------------------------

class TestMultimodalLoss:

    def test_instantiation(self):
        loss = MultimodalLoss(w_align=1.0, cls_loss_name="cross_entropy", w_sd=0.5)
        assert loss.w_align == 1.0

    def test_total_includes_alignment(self, batch, outputs_s, outputs_t):
        loss = MultimodalLoss(w_align=1.0, cls_loss_name="cross_entropy", w_sd=0.5)
        total, parts = loss.total(batch, outputs_s, outputs_t)

        assert "cls" in parts
        assert "sd" in parts
        assert "align" in parts

    def test_alignment_value_range(self, batch, outputs_s):
        """Cosine alignment should be in [0, 2]."""
        loss = MultimodalLoss(w_align=1.0, cls_loss_name="cross_entropy", w_sd=0.0)
        _, parts = loss.total(batch, outputs_s, None)
        assert 0.0 <= parts["align"] <= 2.0

    def test_align_disabled(self, batch, outputs_s):
        loss = MultimodalLoss(w_align=0.0, cls_loss_name="cross_entropy", w_sd=0.0)
        _, parts = loss.total(batch, outputs_s, None)
        assert "align" not in parts


# ---------------------------------------------------------------------------
# MultitaskLoss tests
# ---------------------------------------------------------------------------

class TestMultitaskLoss:

    def test_instantiation(self):
        loss = MultitaskLoss(w_seg=1.0, seg_loss_name="dice_ce", cls_loss_name="cross_entropy", w_sd=0.5)
        assert loss.w_seg == 1.0

    def test_total_includes_segmentation(self, batch_with_seg, outputs_s, outputs_t):
        loss = MultitaskLoss(w_seg=1.0, seg_loss_name="dice_ce", cls_loss_name="cross_entropy", w_sd=0.5)
        total, parts = loss.total(batch_with_seg, outputs_s, outputs_t)

        assert "cls" in parts
        assert "sd" in parts
        assert "seg" in parts
        assert total.requires_grad

    def test_seg_disabled(self, batch_with_seg, outputs_s):
        loss = MultitaskLoss(w_seg=0.0, seg_loss_name="dice_ce", cls_loss_name="cross_entropy", w_sd=0.0)
        _, parts = loss.total(batch_with_seg, outputs_s, None)
        assert "seg" not in parts

    def test_dice_focal_variant(self, batch_with_seg, outputs_s):
        loss = MultitaskLoss(w_seg=1.0, seg_loss_name="dice_focal", cls_loss_name="cross_entropy", w_sd=0.0)
        total, parts = loss.total(batch_with_seg, outputs_s, None)
        assert "seg" in parts
        assert total.item() > 0

    def test_gradient_flows_to_seg(self, batch_with_seg, outputs_s):
        loss = MultitaskLoss(w_seg=1.0, seg_loss_name="dice_ce", cls_loss_name="cross_entropy", w_sd=0.0)
        total, _ = loss.total(batch_with_seg, outputs_s, None)
        total.backward()
        assert outputs_s["seg_logits"].grad is not None


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestBuildLoss:

    def test_vision_only(self):
        cfg = OmegaConf.create({"mode": "vision_only"})
        loss = build_loss(cfg)
        assert isinstance(loss, VisionOnlyLoss)

    def test_multimodal(self):
        cfg = OmegaConf.create({"mode": "multimodal"})
        loss = build_loss(cfg)
        assert isinstance(loss, MultimodalLoss)

    def test_multitask(self):
        cfg = OmegaConf.create({"mode": "multitask"})
        loss = build_loss(cfg)
        assert isinstance(loss, MultitaskLoss)

    def test_unknown_mode_defaults_to_vision_only(self):
        cfg = OmegaConf.create({"mode": "unknown_mode"})
        loss = build_loss(cfg)
        assert isinstance(loss, VisionOnlyLoss)

    def test_loss_block_config(self):
        cfg = OmegaConf.create({
            "mode": "multitask",
            "loss": {
                "classification": "cross_entropy",
                "segmentation": "dice_focal",
                "w_sd": 0.3,
                "w_seg": 0.7,
            }
        })
        loss = build_loss(cfg)
        assert isinstance(loss, MultitaskLoss)
        assert loss.w_sd == 0.3
        assert loss.w_seg == 0.7

    def test_backward_compat_top_level_keys(self):
        """Top-level w_sd/w_align should work when loss.* block is absent."""
        cfg = OmegaConf.create({
            "mode": "multimodal",
            "w_sd": 0.3,
            "w_align": 2.0,
        })
        loss = build_loss(cfg)
        assert isinstance(loss, MultimodalLoss)
        assert loss.w_sd == 0.3
        assert loss.w_align == 2.0

    def test_loss_block_overrides_top_level(self):
        """loss.* block should take precedence over top-level keys."""
        cfg = OmegaConf.create({
            "mode": "vision_only",
            "w_sd": 0.9,        # top-level (lower priority)
            "loss": {
                "w_sd": 0.1,    # loss block (higher priority)
            }
        })
        loss = build_loss(cfg)
        assert loss.w_sd == 0.1

    def test_full_base_yaml(self):
        """Verify factory works with the actual base.yaml config."""
        base_cfg = OmegaConf.load("configs/base.yaml")
        # Override mode to test each variant
        for mode in ["vision_only", "multimodal", "multitask"]:
            cfg = OmegaConf.merge(base_cfg, {"mode": mode})
            loss = build_loss(cfg)
            assert loss is not None
