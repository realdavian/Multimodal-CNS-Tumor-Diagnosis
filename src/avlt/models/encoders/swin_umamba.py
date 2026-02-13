"""
Swin-UMamba Vision Encoder (2D Slice-Wise).

Uses the VMamba-Tiny backbone (VSSMEncoder) from the Swin-UMamba repository
to process 3D volumes as a sequence of 2D slices.

Requirements:
    - mamba-ssm (CUDA selective scan kernels)
    - causal-conv1d
    - einops
    - Swin-UMamba repo cloned at: src/avlt/models/external/Swin-UMamba/

Pipeline:
    Input [B, 4, D, H, W]
    → Reshape to [B*D, 4, H, W]
    → VSSMEncoder (VMamba backbone with 4 stages)
    → Extract deepest feature map [B*D, 768, H', W']
    → Global Average Pool → [B*D, 768]
    → Reshape [B, D, 768]
    → SliceAttention aggregation [B, 768]
    → Projection [B, out_dim]

NOTE: VSSMEncoder is a 2D model from the Swin-UMamba paper.
      We process each slice independently, then aggregate with attention.
      This is the 2.5D approach — see swin3d.py for full 3D.
"""

import re
import os
import sys
import types
import importlib.util
import torch
import torch.nn as nn

# ── Load VSSMEncoder from the Swin-UMamba codebase ───────────────────────
# The Swin-UMamba repo was installed via `pip install -e .` which registers
# 'nnunetv2' as a package. However, it's missing the real nnU-Net framework's
# plans_handler module. We stub that out before importing VSSMEncoder.

def _ensure_plans_handler_stub():
    """
    Stub out nnunetv2.utilities.plans_handling.plans_handler if not importable.
    SwinUMamba.py imports ConfigurationManager and PlansManager from this module,
    but they're only used in factory functions we don't call.
    """
    try:
        from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager
        return  # Already available, no stub needed
    except (ImportError, ModuleNotFoundError):
        pass

    # Create stub package hierarchy with __path__ so sub-imports resolve
    for mod_name in [
        "nnunetv2",
        "nnunetv2.utilities",
        "nnunetv2.utilities.plans_handling",
        "nnunetv2.utilities.plans_handling.plans_handler",
    ]:
        if mod_name not in sys.modules:
            mod = types.ModuleType(mod_name)
            mod.__path__ = []
            mod.__package__ = mod_name
            sys.modules[mod_name] = mod

    # Add dummy classes to the plans_handler stub
    handler = sys.modules["nnunetv2.utilities.plans_handling.plans_handler"]
    handler.ConfigurationManager = type("ConfigurationManager", (), {})
    handler.PlansManager = type("PlansManager", (), {})

_ensure_plans_handler_stub()

# Now import VSSMEncoder — try the installed package first, then fallback to importlib
try:
    from nnunetv2.nets.SwinUMamba import VSSMEncoder
except (ImportError, ModuleNotFoundError):
    # Fallback: load directly from cloned source file
    _src = os.path.join(
        os.path.dirname(__file__), "..", "external",
        "Swin-UMamba", "swin_umamba", "nnunetv2", "nets", "SwinUMamba.py"
    )
    _src = os.path.abspath(_src)
    _spec = importlib.util.spec_from_file_location("_swin_umamba_src", _src)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    VSSMEncoder = _mod.VSSMEncoder

from ..layers import SliceAttention


def _load_pretrained_weights(encoder: VSSMEncoder, ckpt_path: str):
    """
    Load VMamba pretrained weights into VSSMEncoder.
    Adapts the official load_pretrained_ckpt() to work with the standalone
    encoder (no full SwinUMamba wrapper).

    NOTE: The pretrained checkpoint has keys like 'layers.0.blocks.0...'
          while VSSMEncoder separates downsamples into a separate ModuleList.
          The official loader handles this remapping for us.
    """
    if not os.path.exists(ckpt_path):
        print(f"WARNING: VMamba pretrained weights not found at {ckpt_path}")
        return encoder

    print(f"Loading VMamba pretrained weights from {ckpt_path}")
    skip_params = [
        "norm.weight", "norm.bias",
        "head.weight", "head.bias",
        "patch_embed.proj.weight", "patch_embed.proj.bias",
        "patch_embed.norm.weight", "patch_embed.norm.bias",
    ]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_dict = encoder.state_dict()
    loaded, skipped = 0, 0

    for k, v in ckpt["model"].items():
        if k in skip_params:
            skipped += 1
            continue

        # Remap downsample keys: 'layers.X.downsample' → 'downsamples.X'
        kr = k
        if "downsample" in kr:
            i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
            kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")

        if kr in model_dict and v.shape == model_dict[kr].shape:
            model_dict[kr] = v
            loaded += 1
        else:
            skipped += 1

    encoder.load_state_dict(model_dict)
    print(f"VMamba weights loaded: {loaded} params loaded, {skipped} skipped")
    return encoder


class SwinUMambaVisionEncoder(nn.Module):
    """
    2D Swin-UMamba vision encoder with slice-wise processing.

    Uses VSSMEncoder (VMamba backbone) to process each slice of a 3D volume
    independently, then aggregates slice features using SliceAttention.

    Args:
        in_channels: Number of input modalities (4 for T1, T1c, T2, FLAIR)
        out_dim: Output feature dimension (768 to match text encoder)
        pretrained_path: Path to VMamba-Tiny pretrained weights
        image_size: Spatial resolution of each slice (accepted for API compat)
    """

    def __init__(
        self,
        in_channels=4,
        out_dim=768,
        pretrained_path="data/pretrained/vmamba/vmamba_tiny_e292.pth",
        # NOTE: These params are accepted for factory API compatibility
        image_size=224,
        backbone=None,
        pretrained=None,
        cnn_stem=None,
        **kwargs,
    ):
        super().__init__()

        # VMamba-Tiny configuration (matches paper)
        # Stages produce feature maps with dims [96, 192, 384, 768]
        self.backbone = VSSMEncoder(
            patch_size=4,         # 4×4 patch embedding
            in_chans=in_channels, # 4 MRI modalities
            depths=[2, 2, 9, 2],  # VMamba-Tiny block depths
            dims=[96, 192, 384, 768],  # Channel dims per stage
        )

        # Load ImageNet-pretrained VMamba-Tiny weights
        # NOTE: in_chans=4 vs pretrained in_chans=3 means patch_embed weights
        #       won't match — they are skipped automatically
        _load_pretrained_weights(self.backbone, pretrained_path)

        # VSSMEncoder's deepest stage outputs 768 channels
        backbone_dim = 768

        # Slice aggregation via learnable attention (shared module)
        self.slice_attn = SliceAttention(backbone_dim)

        # Final projection (identity if out_dim == backbone_dim)
        self.proj = nn.Linear(backbone_dim, out_dim) if out_dim != backbone_dim else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [B, C, D, H, W] — 3D multi-modal MRI volume

        Returns:
            [B, out_dim] — aggregated feature vector
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input [B, C, D, H, W], got {x.shape}")

        B, C, D, H, W = x.shape

        # ── Step 1: Reshape for 2D slice processing ──────────────────
        # [B, C, D, H, W] → [B*D, C, H, W]
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        # ── Step 2: Extract features via VSSMEncoder ─────────────────
        # VSSMEncoder.forward() returns a list of 5 feature maps:
        #   [0]: original input  [B*D, C, H, W]
        #   [1]: stage 1 output  [B*D, 96,  H/4,  W/4]
        #   [2]: stage 2 output  [B*D, 192, H/8,  W/8]
        #   [3]: stage 3 output  [B*D, 384, H/16, W/16]
        #   [4]: stage 4 output  [B*D, 768, H/32, W/32]  ← we use this
        feature_maps = self.backbone(x_2d)
        deep_feat = feature_maps[-1]  # [B*D, 768, H', W']

        # ── Step 3: Global Average Pool per slice ────────────────────
        # [B*D, 768, H', W'] → [B*D, 768]
        feat = deep_feat.mean(dim=[-2, -1])

        # ── Step 4: Reshape to sequence of slice features ────────────
        # [B*D, 768] → [B, D, 768]
        feat = feat.view(B, D, -1)

        # ── Step 5: Aggregate slices via attention ───────────────────
        # [B, D, 768] → [B, 768]
        feat = self.slice_attn(feat)

        # ── Step 6: Project to output dimension ─────────────────────
        return self.proj(feat)  # [B, out_dim]
