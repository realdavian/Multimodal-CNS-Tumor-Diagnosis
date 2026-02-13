"""
3D Swin Transformer Vision Encoder using MONAI's SwinUNETR pretrained encoder.

Architecture (paper Section 2.3):
    Input [B, C, D, H, W]
    → MONAI SwinTransformer (3D cubic patches, shifted-window attention)
    → Hierarchical 3D feature maps at multiple scales
    → Adaptive 3D global average pooling on deepest features
    → Linear projection → [B, out_dim]

NOTE: Uses SwinUNETR's encoder (swinViT) which has pretrained weights from
self-supervised learning on 5,050+ CT/MRI volumes. We discard the decoder
since we only need features for classification, not segmentation.
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class Swin3DVisionEncoder(nn.Module):
    """
    3D Swin Transformer vision encoder extracted from MONAI's SwinUNETR.

    Pipeline:
        [B, 4, D, H, W] → SwinUNETR.swinViT → hierarchical 3D features
                         → AdaptiveAvgPool3d on deepest layer → [B, C']
                         → Linear → [B, out_dim]

    Args:
        in_channels: Number of input channels (4 for T1, T1c, T2, FLAIR)
        out_dim: Output feature dimension (768 to match text encoder)
        feature_size: Base channel count; deeper layers multiply this
                      (48 → stages produce 48, 96, 192, 384, 768)
        spatial_dims: Number of spatial dimensions (3 for volumetric data)
        use_v2: Whether to use Swin Transformer V2 (improved stability)
    """

    def __init__(
        self,
        in_channels=4,
        out_dim=768,
        feature_size=48,
        spatial_dims=3,
        use_v2=False,
        # NOTE: These params are accepted but ignored for API compatibility
        #       with the factory function in __init__.py
        image_size=None,
        backbone=None,
        pretrained=None,
        cnn_stem=None,
    ):
        super().__init__()

        # Build full SwinUNETR, then extract only the encoder
        # NOTE: out_channels=1 is a dummy value — we never use the decoder
        # NOTE: use_checkpoint=True enables gradient checkpointing to save memory
        #       at the cost of ~30% slower training — essential for 3D volumes
        swin_unetr = SwinUNETR(
            in_channels=in_channels,
            out_channels=1,
            feature_size=feature_size,
            spatial_dims=spatial_dims,
            use_v2=use_v2,
            img_size=image_size,
            use_checkpoint=True,
        )

        # Extract the Swin Transformer encoder (discard UNet decoder)
        self.encoder = swin_unetr.swinViT

        # The deepest swinViT output has feature_size * 16 channels
        # e.g., feature_size=48 → deepest = 48 * 16 = 768
        encoder_out_dim = feature_size * 16

        # Pool 3D feature maps to a single vector per sample
        # Swin3D naturally outputs [B, 768] via global pooling, 
        # but if we wanted to be consistent with 'slice_wise' aggregation we could use SliceAttention.
        # However, Swin3D is fully volumetric, so standard GAP is correct.
        # The 'SliceAttention' class in this file was unused in previous steps (Swin3D didn't use it).
        # We will stick to standard GAP for Swin3D as per paper/MONAI.
        self.pool = nn.AdaptiveAvgPool3d(1)

        # Project to desired output dimension
        self.proj = nn.Linear(encoder_out_dim, out_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, C, D, H, W]

        Returns:
            Feature tensor of shape [B, out_dim]
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input [B, C, D, H, W], got {x.shape}")

        # swinViT returns a list of hierarchical feature maps
        hidden_states = self.encoder(x)

        # Use the deepest feature map
        deep_features = hidden_states[-1]  # [B, 768, D', H', W']

        # Global average pool → [B, 768, 1, 1, 1] → squeeze → [B, 768]
        pooled = self.pool(deep_features).flatten(1)

        # Project to output dimension
        return self.proj(pooled)  # [B, out_dim]
