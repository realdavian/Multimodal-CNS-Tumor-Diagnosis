"""
Vision-only variant of the AVLT model.

Strips the language-encoding branch and cross-attention fusion,
keeping only the vision encoder → classifier head.
Compatible with the teacher-student self-distillation training loop.
"""

import torch.nn as nn
from .encoders import create_vision_encoder


class AVLTVisionOnly(nn.Module):
    """Vision encoder + classification head (no text branch, no fusion)."""

    def __init__(
        self,
        num_classes: int = 2,
        image_size: int = 224,
        backbone: str = "vit_base_patch16_224",
        dropout: float = 0.3,
        vision_variant: str = "fixed",
    ):
        super().__init__()
        self.vision = create_vision_encoder(
            variant=vision_variant,
            image_size=image_size,
            backbone=backbone,
            pretrained=True,
            cnn_stem=True,
            out_dim=768,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(768, num_classes)

    def forward(self, images):
        """
        Args:
            images: (B, C, D, H, W) 3-D volume or (B, C, H, W) 2-D image.

        Returns:
            dict: {
                "os_logits": (B, num_classes)
                "f_v":       (B, 768) vision features
            }
        """
        f_v = self.vision(images)
        logits = self.head(self.dropout(f_v))
        return {
            "os_logits": logits,
            "f_v": f_v
        }
