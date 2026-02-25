"""
Multitask variant of the AVLT model.

Vision-only model with segmentation decoder included.
Outputs both classification logits and segmentation masks.
Compatible with the teacher-student self-distillation training loop.
"""

import torch.nn as nn
from .encoders import create_vision_encoder

class AVLTVisionMultitask(nn.Module):
    """Vision encoder + classification head + segmentation decoder."""

    def __init__(
        self,
        num_classes: int = 3,           # Survival classification (e.g. 3 classes)
        num_seg_classes: int = 4,       # BraTS Tumor Segmentation (e.g. 4 classes)
        image_size: int = 224,
        backbone: str = "vit_base_patch16_224",
        dropout: float = 0.3,
        vision_variant: str = "swin3d_multitask",
    ):
        super().__init__()
        # Ensure we pass the requested seg classes
        self.vision = create_vision_encoder(
            variant=vision_variant,
            image_size=image_size,
            backbone=backbone,
            pretrained=True,
            cnn_stem=True,
            out_dim=768,
            num_seg_classes=num_seg_classes,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(768, num_classes)

    def forward(self, images):
        """
        Args:
            images: (B, C, D, H, W) 3-D volume.

        Returns:
            dict: {
                "os_logits":  (B, num_classes)
                "seg_logits": (B, num_seg_classes, D, H, W)
                "f_v":        (B, 768)
            }
        """
        # Multitask Vision Encoder returns extracted feature array + seg mask logits
        f_v, seg_logits = self.vision(images)
        
        # Pass deep features through classification head for Overall Survival (OS) prediction
        os_logits = self.head(self.dropout(f_v))
        
        return {
            "os_logits": os_logits,
            "seg_logits": seg_logits,
            "f_v": f_v
        }
