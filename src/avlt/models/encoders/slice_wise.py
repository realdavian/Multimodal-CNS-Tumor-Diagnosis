import torch
import torch.nn as nn
import timm
from transformers import AutoModel
from ..layers import SliceAttention


class SliceWiseVisionEncoder(nn.Module):
    """
    3D Encoder that processes 3D MRI volumes [B, C, D, H, W] using 2D ViT. Aggregates features across slices using attention.
    Pipeline:
    1. Process each of the D slices through CNN Stem + ViT (sharing weights)
    2. Aggregate resulting D feature vectors using SliceAttention
    """
    def __init__(self, image_size=224, backbone='vit_base_patch16_224', pretrained=True, cnn_stem=True, out_dim=768):
        super().__init__()
        self.cnn_stem = None
        vit_img_size = image_size
        
        if cnn_stem:
            self.cnn_stem = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                # NOTE: Using MaxPool2d(2) as in VisionEncoderFixed to align with ViT
                nn.MaxPool2d(2)  # 224 -> 112
            )
            in_ch = 32
            vit_img_size = image_size // 2  # 112
        else:
            in_ch = 4
        
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, in_chans=in_ch, num_classes=0,
            img_size=vit_img_size
        )
        self.proj = nn.Linear(self.backbone.num_features, out_dim)
        self.slice_attn = SliceAttention(out_dim)

    def forward(self, x):
        # x input shape: [B, C, D, H, W]
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input [B, C, D, H, W], got {x.shape}")
            
        B, C, D, H, W = x.shape
        
        # Reshape to process all slices as a batch: [B*D, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        
        # 1. Feature Extraction per slice
        if self.cnn_stem is not None:
            x = self.cnn_stem(x)  # [B*D, 32, 112, 112]
            
        feat = self.backbone(x)   # [B*D, hidden_dim]
        feat = self.proj(feat)    # [B*D, 768]
        
        # 2. Aggregation across slices
        feat = feat.view(B, D, -1)  # [B, D, 768]
        feat = self.slice_attn(feat) # [B, 768]
        
        return feat