import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoConfig


class VisionEncoderFixed(nn.Module):
    """
    Fixed implementation with MaxPool2d.
    Properly configures ViT with img_size=112 to match the pooled CNN stem output.
    This is computationally more efficient (4x fewer pixels) while working correctly.
    """
    def __init__(self, image_size=224, backbone='vit_base_patch16_224', pretrained=True, cnn_stem=True, out_dim=768):
        super().__init__()
        self.cnn_stem = None
        vit_img_size = image_size  # Default: no change
        
        if cnn_stem:
            self.cnn_stem = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2)  # 224 -> 112
            )
            in_ch = 32
            vit_img_size = image_size // 2  # 224 -> 112
        else:
            in_ch = 4
        
        # FIX: Tell ViT the correct input size after CNN stem pooling
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, in_chans=in_ch, num_classes=0,
            img_size=vit_img_size
        )
        self.proj = nn.Linear(self.backbone.num_features, out_dim)

    def forward(self, x):
        if self.cnn_stem is not None:
            x = self.cnn_stem(x)
        feat = self.backbone(x)
        return self.proj(feat)