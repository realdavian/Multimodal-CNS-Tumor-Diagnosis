import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoConfig


class VisionEncoderNoPool(nn.Module):
    """
    Alternative fix: No pooling, preserves 224x224 spatial dimensions.
    More computationally expensive but uses pretrained weights at native resolution.
    """
    def __init__(self, image_size=224, backbone='vit_base_patch16_224', pretrained=True, cnn_stem=True, out_dim=768):
        super().__init__()
        self.cnn_stem = None
        if cnn_stem:
            self.cnn_stem = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                # No pooling - preserve 224x224
            )
            in_ch = 32
        else:
            in_ch = 4
        self.backbone = timm.create_model(backbone, pretrained=pretrained, in_chans=in_ch, num_classes=0)
        self.proj = nn.Linear(self.backbone.num_features, out_dim)

    def forward(self, x):
        if self.cnn_stem is not None:
            x = self.cnn_stem(x)
        feat = self.backbone(x)
        return self.proj(feat)