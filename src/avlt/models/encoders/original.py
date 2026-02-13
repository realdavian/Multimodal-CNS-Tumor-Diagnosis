import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoConfig


class VisionEncoder(nn.Module):
    """
    Original author's implementation (BROKEN).
    Has MaxPool2d that reduces spatial dimensions to 112x112, 
    but ViT is created expecting 224x224 input.
    This will raise: AssertionError: Input height (56) doesn't match model (224).
    Kept for reference only.
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
                nn.MaxPool2d(2)  # 224 -> 112, but ViT still expects 224!
            )
            in_ch = 32
        else:
            in_ch = 4
        # BUG: ViT expects 224x224 but CNN stem outputs 112x112
        self.backbone = timm.create_model(backbone, pretrained=pretrained, in_chans=in_ch, num_classes=0)
        self.proj = nn.Linear(self.backbone.num_features, out_dim)

    def forward(self, x):
        if self.cnn_stem is not None:
            x = self.cnn_stem(x)
        feat = self.backbone(x)
        return self.proj(feat)