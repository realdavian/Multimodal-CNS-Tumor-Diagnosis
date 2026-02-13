
import torch
import torch.nn as nn
from .encoders import create_vision_encoder, TextEncoder
from .fusion import CrossAttentionFusion

class AVLT(nn.Module):
    def __init__(self, num_classes=2, image_size=224, backbone='vit_base_patch16_224',
                 text_model='emilyalsentzer/Bio_ClinicalBERT', dropout=0.3,
                 vision_variant='fixed'):
        super().__init__()
        self.vision = create_vision_encoder(
            variant=vision_variant,
            image_size=image_size,
            backbone=backbone,
            pretrained=True,
            cnn_stem=True,
            out_dim=768
        )
        self.text = TextEncoder(model_name=text_model, out_dim=768, freeze_layers=0)
        self.fusion = CrossAttentionFusion(dim=768, num_heads=8, dropout=0.1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(768, num_classes)

    def forward(self, images, input_ids, attention_mask):
        f_v = self.vision(images)
        f_t = self.text(input_ids, attention_mask)
        f_fused, alpha, beta = self.fusion(f_v, f_t)
        logits = self.head(self.dropout(f_fused))
        return logits, f_v, f_t, f_fused, alpha, beta
