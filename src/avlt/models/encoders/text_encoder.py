import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TextEncoder(nn.Module):
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT', out_dim=768, freeze_layers=0):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=cfg)
        # Optionally freeze some lower layers
        if freeze_layers > 0:
            n_freeze = freeze_layers
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < n_freeze:
                    for p in layer.parameters():
                        p.requires_grad = False
        self.proj = nn.Linear(self.bert.config.hidden_size, out_dim)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:,0]  # CLS
        return self.proj(pooled)