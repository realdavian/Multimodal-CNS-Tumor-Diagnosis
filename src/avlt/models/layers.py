import torch
import torch.nn as nn

class SliceAttention(nn.Module):
    """
    aggregated features across slices using learnable attention weights.
    Approximates the paper's 'multi-head surgery' which attends across patches in 3D.
    """
    def __init__(self, dim=768):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1)
        )
    
    def forward(self, x):
        # x: [B, D, dim]
        # Calculate attention weights for each slice
        weights = self.attn(x)  # [B, D, 1]
        weights = torch.softmax(weights, dim=1)
        
        # Weighted sum across D dimension
        return (x * weights).sum(dim=1)  # [B, dim]
