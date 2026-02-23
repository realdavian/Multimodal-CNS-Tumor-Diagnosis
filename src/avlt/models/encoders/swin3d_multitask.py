import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class Swin3DMultitaskEncoder(nn.Module):
    """
    3D Swin Transformer architecture that outputs BOTH dense volumetric segmentation
    masks and a global extracted 1D feature vector for classification.
    """
    def __init__(
        self,
        in_channels=4,
        out_dim=768,         # Dimension of the classification extraction
        num_seg_classes=4,   # Ex: Background(0), Necrotic(1), Edema(2), Enhancing(4)
        feature_size=48,
        spatial_dims=3,
        use_v2=False,
        image_size=(224, 224, 128),
        # API compatibility args for Factory
        backbone=None,
        pretrained=None,
        cnn_stem=None,
    ):
        super().__init__()

        # Full UNETR model (Encoder + Decoder)
        self.swin_unetr = SwinUNETR(
            in_channels=in_channels,
            out_channels=num_seg_classes,
            feature_size=feature_size,
            spatial_dims=spatial_dims,
            use_v2=use_v2,
            img_size=image_size,
            use_checkpoint=True,
        )

        encoder_out_dim = feature_size * 16

        # Standard pooling approach for OS classifier branch
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(encoder_out_dim, out_dim)

    def forward(self, x):
        """
        Returns:
            f_v: [B, out_dim] classification features
            seg_logits: [B, num_seg_classes, D, H, W]
        """
        # We manually step through SwinUNETR's internal logic to avoid duplicate swinViT passes
        hidden_states_out = self.swin_unetr.swinViT(x)
        enc0 = self.swin_unetr.encoder1(x)
        enc1 = self.swin_unetr.encoder2(hidden_states_out[0])
        enc2 = self.swin_unetr.encoder3(hidden_states_out[1])
        enc3 = self.swin_unetr.encoder4(hidden_states_out[2])
        dec4 = self.swin_unetr.encoder10(hidden_states_out[4])
        
        dec3 = self.swin_unetr.decoder5(dec4, hidden_states_out[3])
        dec2 = self.swin_unetr.decoder4(dec3, enc3)
        dec1 = self.swin_unetr.decoder3(dec2, enc2)
        dec0 = self.swin_unetr.decoder2(dec1, enc1)
        out = self.swin_unetr.decoder1(dec0, enc0)
        seg_logits = self.swin_unetr.out(out)

        # Deep features extraction for OS classification
        deep_features = hidden_states_out[-1]  # [B, 768, D', H', W']
        pooled = self.pool(deep_features).flatten(1)
        f_v = self.proj(pooled)
        
        return f_v, seg_logits
