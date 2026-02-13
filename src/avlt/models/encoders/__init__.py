from .original_fixed import VisionEncoderFixed
from .no_pool import VisionEncoderNoPool
from .slice_wise import SliceWiseVisionEncoder
from .original import VisionEncoder as VisionEncoderOriginal
from .swin3d import Swin3DVisionEncoder
# from .swin_umamba import SwinUMambaVisionEncoder
from .text_encoder import TextEncoder

__all__ = [
    "VisionEncoderFixed",
    "SliceWiseVisionEncoder",
    "VisionEncoderOriginal",
    "VisionEncoderNoPool",
    "Swin3DVisionEncoder",
    # "SwinUMambaVisionEncoder",
    "TextEncoder",
]


# Factory function
def create_vision_encoder(variant='fixed', **kwargs):
    """
    Factory function to create VisionEncoder variants.
    
    Args:
        variant: str, one of:
            - 'original': Author's broken implementation (will crash)
            - 'fixed': Fixed with MaxPool2d + correct img_size
            - 'no_pool': No pooling, full 224x224 resolution
            - 'slice_wise': 2.5D ViT + slice attention aggregation
            - 'swin3d': 3D Swin Transformer from MONAI (paper architecture)
            - 'swinumamba': 2D Swin-UMamba (VMamba backbone) + slice attention
        **kwargs: Arguments passed to the encoder constructor
    
    Returns:
        VisionEncoder instance
    """
    variants = {
        'original': VisionEncoderOriginal,
        'fixed': VisionEncoderFixed,
        'no_pool': VisionEncoderNoPool,
        'slice_wise': SliceWiseVisionEncoder,
        'swin3d': Swin3DVisionEncoder,
        # 'swinumamba': SwinUMambaVisionEncoder,
    }
    if variant not in variants:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(variants.keys())}")
    return variants[variant](**kwargs)