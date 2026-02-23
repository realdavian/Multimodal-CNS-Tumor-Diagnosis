from .original_fixed import VisionEncoderFixed
from .no_pool import VisionEncoderNoPool
from .slice_wise import SliceWiseVisionEncoder
from .original import VisionEncoder as VisionEncoderOriginal
from .swin3d import Swin3DVisionEncoder
from .swin3d_multitask import Swin3DMultitaskEncoder
# from .swin_umamba import SwinUMambaVisionEncoder
from .text_encoder import TextEncoder

__all__ = [
    "VisionEncoderFixed",
    "SliceWiseVisionEncoder",
    "VisionEncoderOriginal",
    "VisionEncoderNoPool",
    "Swin3DVisionEncoder",
    "Swin3DMultitaskEncoder",
    # "SwinUMambaVisionEncoder",
    "TextEncoder",
]


# Factory function
def create_vision_encoder(variant='fixed', **kwargs):
    """
    Factory function to create VisionEncoder variants.
    """
    variants = {
        'original': VisionEncoderOriginal,
        'fixed': VisionEncoderFixed,
        'no_pool': VisionEncoderNoPool,
        'slice_wise': SliceWiseVisionEncoder,
        'swin3d': Swin3DVisionEncoder,
        'swin3d_multitask': Swin3DMultitaskEncoder,
        # 'swinumamba': SwinUMambaVisionEncoder,
    }
    if variant not in variants:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(variants.keys())}")
    return variants[variant](**kwargs)