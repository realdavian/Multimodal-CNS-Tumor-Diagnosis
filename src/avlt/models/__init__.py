from .avlt_vision_only import AVLTVisionOnly
from .avlt import AVLT
from .avlt_multitask import AVLTVisionMultitask

from ..utils.loggers import logger

def create_model(cfg, device):
    """
    Factory function to instantiate the top-level AVLT model based on mode.
    
    Args:
        cfg: The configuration dictionary/OmegaConf.
        device: The target device (e.g., 'cuda', 'cpu').
        
    Returns:
        torch.nn.Module: The requested model wrapper, moved to the requested device.
    """
    try:
        from omegaconf import OmegaConf
        mode = OmegaConf.select(cfg, "mode", default="vision_only")
        num_classes = OmegaConf.select(cfg, "num_classes", default=3)
        num_seg_classes = OmegaConf.select(cfg, "num_seg_classes", default=4)
        image_size = OmegaConf.select(cfg, "image_size", default=224)
        backbone = OmegaConf.select(cfg, "vision.backbone", default="vit_base_patch16_224")
        dropout = OmegaConf.select(cfg, "dropout", default=0.3)
        vision_variant = OmegaConf.select(cfg, "vision.variant", default="fixed")
    except (ImportError, AttributeError):
        mode = cfg.get("mode", "vision_only")
        num_classes = cfg.get("num_classes", 3)
        num_seg_classes = cfg.get("num_seg_classes", 4)
        image_size = cfg.get("image_size", 224)
        backbone = cfg.get("vision", {}).get("backbone", "vit_base_patch16_224")
        dropout = cfg.get("dropout", 0.3)
        vision_variant = cfg.get("vision", {}).get("variant", "fixed")

    logger.debug(f"Building top-level model for mode={mode} (variant={vision_variant})")

    if mode == "multimodal":
        model = AVLT(
            num_classes=num_classes,
            image_size=image_size,
            backbone=backbone,
            dropout=dropout,
            vision_variant=vision_variant,
        )
    elif mode == "multitask":
        model = AVLTVisionMultitask(
            num_classes=num_classes,
            num_seg_classes=num_seg_classes,
            image_size=image_size,
            backbone=backbone,
            dropout=dropout,
            vision_variant=vision_variant,
        )
    else:  # vision_only
        model = AVLTVisionOnly(
            num_classes=num_classes,
            image_size=image_size,
            backbone=backbone,
            dropout=dropout,
            vision_variant=vision_variant,
        )

    model = model.to(device)
    
    # Calculate parameter count
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model [{type(model).__name__}] created. "
                f"Total params: {pytorch_total_params:,} | Trainable: {pytorch_trainable_params:,}")

    return model

__all__ = [
    "AVLTVisionOnly",
    "AVLT",
    "AVLTVisionMultitask",
    "create_model",
]
