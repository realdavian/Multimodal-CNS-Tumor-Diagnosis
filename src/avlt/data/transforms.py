"""
Reusable MONAI transform factories for BraTS datasets.

Two factory functions:
- ``build_base_transforms`` -- deterministic preprocessing (load, orient, normalize, resize)
- ``build_train_augmentations`` -- stochastic augmentations applied only during training

Usage:
    base = build_base_transforms(image_size=224, num_slices=128, include_seg=True)
    aug  = build_train_augmentations(include_seg=True)
    full_train = Compose(base.transforms + aug.transforms)
"""

import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ConcatItemsd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    Resized,
    ToTensord,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
)


# ---------------------------------------------------------------------------
# Base Transforms (deterministic preprocessing)
# ---------------------------------------------------------------------------

def build_base_transforms(image_size: int, num_slices: int, include_seg: bool = False):
    """
    Build the deterministic preprocessing pipeline.

    Args:
        image_size:   Spatial H/W target size.
        num_slices:   Spatial D (depth) target size.
        include_seg:  If True, also load and resize the ``seg`` key
                      with nearest-neighbor interpolation.

    Returns:
        monai.transforms.Compose with the full preprocessing chain.
    """
    img_keys = ["t1n", "t1c", "t2w", "t2f"]
    orient_keys = ["image"]
    resize_steps = [
        Resized(keys=["image"], spatial_size=(image_size, image_size, num_slices), mode="trilinear"),
    ]
    transpose_keys = ["image"]
    tensor_keys = ["image"]

    if include_seg:
        orient_keys.append("seg")
        resize_steps.append(
            Resized(keys=["seg"], spatial_size=(image_size, image_size, num_slices), mode="nearest"),
        )
        transpose_keys.append("seg")
        tensor_keys.append("seg")

    load_keys = img_keys + (["seg"] if include_seg else [])

    transforms = [
        LoadImaged(keys=load_keys),
        EnsureChannelFirstd(keys=load_keys),
    ]

    # Seg label remapping must happen before concat (only for multitask)
    if include_seg:
        from .brats_multitask import SelectAndMapLabelsd
        transforms.append(SelectAndMapLabelsd(keys=["seg"]))

    transforms.extend([
        ConcatItemsd(keys=img_keys, name="image"),
        Orientationd(keys=orient_keys, axcodes="RAS"),
        ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0, b_max=1, clip=True),
        *resize_steps,
        monai.transforms.Transposed(keys=transpose_keys, indices=(0, 3, 1, 2)),
        ToTensord(keys=tensor_keys),
    ])

    return Compose(transforms)


# ---------------------------------------------------------------------------
# Training Augmentations (stochastic, applied only during training)
# ---------------------------------------------------------------------------

def build_train_augmentations(include_seg: bool = False):
    """
    Build stochastic data augmentation transforms for training.

    Spatial transforms (flips, rotations) are applied to both the image
    and segmentation mask to keep them aligned. Intensity transforms
    are applied only to the image.

    Args:
        include_seg:  If True, spatial augmentations also target the ``seg`` key.

    Returns:
        monai.transforms.Compose with the augmentation chain.
    """
    # Spatial transforms target both image and (optionally) seg
    spatial_keys = ["image"] + (["seg"] if include_seg else [])

    transforms = [
        # Spatial augmentations (applied identically to image + seg)
        RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=2),
        RandRotate90d(keys=spatial_keys, prob=0.5, max_k=3, spatial_axes=(1, 2)),

        # Intensity augmentations (image only, never seg)
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
    ]

    return Compose(transforms)
