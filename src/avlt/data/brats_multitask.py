import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from monai.transforms import Compose, MapTransform

from .transforms import build_base_transforms, build_train_augmentations
from ..utils.loggers import logger

class SelectAndMapLabelsd(MapTransform):
    """
    BraTS Segmentation masks often contain specific labels such as 0, 1, 2, 4.
    This transform converts them into continuous indices (e.g., [0, 1, 2, 3]) 
    so CrossEntropyLoss handles them safely.
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # Map values: 0 -> 0 (Background), 1 -> 1 (Necrotic), 2 -> 2 (Edema), 4 -> 3 (Enhancing)
            mask = d[key]
            d[key][mask == 4] = 3
        return d

class BraTSMultitaskDataset(Dataset):
    """
    BraTS Pediatric dataset for Multitask Learning.
    Loads 4 MRI modalities (t1n, t1c, t2w, t2f) for inputs AND 
    `seg.nii.gz` for Dense segmentation targets.
    Returns:
       dict with `image`, `os_label`, and `seg_mask`.
    """

    def __init__(
        self,
        data_root: str,
        cohort_csv: str,
        split: str = "train",
        image_size: int = 224,
        num_slices: int = 128,
        mode: str = "multitask",
        augment: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.mode = mode
        self.image_size = image_size
        self.num_slices = num_slices

        csv_path = cohort_csv
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load cohort CSV at {csv_path}: {e}")

        df = df.dropna(subset=["OS_class"])
        df["OS_class"] = df["OS_class"].astype(int)
        df = df.sort_values(by="BraTS-SubjectID").reset_index(drop=True)

        self.data = []
        for _, row in df.iterrows():
            subject_id = row["BraTS-SubjectID"]
            label = row["OS_class"]

            subject_dir = os.path.join(data_root, subject_id)
            if not os.path.isdir(subject_dir):
                continue
            
            files = {
                "t1n": os.path.join(subject_dir, f"{subject_id}-t1n.nii.gz"),
                "t1c": os.path.join(subject_dir, f"{subject_id}-t1c.nii.gz"),
                "t2w": os.path.join(subject_dir, f"{subject_id}-t2w.nii.gz"),
                "t2f": os.path.join(subject_dir, f"{subject_id}-t2f.nii.gz"),
                "seg": os.path.join(subject_dir, f"{subject_id}-seg.nii.gz"),
                "label": label,
                "subject_id": subject_id,
            }
            
            all_images_exist = all(os.path.exists(f) for k, f in files.items() if k not in ["label", "subject_id"])
            if all_images_exist:
                self.data.append(files)

        logger.info(f"Multitask BraTS dataset size: {len(self.data)} samples")

        # Build transforms from shared module
        base = build_base_transforms(image_size, num_slices, include_seg=True)
        
        if augment:
            aug = build_train_augmentations(include_seg=True)
            self.transforms = Compose(base.transforms + aug.transforms)
            logger.info("Training augmentations enabled for multitask dataset")
        else:
            self.transforms = base

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        transformed = self.transforms(item)
        
        result = {
            "image": transformed["image"],
            "seg_mask": transformed["seg"].squeeze(0).long(),  # Squeeze the default channel dim (1) for CrossEntropy labels
             "label": torch.tensor(item["label"], dtype=torch.long),
            "subject_id": item["subject_id"]
        }
            
        return result
