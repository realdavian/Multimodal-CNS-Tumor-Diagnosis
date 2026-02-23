import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from monai.transforms import Compose

from .transforms import build_base_transforms, build_train_augmentations
from ..utils.loggers import logger

class BraTSDataset(Dataset):
    """
    BraTS Pediatric dataset for Overall Survival prediction.
    Loads 4 MRI modalities (t1n, t1c, t2w, t2f) and concatenates them.
    Expects labels to be integers (e.g., 0, 1, 2 for OS classes).
    """

    def __init__(
        self,
        data_root: str,
        cohort_csv: str,
        split: str = "train",
        image_size: int = 224,
        num_slices: int = 128,
        mode: str = "vision_only",
        augment: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.mode = mode
        self.image_size = image_size
        self.num_slices = num_slices

        # Read the cohort CSV
        csv_path = cohort_csv
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded BraTS cohort with {len(df)} rows from {csv_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load cohort CSV at {csv_path}: {e}")

        # Filter out rows with NaN OS_class
        df = df.dropna(subset=["OS_class"])
        df["OS_class"] = df["OS_class"].astype(int)
        
        # Sort by subject ID for reproducibility before splitting
        df = df.sort_values(by="BraTS-SubjectID").reset_index(drop=True)

        logger.info(f"BraTS dataset (all): {len(df)} samples with valid OS_class")

        # Create dictionaries of data for MONAI Dataset
        self.data = []
        for _, row in df.iterrows():
            subject_id = row["BraTS-SubjectID"]
            label = row["OS_class"]

            subject_dir = os.path.join(data_root, subject_id)
            if not os.path.isdir(subject_dir):
                logger.warning(f"Subject dir not found: {subject_dir}, skipping.")
                continue
            
            files = {
                "t1n": os.path.join(subject_dir, f"{subject_id}-t1n.nii.gz"),
                "t1c": os.path.join(subject_dir, f"{subject_id}-t1c.nii.gz"),
                "t2w": os.path.join(subject_dir, f"{subject_id}-t2w.nii.gz"),
                "t2f": os.path.join(subject_dir, f"{subject_id}-t2f.nii.gz"),
                "label": label,
                "subject_id": subject_id,
            }
            
            all_exist = all(os.path.exists(f) for k, f in files.items() if k not in ["label", "subject_id"])
            if not all_exist:
                logger.warning(f"Missing one or more MRI modalities for {subject_id}, skipping.")
                continue
                
            self.data.append(files)

        logger.info(f"BraTS dataset size after validation: {len(self.data)} samples")

        # Build transforms from shared module
        base = build_base_transforms(image_size, num_slices, include_seg=False)
        
        if augment:
            aug = build_train_augmentations(include_seg=False)
            self.transforms = Compose(base.transforms + aug.transforms)
            logger.info("Training augmentations enabled for BraTS dataset")
        else:
            self.transforms = base

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Apply MONAI transforms
        transformed = self.transforms(item)
        
        result = {
            "image": transformed["image"],
            "label": torch.tensor(item["label"], dtype=torch.long),
            "subject_id": item["subject_id"]
        }
        
        if self.mode == "multimodal":
            raise NotImplementedError("Multimodal (text) data is not yet implemented for BraTS.")
            
        return result
