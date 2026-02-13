
import numpy as np, torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class SyntheticMultimodalDataset(Dataset):
    def __init__(self, n=256, num_classes=2, image_size=224, num_slices=16, text_model='emilyalsentzer/Bio_ClinicalBERT', split='train'):
        super().__init__()
        self.n = n
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_slices = num_slices
        self.tokenizer = AutoTokenizer.from_pretrained(text_model, use_fast=True)
        rng = np.random.RandomState(0 if split=='train' else 1)
        
        # NOTE: Generating 3D volumes as [N, Channels, Depth, Height, Width]
        # Paper uses 4 channels (T1, T1c, T2, FLAIR) and D slices.
        # Shape: [n, 4, num_slices, 224, 224]
        self.images = rng.randn(n, 4, num_slices, image_size, image_size).astype(np.float32)
        
        self.labels = rng.randint(0, num_classes, size=(n,)).astype(np.int64)
        self.texts = [f"Patient with IDH {'mutation' if rng.rand()>0.5 else 'wildtype'}, age {int(20+rng.rand()*60)}." for _ in range(n)]

    def __len__(self): return self.n

    def __getitem__(self, idx):
        # NOTE: Returning 3D volume [4, D, H, W]
        img = self.images[idx]
        y = self.labels[idx]
        txt = self.texts[idx]
        enc = self.tokenizer(txt, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        return {
            "image": torch.from_numpy(img),
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(y, dtype=torch.long),
            "text": txt,
        }
