"""
Unified synthetic dataset for both vision-only and multimodal pipelines.

Set ``mode="vision_only"`` (default) to skip the tokenizer/text overhead,
or ``mode="multimodal"`` to include text generation and tokenization.

Generated data is cached to disk so subsequent runs with the same shape
parameters skip the expensive ~50s generation step.
"""

import json
import os
import time

import numpy as np
from numpy.random import default_rng
import torch
from torch.utils.data import Dataset

from ..utils.loggers import logger

# Default cache directory (relative to project root)
_DEFAULT_CACHE_DIR = "data/.synthetic_cache"


class SyntheticDataset(Dataset):
    """Synthetic 3-D MRI volumes (+ optional text) for smoke-testing.

    Each sample is a random ``(4, D, H, W)`` volume with a random label.
    When ``mode="multimodal"``, each sample also includes tokenized text.

    Data is cached to disk under ``cache_dir``.  A cache hit requires an
    existing ``.npz`` file whose stored shape metadata matches the
    requested ``(n, num_classes, image_size, num_slices, split)``.

    Args:
        n: Number of samples.
        num_classes: Number of classification targets.
        image_size: Spatial resolution (H = W).
        num_slices: Depth dimension (D).
        split: ``"train"`` or ``"val"`` — controls the random seed.
        mode: ``"vision_only"`` or ``"multimodal"``.
        text_model: HuggingFace model name for the tokenizer (multimodal only).
        text_maxlen: Max token length (multimodal only).
        cache_dir: Directory for cached ``.npz`` files.
    """

    def __init__(
        self,
        n: int = 256,
        num_classes: int = 2,
        image_size: int = 224,
        num_slices: int = 16,
        split: str = "train",
        mode: str = "vision_only",
        text_model: str = "emilyalsentzer/Bio_ClinicalBERT",
        text_maxlen: int = 128,
        cache_dir: str = _DEFAULT_CACHE_DIR,
    ):
        super().__init__()
        self.n = n
        self.mode = mode
        self.text_maxlen = text_maxlen

        # ---- try loading from cache ----
        cache_meta = {
            "n": n,
            "num_classes": num_classes,
            "image_size": image_size,
            "num_slices": num_slices,
            "split": split,
        }
        loaded = self._try_load_cache(cache_dir, split, cache_meta)

        if loaded:
            self.images, self.labels = loaded
        else:
            self.images, self.labels = self._generate_and_cache(
                n, num_classes, image_size, num_slices, split, cache_dir, cache_meta,
            )

        # ---- text (multimodal only) ----
        self.tokenizer = None
        self.texts = None
        if mode == "multimodal":
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(text_model, use_fast=True)
            py_rng = np.random.RandomState(0 if split == "train" else 1)
            self.texts = [
                f"Patient with IDH {'mutation' if py_rng.rand() > 0.5 else 'wildtype'}, "
                f"age {int(20 + py_rng.rand() * 60)}."
                for _ in range(n)
            ]

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_paths(cache_dir: str, split: str):
        """Return (npz_path, meta_path) for a given split."""
        npz = os.path.join(cache_dir, f"synthetic_{split}.npz")
        meta = os.path.join(cache_dir, f"synthetic_{split}.meta.json")
        return npz, meta

    @staticmethod
    def _try_load_cache(cache_dir, split, expected_meta):
        """Load cached data if it exists and matches expected shape."""
        npz_path, meta_path = SyntheticDataset._cache_paths(cache_dir, split)

        if not os.path.exists(npz_path) or not os.path.exists(meta_path):
            return None

        try:
            with open(meta_path) as f:
                stored_meta = json.load(f)
            if stored_meta != expected_meta:
                logger.debug(
                    f"Cache shape mismatch (stored={stored_meta}, "
                    f"requested={expected_meta}). Regenerating."
                )
                return None

            logger.debug(f"Loading cached synthetic data from {npz_path}")
            t0 = time.time()
            data = np.load(npz_path)
            images = data["images"]
            labels = data["labels"]
            logger.debug(f"Loaded cached data in {time.time() - t0:.2f}s")
            return images, labels

        except Exception as e:
            logger.debug(f"Cache load failed ({e}). Regenerating.")
            return None

    @staticmethod
    def _generate_and_cache(n, num_classes, image_size, num_slices, split, cache_dir, cache_meta):
        """Generate synthetic data and save to cache."""
        rng = default_rng(seed=0 if split == "train" else 1)

        logger.debug(
            f"Generating [{n} 4 {num_slices} {image_size} {image_size}] synthetic volumes"
        )
        t0 = time.time()
        images = rng.standard_normal(
            (n, 4, num_slices, image_size, image_size)
        ).astype(np.float32)
        labels = rng.integers(0, num_classes, size=n, dtype=np.int64)
        logger.debug(f"Generated volumes in {time.time() - t0:.2f}s")

        # Save to cache
        try:
            os.makedirs(cache_dir, exist_ok=True)
            npz_path, meta_path = SyntheticDataset._cache_paths(cache_dir, split)

            logger.debug(f"Saving synthetic data cache to {npz_path}")
            t0 = time.time()
            np.savez(npz_path, images=images, labels=labels)
            with open(meta_path, "w") as f:
                json.dump(cache_meta, f)
            logger.debug(f"Saved cache in {time.time() - t0:.2f}s")
        except Exception as e:
            logger.warning(f"Failed to save cache ({e}). Continuing without cache.")

        return images, labels

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        item = {
            "image": torch.from_numpy(self.images[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.mode == "multimodal":
            enc = self.tokenizer(
                self.texts[idx],
                padding="max_length",
                truncation=True,
                max_length=self.text_maxlen,
                return_tensors="pt",
            )
            item["input_ids"] = enc["input_ids"].squeeze(0)
            item["attention_mask"] = enc["attention_mask"].squeeze(0)
            item["text"] = self.texts[idx]
        return item

