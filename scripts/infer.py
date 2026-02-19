"""
Entry point: ``python scripts/infer.py``

Uses Hydra for config composition.

Usage::

    # Vision-only inference (random image)
    python scripts/infer.py ckpt=outputs/avlt_vision_only.pt

    # Vision-only with real volume
    python scripts/infer.py ckpt=outputs/avlt_vision_only.pt image=path/to/volume.npy

    # Multimodal inference
    python scripts/infer.py ckpt=outputs/avlt_multimodal.pt mode=multimodal \\
        text="Patient with IDH mutation and MGMT methylation."
"""

import os

import hydra
from omegaconf import DictConfig
import numpy as np
import torch

from avlt.train.engine import _build_model


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    ckpt_path = cfg.get("ckpt")
    if not ckpt_path:
        raise ValueError("Pass ckpt= on the command line, e.g.: python scripts/infer.py ckpt=outputs/avlt_vision_only.pt")

    mode = cfg.get("mode", "vision_only")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _build_model(cfg, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Prepare image
    image_path = cfg.get("image")
    if image_path and os.path.exists(image_path):
        arr = np.load(image_path)
    else:
        num_slices = cfg.get("num_slices", 16)
        img_size = cfg.image_size
        arr = np.random.randn(4, num_slices, img_size, img_size).astype(np.float32)
        print(f"Using random volume: shape {arr.shape}")

    img = torch.tensor(arr).unsqueeze(0).to(device)

    with torch.no_grad():
        if mode == "multimodal":
            from transformers import AutoTokenizer
            text = cfg.get("text", "Patient with IDH mutation and MGMT methylation.")
            tok = AutoTokenizer.from_pretrained(cfg.text.model_name)
            enc = tok(
                text, padding="max_length", truncation=True,
                max_length=cfg.get("text_maxlen", 128), return_tensors="pt",
            )
            ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            logits, *_ = model(img, ids, attn)
        else:
            logits, *_ = model(img)

        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]

    print(f"Probabilities: {prob}")
    print(f"Predicted class: {prob.argmax()}")


if __name__ == "__main__":
    main()
