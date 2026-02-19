"""
Entry point: ``python scripts/eval.py``

Uses Hydra for config composition.

Usage::

    # Evaluate a vision-only checkpoint
    python scripts/eval.py ckpt=outputs/avlt_vision_only.pt

    # Evaluate a multimodal checkpoint
    python scripts/eval.py ckpt=outputs/avlt_multimodal.pt mode=multimodal
"""

import os
import json

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from avlt.train.engine import create_dataloaders, _build_model
from avlt.utils.metrics import MetricTracker
from avlt.viz.plots import save_confusion, save_roc


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    ckpt_path = cfg.get("ckpt")
    if not ckpt_path:
        raise ValueError("Pass ckpt= on the command line, e.g.: python scripts/eval.py ckpt=outputs/avlt_vision_only.pt")

    mode = cfg.get("mode", "vision_only")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, val_dl = create_dataloaders(cfg)
    model = _build_model(cfg, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    metrics = MetricTracker(cfg.num_classes)
    y_true, y_prob = [], []

    with torch.no_grad():
        for batch in val_dl:
            imgs = batch["image"].to(device)
            y = batch["label"].to(device)

            if mode == "multimodal":
                ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                logits, *_ = model(imgs, ids, attn)
            else:
                logits, *_ = model(imgs)

            metrics.update(logits, y)
            y_true.append(y.cpu())
            y_prob.append(torch.softmax(logits, dim=1).cpu())

    y_true = torch.cat(y_true).numpy()
    y_prob = torch.cat(y_prob).numpy()

    report = metrics.report()
    print("Eval metrics:", report)

    out_dir = cfg.outputs
    os.makedirs(out_dir, exist_ok=True)
    save_confusion(y_true, y_prob.argmax(1), os.path.join(out_dir, "confusion_eval.png"))
    save_roc(y_true, y_prob, os.path.join(out_dir, "roc_eval.png"))
    with open(os.path.join(out_dir, "metrics_eval.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
