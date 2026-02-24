"""
Pipeline diagnostic: traces shapes, value ranges, and architecture design
through the entire multitask pipeline on CPU to identify convergence issues.

Uses real data for data pipeline analysis and small mock tensors for model tracing.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from omegaconf import OmegaConf

def main():
    print("=" * 80)
    print("PIPELINE DIAGNOSTIC: Multitask BraTS (CPU mode)")
    print("=" * 80)

    # Load config
    base_cfg = OmegaConf.load("configs/base.yaml")
    exp_cfg = OmegaConf.load("configs/experiments/brats_os_multitask.yaml")
    cfg = OmegaConf.merge(base_cfg, exp_cfg)

    device = "cpu"

    # =========================================================================
    # STAGE 1: DATA LOADING - Real data, real transforms
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 1: DATA LOADING & TRANSFORMS (real data)")
    print("=" * 80)

    from avlt.data.brats_multitask import BraTSMultitaskDataset
    ds = BraTSMultitaskDataset(
        data_root=OmegaConf.select(cfg, "data_root"),
        cohort_csv=OmegaConf.select(cfg, "cohort_csv"),
        split="train",
        image_size=OmegaConf.select(cfg, "image_size", default=224),
        num_slices=OmegaConf.select(cfg, "num_slices", default=128),
        mode="multitask",
        augment=False,
    )

    sample = ds[0]
    image = sample["image"]
    seg_mask = sample["seg_mask"]
    label = sample["label"]
    subject_id = sample["subject_id"]

    print(f"\nSubject: {subject_id}")
    print(f"  image.shape:    {image.shape}")
    print(f"  image.dtype:    {image.dtype}")
    print(f"  image.min:      {image.min().item():.4f}")
    print(f"  image.max:      {image.max().item():.4f}")
    print(f"  image.mean:     {image.mean().item():.4f}")
    print(f"  image.std:      {image.std().item():.4f}")

    print(f"\n  seg_mask.shape: {seg_mask.shape}")
    print(f"  seg_mask.dtype: {seg_mask.dtype}")
    unique_labels = torch.unique(seg_mask)
    print(f"  seg_mask unique values: {unique_labels.tolist()}")
    for lbl in unique_labels:
        count = (seg_mask == lbl).sum().item()
        total = seg_mask.numel()
        pct = count / total * 100
        print(f"    Label {lbl.item()}: {count:,} voxels ({pct:.2f}%)")

    print(f"\n  OS label: {label.item()} (dtype: {label.dtype})")

    # =========================================================================
    # STAGE 2: BATCH CONSTRUCTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 2: BATCH CONSTRUCTION")
    print("=" * 80)

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(dl))

    imgs = batch["image"]
    seg_masks = batch["seg_mask"]
    labels = batch["label"]

    print(f"\n  Batch images.shape:    {imgs.shape}  (expect [B, 4, D, H, W])")
    print(f"  Batch seg_masks.shape: {seg_masks.shape}  (expect [B, D, H, W])")
    print(f"  Batch labels.shape:    {labels.shape}  (expect [B])")
    print(f"  Batch labels:          {labels.tolist()}")

    # =========================================================================
    # STAGE 3: MODEL FORWARD PASS (small mock tensors on CPU)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 3: MODEL ARCHITECTURE TRACE (mock tensors on CPU)")
    print("=" * 80)

    # Use small spatial dims so it fits in CPU memory quickly
    mock_size = 64
    mock_depth = 64
    mock_batch = 1
    print(f"\n  Using mock input: [{mock_batch}, 4, {mock_depth}, {mock_size}, {mock_size}]")

    from avlt.models.avlt_multitask import AVLTVisionMultitask
    model = AVLTVisionMultitask(
        num_classes=OmegaConf.select(cfg, "num_classes"),
        num_seg_classes=OmegaConf.select(cfg, "num_seg_classes", default=4),
        image_size=mock_size,
        backbone=OmegaConf.select(cfg, "vision.backbone"),
        dropout=0.0,
        vision_variant=OmegaConf.select(cfg, "vision.variant"),
    ).to(device)

    mock_input = torch.randn(mock_batch, 4, mock_depth, mock_size, mock_size, device=device)

    model.eval()
    with torch.no_grad():
        os_logits, seg_logits, f_v = model(mock_input)

    print(f"\n  os_logits.shape:  {os_logits.shape}  (expect [{mock_batch}, {OmegaConf.select(cfg, 'num_classes')}])")
    print(f"  os_logits values: {os_logits}")
    print(f"  os_logits.min:    {os_logits.min().item():.6f}")
    print(f"  os_logits.max:    {os_logits.max().item():.6f}")

    probs = torch.softmax(os_logits, dim=1)
    print(f"  os_probs:         {probs}")

    print(f"\n  seg_logits.shape: {seg_logits.shape}  (expect [{mock_batch}, 4, {mock_depth}, {mock_size}, {mock_size}])")
    print(f"  seg_logits.min:   {seg_logits.min().item():.6f}")
    print(f"  seg_logits.max:   {seg_logits.max().item():.6f}")
    print(f"  seg_logits.mean:  {seg_logits.mean().item():.6f}")
    print(f"  seg_logits.std:   {seg_logits.std().item():.6f}")

    seg_preds = seg_logits.argmax(dim=1)
    print(f"\n  seg_preds unique: {torch.unique(seg_preds).tolist()}")

    print(f"\n  f_v.shape:        {f_v.shape}  (expect [{mock_batch}, 768])")
    print(f"  f_v.min:          {f_v.min().item():.6f}")
    print(f"  f_v.max:          {f_v.max().item():.6f}")

    # =========================================================================
    # STAGE 4: SHAPE COMPATIBILITY
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 4: SHAPE COMPATIBILITY CHECK")
    print("=" * 80)

    # Key check: does model output spatial size == input spatial size?
    input_spatial = mock_input.shape[2:]
    output_spatial = seg_logits.shape[2:]
    match = input_spatial == output_spatial
    print(f"\n  Input spatial:      {input_spatial}")
    print(f"  Output seg spatial: {output_spatial}")
    print(f"  MATCH: {'YES' if match else 'NO -- MISMATCH DETECTED!'}")

    if not match:
        print(f"  *** CRITICAL: seg_logits spatial != input spatial ***")
        print(f"  DiceCELoss will compute loss between mismatched spatial dims!")

    # Also verify: are data seg_masks the same spatial as data images?
    data_img_spatial = imgs.shape[2:]
    data_seg_spatial = seg_masks.shape[1:]
    match2 = data_img_spatial == data_seg_spatial
    print(f"\n  Data image spatial: {data_img_spatial}")
    print(f"  Data seg spatial:   {data_seg_spatial}")
    print(f"  MATCH: {'YES' if match2 else 'NO -- MISMATCH!'}")

    # =========================================================================
    # STAGE 5: LOSS COMPUTATION (mock tensors)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 5: LOSS COMPUTATION")
    print("=" * 80)

    from avlt.train.losses import Losses
    losses = Losses(w_align=0.0, w_sd=0.5, w_seg=1.0)

    # Create mock targets matching model output spatial dims
    mock_labels = torch.tensor([0], dtype=torch.long, device=device)
    mock_seg_mask = torch.zeros(mock_batch, *output_spatial, dtype=torch.long, device=device)
    # Put some tumor labels in the center
    center = [s // 2 for s in output_spatial]
    r = 3
    mock_seg_mask[:, 
        max(0,center[0]-r):center[0]+r, 
        max(0,center[1]-r):center[1]+r, 
        max(0,center[2]-r):center[2]+r] = 1

    l_cls = losses.classification(os_logits, mock_labels)
    print(f"\n  Classification loss (CE):   {l_cls.item():.6f}")

    l_seg = losses.segmentation(seg_logits, mock_seg_mask)
    print(f"  Segmentation loss (DiceCE): {l_seg.item():.6f}")

    # DiceCELoss internal check
    print(f"\n  --- DiceCE internal check ---")
    seg_mask_expanded = mock_seg_mask.unsqueeze(1)
    print(f"  seg_mask for DiceCE shape: {seg_mask_expanded.shape}  (need [B, 1, D, H, W])")
    print(f"  seg_logits shape:          {seg_logits.shape}  (need [B, C, D, H, W])")
    print(f"  DiceCE to_onehot_y=True: will convert {seg_mask_expanded.shape} -> [{mock_batch}, 4, ...]")
    print(f"  DiceCE softmax=True: will softmax seg_logits along dim=1")
    print(f"  DiceCE include_background=False: will ignore class 0 in Dice")

    # =========================================================================
    # STAGE 6: SWINUNETR INTERNAL SHAPE TRACE
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 6: SWINUNETR INTERNAL SHAPE TRACE")
    print("=" * 80)

    with torch.no_grad():
        encoder = model.vision
        x = mock_input
        print(f"\n  Input: {x.shape}")

        hidden_states_out = encoder.swin_unetr.swinViT(x)
        for i, hs in enumerate(hidden_states_out):
            print(f"  swinViT hidden_states[{i}]: {hs.shape}")

        enc0 = encoder.swin_unetr.encoder1(x)
        print(f"\n  encoder1 (enc0): {enc0.shape}")
        enc1 = encoder.swin_unetr.encoder2(hidden_states_out[0])
        print(f"  encoder2 (enc1): {enc1.shape}")
        enc2 = encoder.swin_unetr.encoder3(hidden_states_out[1])
        print(f"  encoder3 (enc2): {enc2.shape}")
        enc3 = encoder.swin_unetr.encoder4(hidden_states_out[2])
        print(f"  encoder4 (enc3): {enc3.shape}")
        dec4 = encoder.swin_unetr.encoder10(hidden_states_out[4])
        print(f"  encoder10 (bottleneck): {dec4.shape}")

        dec3 = encoder.swin_unetr.decoder5(dec4, hidden_states_out[3])
        print(f"\n  decoder5: {dec3.shape}")
        dec2 = encoder.swin_unetr.decoder4(dec3, enc3)
        print(f"  decoder4: {dec2.shape}")
        dec1 = encoder.swin_unetr.decoder3(dec2, enc2)
        print(f"  decoder3: {dec1.shape}")
        dec0 = encoder.swin_unetr.decoder2(dec1, enc1)
        print(f"  decoder2: {dec0.shape}")
        out = encoder.swin_unetr.decoder1(dec0, enc0)
        print(f"  decoder1: {out.shape}")
        seg_out = encoder.swin_unetr.out(out)
        print(f"  out head: {seg_out.shape}")

        # Classification branch
        deep = hidden_states_out[-1]
        print(f"\n  Classification branch:")
        print(f"    deep features:              {deep.shape}")
        pooled = encoder.pool(deep).flatten(1)
        print(f"    AdaptiveAvgPool3d+flatten:   {pooled.shape}")
        print(f"    proj Linear: in={encoder.proj.in_features}, out={encoder.proj.out_features}")
        f_v_out = encoder.proj(pooled)
        print(f"    f_v output:                  {f_v_out.shape}")

    # =========================================================================
    # STAGE 7: GRADIENT FLOW CHECK (CPU, small tensors)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 7: GRADIENT FLOW")
    print("=" * 80)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    os_logits2, seg_logits2, f_v2 = model(mock_input)
    total_loss, parts = losses.total(
        logits_s=os_logits2,
        logits_t=None,
        y=mock_labels,
        f_v=f_v2,
        seg_logits_s=seg_logits2,
        seg_mask=mock_seg_mask,
    )

    print(f"\n  Total loss:   {total_loss.item():.6f}")
    for k, v in parts.items():
        print(f"  Loss {k}:     {v:.6f}")

    total_loss.backward()

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    nonzero = {k: v for k, v in grad_norms.items() if v > 0}
    zero = {k: v for k, v in grad_norms.items() if v == 0}

    print(f"\n  Params with gradients:      {len(nonzero)}")
    print(f"  Params with ZERO gradients: {len(zero)}")

    if zero:
        print(f"  *** WARNING: {len(zero)} params have zero gradient ***")
        for name in list(zero.keys())[:10]:
            print(f"    - {name}")

    sorted_grads = sorted(nonzero.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 largest gradient norms:")
    for name, norm in sorted_grads[:10]:
        print(f"    {norm:.6f}  {name}")

    nan_grads = [k for k, v in grad_norms.items() if not np.isfinite(v)]
    if nan_grads:
        print(f"\n  *** CRITICAL: NaN/Inf gradients in {len(nan_grads)} params ***")

    # =========================================================================
    # STAGE 8: DATASET LABEL DISTRIBUTION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 8: DATASET LABEL DISTRIBUTION")
    print("=" * 80)

    os_labels = [ds.data[i]["label"] for i in range(len(ds))]
    from collections import Counter
    label_counts = Counter(os_labels)
    print(f"\n  OS class distribution:")
    for cls, count in sorted(label_counts.items()):
        print(f"    Class {cls}: {count} samples ({count/len(os_labels)*100:.1f}%)")

    print(f"\n  Seg mask distribution (first 5 samples):")
    for i in range(min(5, len(ds))):
        s = ds[i]
        seg = s["seg_mask"]
        uniq = torch.unique(seg)
        pcts = {int(u): f"{(seg == u).sum().item() / seg.numel() * 100:.1f}%" for u in uniq}
        print(f"    Sample {i} ({s['subject_id']}): {pcts}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
