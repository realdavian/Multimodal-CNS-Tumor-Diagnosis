
import os, torch, json
from torch.utils.data import DataLoader, random_split
from ..models.avlt import AVLT
from ..data.dataset import SyntheticMultimodalDataset
from .losses import Losses
from ..utils.metrics import MetricTracker
from ..viz.plots import save_confusion, save_roc

def create_dataloaders(cfg):
    if cfg["dataset"] == "synthetic":
        ds = SyntheticMultimodalDataset(n=256, num_classes=cfg["num_classes"],
                                        image_size=cfg["image_size"],
                                        num_slices=cfg.get("num_slices", 16),
                                        text_model=cfg["text"]["model_name"], split='train')
        n_train = int(0.8 * len(ds))
        n_val = len(ds) - n_train
        train_ds, val_ds = random_split(ds, [n_train, n_val])
        return DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"]), \
               DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    else:
        raise NotImplementedError("Integrate your real datasets here.")

def train_loop(cfg, device=None, max_steps=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg["outputs"], exist_ok=True)
    train_dl, val_dl = create_dataloaders(cfg)
    model_s = AVLT(num_classes=cfg["num_classes"], image_size=cfg["image_size"],
                   backbone=cfg["vision"]["backbone"], text_model=cfg["text"]["model_name"],
                   dropout=cfg["dropout"],
                   vision_variant=cfg["vision"].get("variant", "fixed")).to(device)
    model_t = AVLT(num_classes=cfg["num_classes"], image_size=cfg["image_size"],
                   backbone=cfg["vision"]["backbone"], text_model=cfg["text"]["model_name"],
                   dropout=cfg["dropout"],
                   vision_variant=cfg["vision"].get("variant", "fixed")).to(device)
    model_t.load_state_dict(model_s.state_dict())
    for p in model_t.parameters(): p.requires_grad = False

    # NOTE: Multi-GPU support via DataParallel
    # Wraps models to split batches across all visible GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Using {n_gpus} GPUs via DataParallel")
        model_s = torch.nn.DataParallel(model_s)
        model_t = torch.nn.DataParallel(model_t)

    opt = torch.optim.AdamW(model_s.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg["trainer"]["mixed_precision"] and device.startswith('cuda'))
    losses = Losses(cfg["w_align"], cfg["w_sd"])


    step = 0
    for epoch in range(cfg["epochs"]):
        model_s.train()
        for batch in train_dl:
            step += 1
            imgs = batch["image"].to(device)
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y = batch["label"].to(device)
            with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                logits_s, f_v, f_t, f_fused, alpha, beta = model_s(imgs, ids, attn)
                with torch.no_grad():
                    logits_t, *_ = model_t(imgs, ids, attn)
                loss, parts = losses.total(logits_s, logits_t, y, f_v, f_t)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), cfg["trainer"]["grad_clip"])
            scaler.step(opt); scaler.update(); opt.zero_grad()
            # EMA update
            # NOTE: Handle DataParallel by accessing .module if wrapped
            m = cfg["ema_momentum"]
            model_s_inner = model_s.module if hasattr(model_s, 'module') else model_s
            model_t_inner = model_t.module if hasattr(model_t, 'module') else model_t
            with torch.no_grad():
                for (n_s, p_s), (n_t, p_t) in zip(model_s_inner.state_dict().items(), model_t_inner.state_dict().items()):
                    p_t.copy_(m*p_t + (1-m)*p_s)
            if step % cfg["trainer"]["log_every"] == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} | cls {parts['cls']:.3f} align {parts['align']:.3f} sd {parts['sd']:.3f}")
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    # Save model
    # NOTE: Save underlying module if wrapped with DataParallel
    ckpt = os.path.join(cfg["outputs"], "avlt_synthetic.pt")
    model_to_save = model_s.module if hasattr(model_s, 'module') else model_s
    torch.save(model_to_save.state_dict(), ckpt)
    print("Saved:", ckpt)

    # Eval
    metrics = MetricTracker(cfg["num_classes"])
    model_s.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in val_dl:
            imgs = batch["image"].to(device)
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y = batch["label"].to(device)
            logits, *_ = model_s(imgs, ids, attn)
            metrics.update(logits, y)
            y_true.append(y.cpu())
            y_prob.append(torch.softmax(logits, dim=1).cpu())
    y_true = torch.cat(y_true).numpy()
    y_prob = torch.cat(y_prob).numpy()
    report = metrics.report()
    with open(os.path.join(cfg["outputs"], "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    from ..viz.plots import save_confusion, save_roc
    save_confusion(y_true, y_prob.argmax(1), os.path.join(cfg["outputs"], "confusion.png"))
    save_roc(y_true, y_prob, os.path.join(cfg["outputs"], "roc.png"))
