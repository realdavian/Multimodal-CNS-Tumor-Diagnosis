
import argparse, yaml, os
from avlt.train.engine_vision_only import train_loop as vision_only_train_loop # replaced for vision only training
from avlt.train.engine import train_loop # original training loop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--train_engine", type=str, default="vision_only",
                        choices=["vision_only", "original"],
                        help="Training engine: 'vision_only' or 'original'")
    parser.add_argument("--synthetic", type=str, default="true")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--vision_variant", type=str, default=None,
                        choices=["fixed", "no_pool", "original", "swin3d"],
                        help="Override vision encoder variant: 'fixed', 'no_pool', 'original', or 'swin3d'")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.synthetic.lower() in ("true","1","yes"):
        cfg["dataset"] = "synthetic"
    if args.vision_variant:
        cfg["vision"]["variant"] = args.vision_variant
    os.makedirs(cfg["outputs"], exist_ok=True)
    if args.train_engine == "vision_only":
        vision_only_train_loop(cfg, max_steps=args.max_steps)
    else:
        train_loop(cfg, max_steps=args.max_steps)

if __name__ == "__main__":
    main()
