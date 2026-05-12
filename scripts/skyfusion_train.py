"""Train and test the custom SkyFusion YOLO26 model on the SkyFusion dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from skyfusion_common import DATA_CFG, MODEL_CFG, PROJECT_DIR

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs for the full run.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size. Defaults to 16 on CUDA, 2 on CPU.")
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device string. Defaults to 0 on CUDA, cpu otherwise.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Dataloader worker count.")
    parser.add_argument("--name", default="skyfusion_yolo26s", help="Run name under runs/final_project.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of train data to use.")
    parser.add_argument("--cache", action="store_true", help="Cache images for faster repeated training.")
    parser.add_argument("--skip-test", action="store_true", help="Skip the final test-split validation.")
    parser.add_argument("--quick", action="store_true", help="One-epoch smoke run on a tiny fraction of the dataset.")
    return parser.parse_args()


def main() -> None:
    """Run training, then evaluate the best checkpoint on the test split."""
    args = parse_args()

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    batch = args.batch if args.batch is not None else (16 if torch.cuda.is_available() else 2)
    epochs = 1 if args.quick else args.epochs
    fraction = min(args.fraction, 0.005) if args.quick else args.fraction
    workers = 0 if args.quick else args.workers

    model = YOLO(str(MODEL_CFG))
    model.train(
        data=str(DATA_CFG),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=device,
        workers=workers,
        project=str(PROJECT_DIR),
        name=args.name,
        exist_ok=True,
        pretrained=False,
        plots=True,
        cos_lr=True,
        optimizer="AdamW",
        lr0=0.0015,
        lrf=0.02,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        patience=50,
        close_mosaic=20,
        mosaic=1.0,
        mixup=0.05,
        cutmix=0.05,
        hsv_h=0.015,
        hsv_s=0.60,
        hsv_v=0.35,
        degrees=7.0,
        translate=0.12,
        scale=0.55,
        shear=2.0,
        fliplr=0.5,
        flipud=0.5,
        cls_pw=0.25,
        deterministic=False,
        amp=torch.cuda.is_available() and device != "cpu",
        cache=args.cache,
        fraction=fraction,
    )

    run_dir = Path(model.trainer.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    print(f"Best checkpoint: {best_pt}")

    if args.skip_test:
        return

    metrics = YOLO(str(best_pt)).val(
        data=str(DATA_CFG),
        split="test",
        imgsz=640,
        batch=batch,
        device=device,
        workers=workers,
        project=str(PROJECT_DIR),
        name=f"{args.name}_test",
        exist_ok=True,
        plots=True,
        save_json=True,
        conf=0.001,
        iou=0.70,
    )
    print(metrics.results_dict)
    print(f"mAP50: {metrics.box.map50:.5f}")
    print(f"mAP50-95: {metrics.box.map:.5f}")
    print(f"precision: {metrics.box.mp:.5f}")
    print(f"recall: {metrics.box.mr:.5f}")


if __name__ == "__main__":
    main()
