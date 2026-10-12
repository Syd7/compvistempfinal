"""Validate a SkyFusion checkpoint on the validation or test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from skyfusion_common import DATA_CFG, PROJECT_DIR, ROOT

from ultralytics import YOLO

DEFAULT_WEIGHTS = ROOT / "runs" / "final_project" / "skyfusion_yolo26s" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to best.pt or another checkpoint.")
    parser.add_argument("--split", choices=("val", "test", "train"), default="test", help="Dataset split to evaluate.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size. Defaults to 16 on CUDA, 2 on CPU.")
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device string. Defaults to 0 on CUDA, cpu otherwise.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Dataloader worker count.")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation for a second score run.")
    return parser.parse_args()


def run_val(weights: str, split: str, batch: int, device: str, workers: int, augment: bool = False):
    """Run validation and print the metrics needed for submission."""
    metrics = YOLO(weights).val(
        data=str(DATA_CFG),
        split=split,
        imgsz=640,
        batch=batch,
        device=device,
        workers=workers,
        project=str(PROJECT_DIR),
        name=f"{Path(weights).stem}_{split}{'_tta' if augment else ''}",
        exist_ok=True,
        plots=True,
        save_json=True,
        conf=0.001,
        iou=0.70,
        augment=augment,
    )
    print(metrics.results_dict)
    print(f"split: {split}")
    print(f"augment: {augment}")
    print(f"mAP50: {metrics.box.map50:.5f}")
    print(f"mAP50-95: {metrics.box.map:.5f}")
    print(f"precision: {metrics.box.mp:.5f}")
    print(f"recall: {metrics.box.mr:.5f}")
    for row in metrics.summary():
        print(row)
    return metrics


def main() -> None:
    """Run checkpoint validation."""
    args = parse_args()
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    batch = args.batch if args.batch is not None else (16 if torch.cuda.is_available() else 2)
    run_val(args.weights, args.split, batch, device, args.workers)
    if args.tta:
        run_val(args.weights, args.split, batch, device, args.workers, augment=True)


if __name__ == "__main__":
    main()
