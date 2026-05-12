"""Smoke-test the custom SkyFusion YOLO26 model wiring."""

from __future__ import annotations

from skyfusion_common import MODEL_CFG

import torch
from ultralytics import YOLO


def main() -> None:
    """Build the model, print its summary, and run a dummy forward pass."""
    model = YOLO(str(MODEL_CFG))
    model.model.info(imgsz=640, verbose=True)
    print(model.model.model[0])
    model.model.eval()
    with torch.no_grad():
        y = model.model(torch.randn(1, 3, 640, 640))
    print(type(y).__name__)


if __name__ == "__main__":
    main()
