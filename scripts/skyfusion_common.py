"""Shared paths and runtime setup for the SkyFusion final-project scripts."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_CFG = ROOT / "ultralytics" / "cfg" / "models" / "ext" / "skyfusion-yolo26s.yaml"
DATA_CFG = ROOT / "skyfusion.v1i.yolov11 (1)" / "data.yaml"
PROJECT_DIR = ROOT / "runs" / "final_project"
YOLO_CONFIG_DIR = ROOT / ".yolo_config"


def prepare_yolo_config() -> None:
    """Keep Ultralytics settings and fonts inside the writable repo workspace."""
    os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
    YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (YOLO_CONFIG_DIR / "Ultralytics").mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        source_font = Path("C:/Windows/Fonts/arial.ttf")
        for target_font in (YOLO_CONFIG_DIR / "Arial.ttf", YOLO_CONFIG_DIR / "Ultralytics" / "Arial.ttf"):
            if source_font.exists() and not target_font.exists():
                shutil.copyfile(source_font, target_font)


prepare_yolo_config()
