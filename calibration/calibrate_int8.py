#!/usr/bin/env python3
# Copyright (c) 2026 Janlung Sung
# Tatung University — I4210 AI實務專題
"""Build an INT8-calibrated TensorRT engine from best.pt.

Run on the Jetson once. Reads ~500 representative frames from
calibration/calibration_data/ (which is gitignored — populate per
calibration/README.md) and writes best_int8.engine to the repo root.
"""
from pathlib import Path

from ultralytics import YOLO

CAL_DATA = Path(__file__).parent / "calibration_data"
WEIGHTS = Path(__file__).parent.parent / "best.pt"
OUT = Path(__file__).parent.parent / "best_int8.engine"


def main() -> None:
    if not CAL_DATA.exists() or len(list(CAL_DATA.glob("*.jpg"))) < 50:
        raise SystemExit(f"Need >=50 calibration images at {CAL_DATA}")

    model = YOLO(str(WEIGHTS), task="detect")
    model.export(
        format="engine",
        int8=True,
        data=str(CAL_DATA.parent / "calibration.yaml"),
        imgsz=320,
        batch=1,
        verbose=True,
    )
    src = WEIGHTS.with_suffix(".engine")
    src.rename(OUT)
    print(f"Wrote {OUT}, size = {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
