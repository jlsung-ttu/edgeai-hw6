#!/usr/bin/env python3
# Copyright (c) 2026 Janlung Sung
# Tatung University — I4210 AI實務專題
"""INT8-vs-FP16 accuracy regression gate (Part A, A5).

Reads calibration/accuracy_baseline.json (committed during Part 0) and
asserts the INT8 mAP@50 didn't drop more than the spec's 2 pt threshold
vs the FP16 baseline. Catches bad calibration before it lands on main.

Snapshot pattern: the actual mAP measurement runs once on the Jetson
(Part 0.3) and writes the JSON. CI then re-checks the threshold on every
PR by reading the JSON — fast, cheap, no engine or GPU needed in the
test job.
"""
import json
from pathlib import Path

import pytest

BASELINE = Path(__file__).parent.parent / "calibration" / "accuracy_baseline.json"
MAP50_DROP_LIMIT = 0.02   # in mAP@50 points (matches Part 0 rubric line)


@pytest.mark.skipif(
    not BASELINE.exists(),
    reason="No accuracy baseline yet — run calibration/calibrate_int8.py first",
)
def test_int8_map50_within_threshold_of_fp16() -> None:
    """Catches calibration drift. Re-run Part 0 to (re)generate the baseline."""
    data = json.loads(BASELINE.read_text())
    fp16, int8 = data["fp16_map50"], data["int8_map50"]
    drop = fp16 - int8
    assert drop <= MAP50_DROP_LIMIT, (
        f"INT8 mAP@50 dropped {drop:.4f} pts vs FP16 "
        f"({fp16:.4f} → {int8:.4f}); threshold is {MAP50_DROP_LIMIT}. "
        f"Re-calibrate with more representative frames per docs/OPTIMIZATION.md."
    )


def test_baseline_has_required_fields() -> None:
    """Snapshot file must record provenance, not just numbers."""
    if not BASELINE.exists():
        pytest.skip("No baseline yet")
    data = json.loads(BASELINE.read_text())
    for key in ("fp16_map50", "int8_map50", "test_split", "best_pt_md5"):
        assert key in data, f"accuracy_baseline.json missing {key!r}"
