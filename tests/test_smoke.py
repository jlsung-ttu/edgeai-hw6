#!/usr/bin/env python3
# Copyright (c) 2026 Janlung Sung
# Tatung University — I4210 AI實務專題
"""Lab 12 smoke tests — minimal CPU-only assertions that don't need GPU.

These tests run inside GitHub's free hosted runner (Ubuntu x86), so they
must not import torch with CUDA, must not load TensorRT, and must not
require the IMX219 camera. Their job is to catch the obvious 'I broke
the import graph' or 'the YOLO model file is corrupted' failures BEFORE
the slower Docker build job spends 10 min building an ARM64 image.

Run locally with:    pytest -v tests/

Note: this is the Step 1 baseline (4 functions / 6 collected items via
parametrization). Step 2 introduces the Dockerfile refactor +
entrypoint.sh, both of which are verified by validate_step2.sh rather
than by additional pytest cases — that keeps the pytest smoke set
stable across steps and avoids tests that would deliberately fail on
the first push.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


def test_best_pt_exists():
    """The fine-tuned weights file must be present in the repo."""
    p = REPO_ROOT / "best.pt"
    assert p.exists(), f"{p} missing — did you forget to commit best.pt?"
    assert p.stat().st_size > 1_000_000, "best.pt suspiciously small (<1 MB)"


def test_requirements_pinned():
    """Every dep in requirements.txt should be pinned (== or ~= or >=)."""
    req = (REPO_ROOT / "requirements.txt").read_text()
    for line in req.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Allow editable installs and -e/-r/-c directives
        if line.startswith(("-e", "-r", "-c", "--")):
            continue
        assert "==" in line or "~=" in line or ">=" in line, \
            f"Unpinned dep in requirements.txt: {line!r}"


def test_dockerfile_uses_arm64_base():
    """Dockerfile.ci must use a Jetson-compatible (ARM64) base image."""
    df = (REPO_ROOT / "Dockerfile.ci").read_text()
    # dustynv images are aarch64-only; l4t-base is also aarch64-only.
    # Either is acceptable; a generic python:3.11 base would silently
    # build for x86 and never run on the Jetson.
    assert any(base in df for base in ["dustynv/", "l4t-", "nvcr.io/nvidia/l4t"]), \
        "Dockerfile.ci must FROM a Jetson ARM64 base (dustynv/* or l4t-*)"


@pytest.mark.parametrize("name", ["src/inference_node.py", "best.pt", "requirements.txt"])
def test_required_files(name):
    """Files the Docker COPY steps reference must exist."""
    assert (REPO_ROOT / name).exists(), f"{name} missing"
