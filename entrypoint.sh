#!/usr/bin/env bash
# entrypoint.sh — runs at container start on the Jetson.
#
# What it does:
#   1. If best.engine is missing or older than best.pt, compile it with
#      Ultralytics' yolo export. This needs the GPU (--runtime nvidia)
#      and uses the container's TensorRT 10.3.
#   2. exec the Dockerfile's CMD (default: python3 inference_node.py)
#      so signals (SIGTERM from `docker stop`) reach Python directly.
#
# Why deferred to runtime instead of `RUN` in the Dockerfile?
#   QEMU emulation on the GitHub-hosted x86 runner can fake aarch64
#   instructions but cannot present a CUDA device, so `yolo export
#   format=engine` would fail at build time. Running it at container
#   start on the Jetson sidesteps the issue.
#
# First run takes 5–8 min. Mount /opt/models as a volume so subsequent
# runs reuse the cached engine:
#   docker run -v lab12-models:/opt/models ...

set -euo pipefail

MODEL_DIR=/opt/models
WEIGHTS=${MODEL_DIR}/best.pt
ENGINE=${MODEL_DIR}/best.engine

if [ ! -f "${WEIGHTS}" ]; then
  echo "ERROR: ${WEIGHTS} not found. Did you copy best.pt into the image?" >&2
  exit 1
fi

# Recompile if engine is missing or older than weights.
# The cd is in a subshell so it doesn't leak into the exec below —
# the parent shell stays in /app (the Dockerfile's WORKDIR), which
# is where inference_node.py lives.
if [ ! -f "${ENGINE}" ] || [ "${WEIGHTS}" -nt "${ENGINE}" ]; then
  echo "[entrypoint] Compiling TensorRT engine (this takes 5–8 min on first boot)..."
  (
    cd "${MODEL_DIR}"
    python3 -c "
from ultralytics import YOLO
YOLO('best.pt', task='detect').export(format='engine', imgsz=320, half=True, opset=19)
"
  )
  echo "[entrypoint] Engine compiled: $(ls -lh ${ENGINE} | awk '{print $5}')"
else
  echo "[entrypoint] Reusing cached engine: $(ls -lh ${ENGINE} | awk '{print $5}')"
fi

# Run whatever CMD the Dockerfile specifies (default: python3 inference_node.py).
# `exec` replaces bash with that process so SIGTERM from `docker stop`
# reaches it directly and graceful-shutdown handlers actually run.
# Cwd here is /app (WORKDIR), unchanged because the compile cd was
# scoped to the subshell above.
exec "$@"
