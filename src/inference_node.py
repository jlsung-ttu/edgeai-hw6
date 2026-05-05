#!/usr/bin/env python3
# Copyright (c) 2026 Janlung Sung
# Tatung University — I4210 AI實務專題
"""Inference node: runs YOLO TensorRT engine, publishes detections to MQTT.

Refactored from the Lab 10 monolithic main() into small testable helpers:
- parse_args / build_detection_payload / write_health are pure functions
- connect_mqtt accepts a client_factory so tests can inject a mock
- run_inference_loop accepts injected video_capture / model / mqtt_client
- main() wires the above together and is itself testable end-to-end via mocks
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Any, Callable, Iterable, Optional

import cv2
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

from src import healthcheck

# `ultralytics` is imported lazily inside main()'s default model_factory.
# It pulls torch transitively, and torch has no x86 PyPI wheel for the Jetson
# CUDA build we use (PDM excludes it from the resolution). Deferring the
# import keeps unit tests on the laptop running while the real Jetson
# container still picks it up at first call.

# --- module-level state ---
_running = True


def signal_handler(sig: int, _frame: Any) -> None:
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global _running
    print(f"\n[inference] Received signal {sig}, shutting down...")
    _running = False


def is_running() -> bool:
    """Accessor for the run flag — easier to test than module-global access."""
    return _running


def reset_running_for_tests() -> None:
    """Reset _running to True between tests."""
    global _running
    _running = True


def install_signal_handlers() -> None:
    """Wire SIGTERM/SIGINT to signal_handler. Skipped during tests."""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def write_health(path: str = "/app/inference_health") -> bool:
    """Timestamp heartbeat for Docker HEALTHCHECK. Returns True on success.

    Default lives under the container's WORKDIR (/app) — writable by the
    inference user, readable by healthcheck.py in the same container,
    and not shared with any other process. We deliberately avoid /tmp:
    predictable temp paths invite symlink attacks (Bandit B108) and the
    container's /app is the cleaner choice anyway.
    """
    try:
        with open(path, "w") as f:
            f.write(str(time.time()))
        return True
    except OSError:
        return False


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Pure parsing — no I/O, no globals. argv=None means sys.argv (production)."""
    parser = argparse.ArgumentParser(description="YOLO TensorRT inference node")
    parser.add_argument("--model", default="/opt/models/best.engine",
                        help="Path to TensorRT engine")
    parser.add_argument("--source", default="/opt/data/test_video.mp4",
                        help="Video file or camera index")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--mqtt-broker", default=os.getenv("MQTT_BROKER", "localhost"))
    parser.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument("--mqtt-topic", default="jetson/vision/detections")
    return parser.parse_args(list(argv) if argv is not None else None)


def build_detection_payload(results: Any, frame_count: int,
                             timestamp: Optional[float] = None) -> dict:
    """Convert YOLO results into the MQTT JSON payload schema.

    Pure function — accepts whatever shape `results` has (real YOLO Results
    object, list of mocks with .boxes, etc.) and returns a serializable dict.
    """
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": r.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],
            })
    return {
        "t": round(timestamp if timestamp is not None else time.time(), 3),
        "frame": frame_count,
        "detections": detections,
        "count": len(detections),
    }


def connect_mqtt(broker: str, port: int,
                  client_factory: Optional[Callable[..., mqtt.Client]] = None
                  ) -> mqtt.Client:
    """Create an MQTT client and connect. Tests pass a mock factory."""
    factory = client_factory or (lambda: mqtt.Client(CallbackAPIVersion.VERSION2))
    client = factory()
    print(f"[inference] Connecting to MQTT broker: {broker}:{port}")
    client.connect(broker, port)
    client.loop_start()
    return client


def _build_csi_pipeline(sensor_id: int = 0, width: int = 1920, height: int = 1080,
                          framerate: int = 30, target_w: int = 320, target_h: int = 320
                          ) -> str:
    """Build a GStreamer pipeline string for an IMX219 CSI camera.

    The IMX219 doesn't expose a V4L2 stream cv2's default backend can
    drive — `cv2.VideoCapture("/dev/video0")` opens silently but
    `cap.read()` times out. The Jetson route is the Tegra GStreamer
    plugins (`nvarguscamerasrc`). The dustynv/pytorch base image ships
    them, so the container just needs to USE the right pipeline.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={framerate}/1 ! "
        f"nvvidconv ! "
        f"video/x-raw, width={target_w}, height={target_h}, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=1 sync=false"
    )


def open_video_source(source: str,
                       cap_factory: Optional[Callable[..., Any]] = None) -> Any:
    """Open the video source. Tests pass a cap_factory returning a mock.

    cv2.VideoCapture takes a path (str), camera index (int), or a
    GStreamer pipeline (str + cv2.CAP_GSTREAMER backend). Three source
    forms supported:
      - "csi:N"        → IMX219 CSI sensor N via nvarguscamerasrc pipeline
      - "0", "1", ...  → V4L2 camera index N (USB webcam)
      - "/path/foo.mp4", "rtsp://..."  → file/URL via default backend
    """
    factory = cap_factory or cv2.VideoCapture
    if source.startswith("csi:"):
        sensor_id = int(source.split(":", 1)[1])
        cap = factory(_build_csi_pipeline(sensor_id), cv2.CAP_GSTREAMER)
    elif source.isdigit():
        cap = factory(int(source))
    else:
        cap = factory(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    return cap


def process_one_frame(cap: Any, model: Any, args: argparse.Namespace,
                       client: Any, frame_count: int) -> bool:
    """Read one frame, run inference, publish. Returns False on EOF.

    Loops the video on EOF (so the test container runs continuously).
    """
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            return False
    results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
    payload = build_detection_payload(results, frame_count)
    client.publish(args.mqtt_topic, json.dumps(payload), qos=0)
    return True


def run_inference_loop(cap: Any, model: Any, args: argparse.Namespace,
                        client: Any, max_frames: Optional[int] = None,
                        no_frame_timeout_s: float = 60.0,
                        sleep_on_failure_s: float = 0.1) -> int:
    """Drive the inference loop. max_frames lets tests bound the loop.

    Resilience: when process_one_frame returns False (camera dropped a
    frame, V4L2 select() timed out, etc.), sleep briefly and retry
    instead of exiting. Exit only when no successful frame for
    `no_frame_timeout_s` seconds — that way a flaky camera doesn't
    immediately take down the container (and with it /healthz), but a
    permanently-dead camera still surfaces (container exits → docker
    restart-policy → deploy.sh's healthcheck eventually fails → rollback).

    In production max_frames is None (loop until SIGTERM or hard timeout).
    """
    frame_count = 0
    last_success = time.monotonic()
    while is_running():
        if max_frames is not None and frame_count >= max_frames:
            break
        if process_one_frame(cap, model, args, client, frame_count):
            frame_count += 1
            last_success = time.monotonic()
            if frame_count % 10 == 0:
                write_health()
            continue
        elapsed = time.monotonic() - last_success
        if elapsed > no_frame_timeout_s:
            print(f"[inference] No frames for {elapsed:.1f}s — exiting loop.")
            break
        time.sleep(sleep_on_failure_s)
    return frame_count


def cleanup(cap: Any, client: Any, frame_count: int) -> None:
    """Release resources. Idempotent."""
    cap.release()
    client.loop_stop()
    client.disconnect()
    print(f"[inference] Shutdown complete. Processed {frame_count} frames.")


def _default_model_factory(path: str, task: str) -> Any:   # pragma: no cover
    """Real YOLO loader. Imported lazily so unit tests don't pull torch.
    Skipped from coverage because torch is Jetson-only and tests use the
    injected mock factory; real exercise happens in tests/integration/."""
    # Deliberate lazy import — torch is Jetson-only, can't be loaded on the
    # x86 test job. The HW5-style pylint config (deferred to the compliance
    # refactor task) globally disables `import-outside-toplevel` which is
    # what would otherwise flag this; current ruff config doesn't select PL.
    from ultralytics import YOLO
    return YOLO(path, task=task)


def main(argv: Optional[Iterable[str]] = None,
          model_factory: Optional[Callable[..., Any]] = None,
          cap_factory: Optional[Callable[..., Any]] = None,
          mqtt_factory: Optional[Callable[..., mqtt.Client]] = None,
          max_frames: Optional[int] = None) -> int:
    """Main entry point. All heavy deps are injectable for tests."""
    # Start the /healthz HTTP server on a daemon thread (Part D D1).
    # Daemon=True means it dies with main() — no explicit cleanup needed.
    healthcheck.start_in_thread()

    args = parse_args(argv)
    print(f"[inference] Loading model: {args.model}")
    model = (model_factory or _default_model_factory)(args.model, "detect")
    client = connect_mqtt(args.mqtt_broker, args.mqtt_port, client_factory=mqtt_factory)
    try:
        cap = open_video_source(args.source, cap_factory=cap_factory)
    except RuntimeError as e:
        print(f"[inference] ERROR: {e}")
        return 1
    print(f"[inference] Running inference on {args.source}...")
    frame_count = run_inference_loop(cap, model, args, client, max_frames=max_frames)
    cleanup(cap, client, frame_count)
    return 0


if __name__ == "__main__":   # pragma: no cover
    install_signal_handlers()
    sys.exit(main())
