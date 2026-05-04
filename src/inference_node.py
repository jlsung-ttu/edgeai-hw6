#!/usr/bin/env python3
"""Inference node: runs YOLO26 TensorRT engine, publishes detections to MQTT.

Reads frames from a video file (or camera), runs detection with
the fine-tuned TensorRT engine from Lab 9, and publishes bounding-box
results as JSON messages to an MQTT topic.
"""

import argparse
import json
import os
import signal
import sys
import time

import cv2
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
from ultralytics import YOLO

# --- Graceful shutdown + Docker health check heartbeat ---
running = True


def signal_handler(sig, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global running
    print(f"\n[inference] Received signal {sig}, shutting down...")
    running = False


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def write_health():
    """Timestamp heartbeat for Docker HEALTHCHECK (read by healthcheck.py).
    Silently no-ops if /tmp isn't writable — harmless outside Docker."""
    try:
        with open("/tmp/inference_health", "w") as f:
            f.write(str(time.time()))
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(description="YOLO26 TensorRT inference node")
    parser.add_argument("--model", default="/opt/models/best.engine",
                        help="Path to TensorRT engine (built at image-build time)")
    parser.add_argument("--source", default="/opt/data/test_video.mp4",
                        help="Video file or camera index")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--mqtt-broker", default=os.getenv("MQTT_BROKER", "localhost"))
    parser.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument("--mqtt-topic", default="/sense/vision/detections")
    args = parser.parse_args()

    # Load model. task='detect' is required for .engine files — they don't
    # embed task metadata the way .pt files do, so Ultralytics can't infer it.
    print(f"[inference] Loading model: {args.model}")
    model = YOLO(args.model, task="detect")

    # Connect MQTT
    client = mqtt.Client(CallbackAPIVersion.VERSION2)
    print(f"[inference] Connecting to MQTT broker: {args.mqtt_broker}:{args.mqtt_port}")
    client.connect(args.mqtt_broker, args.mqtt_port)
    client.loop_start()

    # Open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[inference] ERROR: Cannot open source: {args.source}")
        sys.exit(1)

    frame_count = 0
    fps_start = time.monotonic()

    print(f"[inference] Running inference on {args.source}...")
    while running:
        ret, frame = cap.read()
        if not ret:
            # Loop video for continuous testing
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

        # Build detection payload
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": r.names[int(box.cls)],
                    "confidence": round(float(box.conf), 3),
                    "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],
                })

        payload = {
            "t": round(time.time(), 3),
            "frame": frame_count,
            "detections": detections,
            "count": len(detections),
        }
        client.publish(args.mqtt_topic, json.dumps(payload), qos=0)

        frame_count += 1
        # Heartbeat for Docker HEALTHCHECK (every 10 frames ≈ 200 ms at 50 FPS)
        if frame_count % 10 == 0:
            write_health()
        if frame_count % 100 == 0:
            elapsed = time.monotonic() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"[inference] {frame_count} frames, {fps:.1f} FPS, "
                  f"last frame: {len(detections)} detections")

    # Cleanup
    cap.release()
    client.loop_stop()
    client.disconnect()
    print(f"[inference] Shutdown complete. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()
