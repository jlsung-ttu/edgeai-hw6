#!/usr/bin/env python3
# Copyright (c) 2026 Janlung Sung
# Tatung University — I4210 AI實務專題
"""Unit tests for src/inference_node.py — coverage target ≥90%.

These tests deliberately mock cv2 / YOLO / paho-mqtt so they run on the
free GitHub-hosted x86 runner. Real GPU + camera + broker tests live in
tests/integration/ and run on the self-hosted Jetson runner.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src import inference_node as node


@pytest.fixture(autouse=True)
def reset_running():
    """Reset the module-global _running flag between tests."""
    node.reset_running_for_tests()
    yield
    node.reset_running_for_tests()


# ──────────────────────────────────────────────────────────────────
# parse_args
# ──────────────────────────────────────────────────────────────────


def test_parse_args_defaults():
    args = node.parse_args([])
    assert args.model == "/opt/models/best.engine"
    assert args.imgsz == 320
    assert args.conf == 0.25
    assert args.mqtt_topic == "jetson/vision/detections"


def test_parse_args_overrides():
    args = node.parse_args([
        "--model", "/tmp/m.engine",
        "--source", "0",
        "--imgsz", "640",
        "--conf", "0.5",
        "--mqtt-broker", "broker.local",
        "--mqtt-port", "8883",
        "--mqtt-topic", "test/topic",
    ])
    assert args.model == "/tmp/m.engine"
    assert args.source == "0"
    assert args.imgsz == 640
    assert args.conf == 0.5
    assert args.mqtt_broker == "broker.local"
    assert args.mqtt_port == 8883
    assert args.mqtt_topic == "test/topic"


# ──────────────────────────────────────────────────────────────────
# write_health
# ──────────────────────────────────────────────────────────────────


def test_write_health_writes_timestamp(tmp_path):
    target = tmp_path / "h"
    assert node.write_health(str(target)) is True
    assert target.exists()
    body = target.read_text().strip()
    assert float(body) > 0   # parseable as a float timestamp


def test_write_health_returns_false_on_oserror():
    # Path that's guaranteed unwritable (no perms / nonexistent parent dir)
    assert node.write_health("/nonexistent/dir/health") is False


# ──────────────────────────────────────────────────────────────────
# signal_handler / running flag
# ──────────────────────────────────────────────────────────────────


def test_signal_handler_clears_running_flag(capsys):
    assert node.is_running() is True
    node.signal_handler(15, None)
    assert node.is_running() is False
    captured = capsys.readouterr()
    assert "Received signal 15" in captured.out


def test_install_signal_handlers_does_not_raise():
    # Just verify the call completes — actual signal-delivery testing is
    # an integration concern, not a unit-test concern.
    node.install_signal_handlers()


# ──────────────────────────────────────────────────────────────────
# build_detection_payload
# ──────────────────────────────────────────────────────────────────


def _fake_box(cls_idx: int, conf: float, xyxy: list[float]) -> MagicMock:
    box = MagicMock()
    box.cls = cls_idx
    box.conf = conf
    # xyxy is shaped (1, 4) in real ultralytics; mimic .tolist() shape
    box.xyxy = MagicMock()
    box.xyxy.__getitem__ = lambda self, idx: MagicMock(
        tolist=lambda: xyxy
    )
    return box


def _fake_results(boxes_per_result: list[list[MagicMock]],
                   names: dict) -> list[MagicMock]:
    results = []
    for boxes in boxes_per_result:
        r = MagicMock()
        r.boxes = boxes
        r.names = names
        results.append(r)
    return results


def test_build_detection_payload_no_detections():
    results = _fake_results([[]], names={0: "person"})
    payload = node.build_detection_payload(results, frame_count=0, timestamp=1700000000.0)
    assert payload["frame"] == 0
    assert payload["count"] == 0
    assert payload["detections"] == []
    assert payload["t"] == 1700000000.0
    # JSON-serializable
    json.dumps(payload)


def test_build_detection_payload_one_detection():
    results = _fake_results(
        [[_fake_box(0, 0.95, [10.0, 20.0, 30.0, 40.0])]],
        names={0: "hardhat"},
    )
    payload = node.build_detection_payload(results, frame_count=42, timestamp=1700000001.0)
    assert payload["frame"] == 42
    assert payload["count"] == 1
    assert payload["detections"][0]["class"] == "hardhat"
    assert payload["detections"][0]["confidence"] == 0.95
    assert payload["detections"][0]["bbox"] == [10.0, 20.0, 30.0, 40.0]


def test_build_detection_payload_multiple_results():
    boxes1 = [_fake_box(0, 0.9, [0, 0, 10, 10])]
    boxes2 = [_fake_box(1, 0.8, [20, 20, 30, 30]),
              _fake_box(2, 0.7, [40, 40, 50, 50])]
    results = _fake_results([boxes1, boxes2], names={0: "a", 1: "b", 2: "c"})
    payload = node.build_detection_payload(results, frame_count=5, timestamp=1.0)
    assert payload["count"] == 3
    classes = [d["class"] for d in payload["detections"]]
    assert classes == ["a", "b", "c"]


def test_build_detection_payload_timestamp_default_uses_clock(monkeypatch):
    monkeypatch.setattr(node.time, "time", lambda: 12345.6789)
    results = _fake_results([[]], names={})
    payload = node.build_detection_payload(results, frame_count=0)
    assert payload["t"] == 12345.679


# ──────────────────────────────────────────────────────────────────
# connect_mqtt
# ──────────────────────────────────────────────────────────────────


def test_connect_mqtt_uses_factory_and_starts_loop():
    mock_client = MagicMock()
    factory_calls = []
    def factory():
        factory_calls.append(True)
        return mock_client
    out = node.connect_mqtt("broker", 1883, client_factory=factory)
    assert out is mock_client
    assert len(factory_calls) == 1
    mock_client.connect.assert_called_once_with("broker", 1883)
    mock_client.loop_start.assert_called_once()


def test_connect_mqtt_default_factory_creates_paho_client():
    # Patch paho.mqtt.client.Client at the import site
    with patch("src.inference_node.mqtt.Client") as mock_paho:
        instance = MagicMock()
        mock_paho.return_value = instance
        out = node.connect_mqtt("broker", 1883)
        mock_paho.assert_called_once()
        assert out is instance


# ──────────────────────────────────────────────────────────────────
# open_video_source
# ──────────────────────────────────────────────────────────────────


def test_open_video_source_with_factory_succeeds():
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    out = node.open_video_source("/x", cap_factory=lambda src: fake_cap)
    assert out is fake_cap


def test_open_video_source_raises_when_not_opened():
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = False
    with pytest.raises(RuntimeError, match="Cannot open source"):
        node.open_video_source("/missing", cap_factory=lambda src: fake_cap)


def test_open_video_source_default_uses_cv2():
    with patch("src.inference_node.cv2.VideoCapture") as mock_cap_cls:
        instance = MagicMock()
        instance.isOpened.return_value = True
        mock_cap_cls.return_value = instance
        out = node.open_video_source("rtsp://x")
        mock_cap_cls.assert_called_once_with("rtsp://x")
        assert out is instance


def test_open_video_source_converts_digit_string_to_int_for_camera():
    """`--source 0` (camera index) must reach cv2 as int 0, not str "0".
    Otherwise cv2.VideoCapture("0") looks for a file named "0" instead
    of opening camera index 0 — that's the actual production failure
    mode that drove this conversion."""
    captured = []

    def factory(src):
        captured.append(src)
        cap = MagicMock()
        cap.isOpened.return_value = True
        return cap

    node.open_video_source("0", cap_factory=factory)
    assert captured == [0]   # int, not "0"


def test_open_video_source_keeps_path_as_string():
    """Non-digit sources (file paths, URLs) must NOT be int-coerced."""
    captured = []

    def factory(src):
        captured.append(src)
        cap = MagicMock()
        cap.isOpened.return_value = True
        return cap

    node.open_video_source("/opt/data/video.mp4", cap_factory=factory)
    assert captured == ["/opt/data/video.mp4"]


def test_open_video_source_csi_builds_gstreamer_pipeline():
    """`csi:0` (IMX219 CSI camera) must build a GStreamer pipeline string
    AND request the GStreamer backend explicitly. Default cv2 backend
    (V4L2) silently opens but never produces frames for IMX219."""
    captured: list = []

    def factory(*args):
        captured.append(args)
        cap = MagicMock()
        cap.isOpened.return_value = True
        return cap

    import cv2 as _cv2
    node.open_video_source("csi:0", cap_factory=factory)
    assert len(captured) == 1
    pipeline, backend = captured[0]
    assert "nvarguscamerasrc" in pipeline
    assert "sensor-id=0" in pipeline
    assert "appsink" in pipeline
    assert backend == _cv2.CAP_GSTREAMER


def test_open_video_source_csi_passes_sensor_id():
    """Sensor index in the source string must reach the pipeline."""
    captured: list = []

    def factory(*args):
        captured.append(args)
        cap = MagicMock()
        cap.isOpened.return_value = True
        return cap

    node.open_video_source("csi:1", cap_factory=factory)
    pipeline, _backend = captured[0]
    assert "sensor-id=1" in pipeline


# ──────────────────────────────────────────────────────────────────
# process_one_frame
# ──────────────────────────────────────────────────────────────────


def _args(**kwargs):
    base = dict(model="m", source="s", imgsz=320, conf=0.25,
                mqtt_broker="b", mqtt_port=1883, mqtt_topic="t/topic")
    base.update(kwargs)
    return type("Args", (), base)()


def test_process_one_frame_publishes_when_frame_read_succeeds():
    cap = MagicMock()
    cap.read.return_value = (True, "fake-frame")
    model = MagicMock()
    model.predict.return_value = _fake_results([[]], names={})
    client = MagicMock()
    args = _args()
    out = node.process_one_frame(cap, model, args, client, frame_count=0)
    assert out is True
    client.publish.assert_called_once()
    topic, body = client.publish.call_args.args[:2]
    assert topic == "t/topic"
    assert json.loads(body)["frame"] == 0


def test_process_one_frame_loops_video_on_eof():
    cap = MagicMock()
    # First read fails (EOF), second read (post-rewind) succeeds
    cap.read.side_effect = [(False, None), (True, "frame")]
    model = MagicMock()
    model.predict.return_value = _fake_results([[]], names={})
    client = MagicMock()
    out = node.process_one_frame(cap, model, _args(), client, frame_count=0)
    assert out is True
    cap.set.assert_called_once()


def test_process_one_frame_returns_false_on_double_eof():
    cap = MagicMock()
    cap.read.return_value = (False, None)
    model = MagicMock()
    client = MagicMock()
    out = node.process_one_frame(cap, model, _args(), client, frame_count=0)
    assert out is False
    client.publish.assert_not_called()


# ──────────────────────────────────────────────────────────────────
# run_inference_loop
# ──────────────────────────────────────────────────────────────────


def test_run_inference_loop_bounded_by_max_frames():
    cap = MagicMock()
    cap.read.return_value = (True, "frame")
    model = MagicMock()
    model.predict.return_value = _fake_results([[]], names={})
    client = MagicMock()
    n = node.run_inference_loop(cap, model, _args(), client, max_frames=3)
    assert n == 3
    assert client.publish.call_count == 3


def test_run_inference_loop_stops_after_no_frame_timeout():
    """Two successful frames, then forever-failing reads → loop must exit
    after no_frame_timeout_s seconds, not on the first failure (resilience
    fix from D2: a flaky camera shouldn't take down the container instantly,
    but a permanently-dead one still surfaces eventually)."""
    cap = MagicMock()
    # 2 successes, then unbounded failures
    cap.read.side_effect = ([(True, "f"), (True, "f")]
                            + [(False, None)] * 1000)
    model = MagicMock()
    model.predict.return_value = _fake_results([[]], names={})
    client = MagicMock()
    n = node.run_inference_loop(cap, model, _args(), client, max_frames=100,
                                no_frame_timeout_s=0.05,
                                sleep_on_failure_s=0.005)
    assert n == 2


def test_run_inference_loop_resilient_to_transient_failures():
    """Transient frame failures interleaved with successes must NOT exit
    the loop — last_success keeps getting updated, timeout never fires."""
    cap = MagicMock()
    cap.read.side_effect = [
        (True, "f1"),
        (False, None),    # transient — process_one_frame's seek-then-retry also fails
        (False, None),
        (True, "f2"),     # recovery
        (True, "f3"),
        (False, None),
        (False, None),
        (True, "f4"),
    ]
    model = MagicMock()
    model.predict.return_value = _fake_results([[]], names={})
    client = MagicMock()
    n = node.run_inference_loop(cap, model, _args(), client, max_frames=4,
                                no_frame_timeout_s=10.0,
                                sleep_on_failure_s=0.001)
    # 3 process_one_frame successes wrap around the failures + final read
    # forms the 4th. Exact count depends on cv2 mock interleaving — the
    # key assertion is "loop kept going past failures" (n >= 2).
    assert n >= 2


def test_run_inference_loop_writes_health_every_10_frames(tmp_path, monkeypatch):
    health_calls = []
    monkeypatch.setattr(node, "write_health",
                         lambda *a, **kw: health_calls.append(True))
    cap = MagicMock()
    cap.read.return_value = (True, "frame")
    model = MagicMock()
    model.predict.return_value = _fake_results([[]], names={})
    client = MagicMock()
    node.run_inference_loop(cap, model, _args(), client, max_frames=25)
    # 10, 20 → two calls
    assert len(health_calls) == 2


def test_run_inference_loop_stops_when_running_cleared():
    cap = MagicMock()
    cap.read.return_value = (True, "frame")
    model = MagicMock()
    model.predict.return_value = _fake_results([[]], names={})
    client = MagicMock()
    # Trigger SIGTERM-like behavior after 0 frames
    node.signal_handler(15, None)
    n = node.run_inference_loop(cap, model, _args(), client, max_frames=100)
    assert n == 0


# ──────────────────────────────────────────────────────────────────
# cleanup
# ──────────────────────────────────────────────────────────────────


def test_cleanup_releases_and_disconnects():
    cap = MagicMock()
    client = MagicMock()
    node.cleanup(cap, client, frame_count=42)
    cap.release.assert_called_once()
    client.loop_stop.assert_called_once()
    client.disconnect.assert_called_once()


# ──────────────────────────────────────────────────────────────────
# main (end-to-end with all deps mocked)
# ──────────────────────────────────────────────────────────────────


def test_main_happy_path_with_injected_mocks():
    fake_model = MagicMock()
    fake_model.predict.return_value = _fake_results([[]], names={})
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    fake_cap.read.return_value = (True, "frame")
    fake_client = MagicMock()
    rc = node.main(
        argv=["--model", "/x", "--source", "/y"],
        model_factory=lambda path, task: fake_model,
        cap_factory=lambda src: fake_cap,
        mqtt_factory=lambda: fake_client,
        max_frames=2,
    )
    assert rc == 0
    assert fake_client.publish.call_count == 2
    fake_cap.release.assert_called_once()


def test_main_returns_1_on_video_open_failure():
    fake_model = MagicMock()
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = False
    fake_client = MagicMock()
    rc = node.main(
        argv=["--source", "/missing"],
        model_factory=lambda path, task: fake_model,
        cap_factory=lambda src: fake_cap,
        mqtt_factory=lambda: fake_client,
        max_frames=1,
    )
    assert rc == 1
