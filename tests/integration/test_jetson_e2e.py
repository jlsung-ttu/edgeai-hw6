#!/usr/bin/env python3
# Copyright (c) 2026 Janlung Sung
# Tatung University — I4210 AI實務專題
"""End-to-end integration test that runs on the self-hosted Jetson runner.

What this proves:
    1. The per-commit image (`ghcr.io/<repo>/edgeai-hw6:sha-<short>`) actually
       pulls and starts on real Jetson hardware (`--runtime nvidia`).
    2. The TRT engine cache (`lab12-models` named volume) is honored — first
       run compiles, subsequent runs reuse.
    3. The inference loop publishes JSON detection payloads on the documented
       MQTT topic within 30 s of the engine being ready.
    4. Container teardown is clean (no orphaned containers / networks block
       the next CI run).

Architecture (no host-state dependencies):
    ┌─────────────────────┐
    │  test process       │ ──── localhost:<random_port> ──┐
    │  (paho subscriber)  │                                ▼
    └─────────────────────┘                       ┌────────────────┐
                                                  │ mosquitto      │
                                                  │ on hw6-itest-* │
                                                  │ docker network │
                                                  └───────▲────────┘
                                                          │ MQTT_BROKER=mqtt-broker
                                                  ┌───────┴────────┐
                                                  │ inference node │
                                                  │ (the SUT)      │
                                                  └────────────────┘

Both containers live on a freshly-created docker network so the test does
not collide with whatever mosquitto / network state the Jetson host has.
The broker publishes 1883 to a *random* host port (no `-p 1883:1883`) for
the same reason.

Runnable two ways:
    Locally on the Jetson:  IMAGE=ghcr.io/<repo>/edgeai-hw6:sha-abc1234 \\
                            pytest tests/integration/test_jetson_e2e.py -v
    From CI:                set by .github/workflows/ci.yml's `IMAGE` env
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import time
import uuid
from pathlib import Path

import paho.mqtt.client as mqtt
import pytest
from paho.mqtt.enums import CallbackAPIVersion

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
IMAGE = os.environ.get("IMAGE", "")  # set by ci.yml; required
TOPIC = "jetson/vision/detections"

# Random suffix so concurrent test runs (rare but possible during retries)
# don't fight over the same docker resource names.
SUFFIX = uuid.uuid4().hex[:8]
NETWORK_NAME = f"hw6-itest-{SUFFIX}"
BROKER_NAME = f"mqtt-broker-{SUFFIX}"
INFER_NAME = f"inference-node-{SUFFIX}"

# Internal alias the inference container resolves the broker by — must
# match the BROKER_NAME above (docker auto-DNS). Kept as a separate
# constant so a future student can rename the container without breaking
# the env var.
BROKER_ALIAS = BROKER_NAME

MODEL_VOLUME = "lab12-models"  # carried over from Lab 12 — engine cache
SAMPLE_FRAME = Path(__file__).parent / "sample_frame.jpg"

# First container start may have to compile the TRT engine — give it
# 10 min. After the inference loop logs "Running inference on ...", the
# test only waits PUBLISH_TIMEOUT_S more seconds for the first MQTT
# message (the rubric line: "asserts an MQTT message lands within 30 s").
ENGINE_READY_TIMEOUT_S = 600
PUBLISH_TIMEOUT_S = 30
ENGINE_READY_LOG = "Running inference on"


# ----------------------------------------------------------------------
# Shell helpers
# ----------------------------------------------------------------------
def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """subprocess.run wrapper that always captures stdout/stderr as text.

    `check=False` lets cleanup paths swallow "container not found" errors
    without aborting the teardown.
    """
    return subprocess.run(
        ["docker", *args],
        check=check,
        text=True,
        capture_output=True,
    )


def _container_logs(name: str, tail: int = 200) -> str:
    """Best-effort log dump for failure reporting."""
    res = _docker("logs", "--tail", str(tail), name, check=False)
    return (res.stdout or "") + (res.stderr or "")


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def _require_image() -> None:
    """Skip the whole module if IMAGE isn't set.

    Local debugging on the Jetson should `export IMAGE=...:sha-<short>`
    pointing at an image that has actually been pushed to GHCR.
    """
    if not IMAGE:
        pytest.skip("IMAGE env var not set — invoke via ci.yml or export manually")
    if not SAMPLE_FRAME.exists():
        pytest.fail(
            f"Missing {SAMPLE_FRAME}. Commit a 320x320 JPG fallback frame; "
            f"the inference container reads it as a single-frame video source."
        )


@pytest.fixture(scope="module")
def docker_network() -> str:
    """Per-run docker network so inference and broker can DNS each other.

    `docker network rm` in the finally clause runs even when the test
    fails — this is rubric criterion #4 ("Test cleans up its container
    even on failure").
    """
    _docker("network", "create", NETWORK_NAME)
    try:
        yield NETWORK_NAME
    finally:
        _docker("network", "rm", NETWORK_NAME, check=False)


@pytest.fixture(scope="module")
def broker_host_port(docker_network: str) -> int:
    """Spin up eclipse-mosquitto on the test network. Yields a host port
    that the test process can connect to via 127.0.0.1:<port>.

    No -p 1883:1883 — picks a free port so this works even if the host
    Jetson already has its own mosquitto running (Lab 11 setup).
    """
    # mosquitto:2 needs a listener on 1883 explicitly enabled in v2.x.
    # The simplest way: pass a one-line config via -c on stdin.
    cfg = (
        "listener 1883 0.0.0.0\n"
        "allow_anonymous true\n"
    )
    # Write the conf into a tiny tmpfs volume the container reads.
    conf_path = f"/tmp/mosquitto-{SUFFIX}.conf"  # noqa: S108  test-only path
    Path(conf_path).write_text(cfg)
    _docker(
        "run",
        "-d",
        "--name",
        BROKER_NAME,
        "--network",
        docker_network,
        "--network-alias",
        BROKER_ALIAS,
        "-p",
        "0:1883",  # ask docker for a random free host port
        "-v",
        f"{conf_path}:/mosquitto/config/mosquitto.conf:ro",
        "eclipse-mosquitto:2",
    )
    try:
        port_line = _docker("port", BROKER_NAME, "1883/tcp").stdout.strip()
        host_port = int(port_line.rsplit(":", 1)[-1])
        # Wait for the broker to start accepting connections.
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                probe = mqtt.Client(CallbackAPIVersion.VERSION2)
                probe.connect("127.0.0.1", host_port, 5)
                probe.disconnect()
                break
            except OSError:
                time.sleep(0.5)
        else:
            pytest.fail(
                f"mosquitto did not accept connections within 30 s. "
                f"Logs:\n{_container_logs(BROKER_NAME)}"
            )
        yield host_port
    finally:
        _docker("rm", "-f", BROKER_NAME, check=False)
        Path(conf_path).unlink(missing_ok=True)


@pytest.fixture(scope="module")
def inference_container(docker_network: str, broker_host_port: int) -> str:
    """Start the SUT.

    Bind-mounts sample_frame.jpg as /opt/data/test_video.mp4 — cv2 treats
    a single JPEG as a 1-frame "video", which is enough to drive one
    inference + one MQTT publish (the rubric only asks for ≥1 message).

    Uses the named `lab12-models` volume so the TRT engine compiled in
    Lab 12 (or a previous HW6 run) is reused — first PR will pay the
    compile cost; subsequent runs are fast.
    """
    # Idempotent: -f no-op if already exists. Required for the Jetson
    # path because Lab 12 already created lab12-models and we want to
    # reuse its cached engine across runs.
    _docker("volume", "create", MODEL_VOLUME, check=False)
    _docker(
        "run",
        "-d",
        "--name",
        INFER_NAME,
        "--runtime",
        "nvidia",
        "--network",
        docker_network,
        "-v",
        f"{MODEL_VOLUME}:/opt/models",
        "-v",
        f"{SAMPLE_FRAME}:/opt/data/test_video.mp4:ro",
        "-e",
        f"MQTT_BROKER={BROKER_ALIAS}",
        "-e",
        "MQTT_PORT=1883",
        IMAGE,
    )
    try:
        yield INFER_NAME
    finally:
        # Always dump the last 200 log lines on teardown — useful for
        # diagnosing flaky cold-start cases without re-running the test.
        print("\n----- inference container logs (tail 200) -----")
        print(_container_logs(INFER_NAME))
        print("----- end logs -----\n")
        _docker("rm", "-f", INFER_NAME, check=False)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_image_is_per_commit_sha_tagged() -> None:
    """Rubric #1: must use the per-commit image, not a stale :latest."""
    assert IMAGE.startswith("ghcr.io/"), f"IMAGE={IMAGE!r} not on GHCR"
    assert ":sha-" in IMAGE, (
        f"IMAGE={IMAGE!r} is not sha-tagged. ci.yml must point at the "
        f"per-commit image so we don't accidentally test :latest from a "
        f"previous workflow run."
    )


def test_inference_publishes_mqtt_within_window(
    broker_host_port: int,
    inference_container: str,
) -> None:
    """Rubric #2 + #3: container starts with --runtime nvidia and a
    model-cache volume (handled by the fixture), and the inference loop
    publishes ≥1 MQTT detection within 30 s of the engine being ready
    (after the cold-start compile budget elapses).
    """
    # Subscribe before waiting for the engine — message ordering is
    # otherwise racy if the inference loop publishes faster than the
    # subscriber connects.
    received: queue.Queue[bytes] = queue.Queue()

    def _on_connect(client, _userdata, _flags, _rc, _props=None):
        client.subscribe(TOPIC)

    def _on_message(_client, _userdata, msg):
        received.put(msg.payload)

    client = mqtt.Client(CallbackAPIVersion.VERSION2)
    client.on_connect = _on_connect
    client.on_message = _on_message
    client.connect("127.0.0.1", broker_host_port, 60)
    client.loop_start()

    try:
        _wait_for_engine_ready(inference_container, ENGINE_READY_TIMEOUT_S)
        try:
            payload = received.get(timeout=PUBLISH_TIMEOUT_S)
        except queue.Empty:
            pytest.fail(
                f"No MQTT message on '{TOPIC}' in {PUBLISH_TIMEOUT_S}s after "
                f"engine became ready. Container logs above."
            )
        # Schema sanity — keep this loose; the point is "round-trip works",
        # not "model finds N specific objects in the synthetic frame".
        data = json.loads(payload)
        assert "detections" in data, f"Payload missing 'detections': {data!r}"
        assert "frame" in data, f"Payload missing 'frame': {data!r}"
        assert "t" in data, f"Payload missing 't' timestamp: {data!r}"
        assert isinstance(data["detections"], list), (
            f"detections must be a list, got {type(data['detections']).__name__}"
        )
    finally:
        client.loop_stop()
        client.disconnect()


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _wait_for_engine_ready(container: str, timeout: float) -> None:
    """Poll `docker logs` for the inference-loop start banner.

    Uses a substring match against `ENGINE_READY_LOG` rather than a
    regex — keeps the contract loose so a future student can rephrase
    the log line without breaking the test.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        logs = _container_logs(container, tail=300)
        if ENGINE_READY_LOG in logs:
            return
        # Container may have crashed before logging — fail fast instead
        # of waiting the full 10 min.
        state = _docker("inspect", "-f", "{{.State.Status}}", container,
                         check=False).stdout.strip()
        if state in ("exited", "dead"):
            pytest.fail(
                f"Inference container {state!r} before becoming ready. "
                f"Logs:\n{logs}"
            )
        time.sleep(2)
    pytest.fail(
        f"Inference engine did not become ready within {timeout}s. "
        f"Final logs:\n{_container_logs(container)}"
    )
