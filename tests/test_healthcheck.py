#!/usr/bin/env python3
# Copyright (c) 2026 Janlung Sung
# Tatung University — I4210 AI實務專題
"""tests/test_healthcheck.py — unit tests for /healthz endpoint helpers.

The HTTP server itself is exercised end-to-end by Part C's integration
test on the Jetson; these unit tests cover the parsing + start_in_thread
plumbing on x86 without needing a real nvpmodel binary or a live socket.
"""
from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from src import healthcheck

# --- _current_power_mode ---

def test_current_power_mode_returns_empty_when_nvpmodel_missing():
    """No nvpmodel binary on PATH (e.g. CI on x86) must return '' not raise."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert healthcheck._current_power_mode() == ""


def test_current_power_mode_returns_empty_on_timeout():
    """If nvpmodel hangs, we must not block /healthz forever."""
    import subprocess
    with patch("subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="nvpmodel", timeout=2)):
        assert healthcheck._current_power_mode() == ""


def test_current_power_mode_parses_nvpmodel_output():
    """Real nvpmodel -q output: extracts the value after 'Power Mode:'."""
    fake = MagicMock()
    fake.stdout = "NV Power Mode: 15W\nNV Fan Mode: quiet\n"
    with patch("subprocess.run", return_value=fake):
        assert healthcheck._current_power_mode() == "15W"


def test_current_power_mode_handles_missing_power_line():
    """nvpmodel output without a Power Mode line returns ''."""
    fake = MagicMock()
    fake.stdout = "some other output\nno power mode here\n"
    with patch("subprocess.run", return_value=fake):
        assert healthcheck._current_power_mode() == ""


# --- _Handler ---

def _build_handler(path: str) -> healthcheck._Handler:
    """Construct a Handler without running its base __init__ (which would
    try to read from a real socket). We only need do_GET behavior."""
    h = healthcheck._Handler.__new__(healthcheck._Handler)
    h.path = path
    h.wfile = BytesIO()
    h.rfile = BytesIO()
    # Capture send_response/send_header/end_headers/send_error calls
    h.send_response = MagicMock()
    h.send_header = MagicMock()
    h.end_headers = MagicMock()
    h.send_error = MagicMock()
    return h


def test_handler_returns_200_and_correct_json_for_healthz():
    """GET /healthz must return 200 + JSON with the documented schema."""
    h = _build_handler("/healthz")
    with patch.object(healthcheck, "_current_power_mode", return_value="15W"):
        h.do_GET()
    h.send_response.assert_called_once_with(200)
    body = h.wfile.getvalue().decode()
    payload = json.loads(body)
    assert payload["status"] == "healthy"
    assert payload["power_mode"] == "15W"
    assert "model_version" in payload


def test_handler_returns_404_for_unknown_path():
    """Anything other than /healthz must 404, not crash."""
    h = _build_handler("/")
    h.do_GET()
    h.send_error.assert_called_once_with(404)


def test_handler_log_message_is_silent():
    """log_message override must not raise on any args."""
    h = _build_handler("/healthz")
    h.log_message("anything", "goes", 42)   # must not raise


# --- start_in_thread ---

def test_start_in_thread_returns_none_on_bind_failure():
    """If port is busy, return None instead of crashing the inference loop."""
    healthcheck._started = None   # reset module state
    with patch.object(healthcheck, "HTTPServer",
                       side_effect=OSError("Address already in use")):
        assert healthcheck.start_in_thread() is None


def test_start_in_thread_is_idempotent():
    """Second call within the same process must reuse the first thread.

    We pre-seed _started with a fake-alive thread instead of running a
    real one; otherwise the MagicMock's serve_forever returns immediately
    and the thread dies, forcing the function (correctly) to start a new
    one. The behavior under test here is the early-return path, not the
    bind-and-start path.
    """
    healthcheck._started = None
    fake_thread = MagicMock()
    fake_thread.is_alive.return_value = True
    healthcheck._started = fake_thread
    second = healthcheck.start_in_thread()
    assert second is fake_thread


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Don't let one test's _started leak into another."""
    yield
    healthcheck._started = None
