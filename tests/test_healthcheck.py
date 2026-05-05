#!/usr/bin/env python3
# Copyright (c) 2026 Janlung Sung
# Tatung University — I4210 AI實務專題
"""tests/test_healthcheck.py — unit tests for /healthz endpoint helpers.

The HTTP server itself is exercised end-to-end by Part C's integration
test on the Jetson; these unit tests cover the parsing + start_in_thread
plumbing on x86 without needing a real Jetson or live socket.
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src import healthcheck

# --- _current_power_mode (file-based) ---

NVPMODEL_CONF_SAMPLE = """
< POWER_MODEL ID=0 NAME=15W >
< POWER_MODEL ID=1 NAME=7W >
< POWER_MODEL ID=2 NAME=MAXN_SUPER >
"""


def _stage_files(tmp_path: Path, status: str = None, conf: str = NVPMODEL_CONF_SAMPLE):
    """Write status + conf into tmp_path and point healthcheck at them."""
    if status is not None:
        (tmp_path / "status").write_text(status)
        healthcheck._NVPMODEL_STATUS = tmp_path / "status"
    else:
        healthcheck._NVPMODEL_STATUS = tmp_path / "missing-status"
    if conf is not None:
        (tmp_path / "nvpmodel.conf").write_text(conf)
        healthcheck._NVPMODEL_CONF = tmp_path / "nvpmodel.conf"
    else:
        healthcheck._NVPMODEL_CONF = tmp_path / "missing-conf"


def test_current_power_mode_returns_empty_when_status_file_missing(tmp_path):
    """No status file (e.g. on x86 CI) → empty string, not raise."""
    _stage_files(tmp_path, status=None)
    assert healthcheck._current_power_mode() == ""


def test_current_power_mode_returns_empty_when_status_unparseable(tmp_path):
    """Status file present but not in `pmode:NNNN` format → empty string."""
    _stage_files(tmp_path, status="garbage\n")
    assert healthcheck._current_power_mode() == ""


def test_current_power_mode_resolves_id_to_name(tmp_path):
    """pmode:0001 → look up ID=1 in conf → return NAME 7W."""
    _stage_files(tmp_path, status="pmode:0001\n")
    assert healthcheck._current_power_mode() == "7W"


def test_current_power_mode_handles_id_not_in_conf(tmp_path):
    """Active mode ID has no matching entry in conf → empty string."""
    _stage_files(tmp_path, status="pmode:0099\n")
    assert healthcheck._current_power_mode() == ""


def test_current_power_mode_returns_empty_when_conf_missing(tmp_path):
    """Status file exists but conf is missing → empty string, not raise."""
    _stage_files(tmp_path, status="pmode:0000\n", conf=None)
    assert healthcheck._current_power_mode() == ""


def test_current_power_mode_handles_zero_padded_id(tmp_path):
    """pmode:0000 → ID=0 → 15W (proves leading zeros don't break int parse)."""
    _stage_files(tmp_path, status="pmode:0000\n")
    assert healthcheck._current_power_mode() == "15W"


# --- _Handler ---

def _build_handler(path: str) -> healthcheck._Handler:
    """Construct a Handler without running its base __init__ (which would
    try to read from a real socket). We only need do_GET behavior."""
    h = healthcheck._Handler.__new__(healthcheck._Handler)
    h.path = path
    h.wfile = BytesIO()
    h.rfile = BytesIO()
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
    h.log_message("anything", "goes", 42)


# --- start_in_thread ---

def test_start_in_thread_returns_none_on_bind_failure():
    """If port is busy, return None instead of crashing the inference loop."""
    healthcheck._started = None
    with patch.object(healthcheck, "HTTPServer",
                       side_effect=OSError("Address already in use")):
        assert healthcheck.start_in_thread() is None


def test_start_in_thread_is_idempotent():
    """Second call within the same process must reuse the first thread."""
    healthcheck._started = None
    fake_thread = MagicMock()
    fake_thread.is_alive.return_value = True
    healthcheck._started = fake_thread
    second = healthcheck.start_in_thread()
    assert second is fake_thread


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Don't let one test's _started or _NVPMODEL_* leak into another."""
    orig_status = healthcheck._NVPMODEL_STATUS
    orig_conf = healthcheck._NVPMODEL_CONF
    yield
    healthcheck._started = None
    healthcheck._NVPMODEL_STATUS = orig_status
    healthcheck._NVPMODEL_CONF = orig_conf
