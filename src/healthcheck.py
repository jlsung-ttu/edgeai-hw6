#!/usr/bin/env python3
# Copyright (c) 2026 Janlung Sung
# Tatung University — I4210 AI實務專題
"""src/healthcheck.py — minimal /healthz endpoint for the inference container.

Started as a background thread by inference_node.main() so every container
gets the endpoint for free without a sidecar. Exposes a single GET /healthz
returning JSON the deploy-side healthcheck.sh polls.

Reads `power_mode` from `nvpmodel -q` at request time (not from a config
file) so the response reflects what the *Jetson kernel* says is active —
that's what catches "deploy.sh ran but nvpmodel silently no-op'd" failures.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

PORT = int(os.environ.get("HEALTHZ_PORT", "8000"))
# Bind to loopback by default. Production deployments use docker compose's
# `network_mode: host` (deploy/docker-compose.yml), so the host's
# healthcheck.sh reaches us via 127.0.0.1:8000 without needing the
# container to bind on all interfaces. Override to 0.0.0.0 only when a
# bridged-network deployment needs external reachability.
BIND = os.environ.get("HEALTHZ_BIND", "127.0.0.1")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")


def _current_power_mode() -> str:
    """Best-effort read of the live nvpmodel state.

    Returns empty string if nvpmodel isn't on PATH (e.g. running in CI on
    x86, where there's no Jetson). The deploy-side healthcheck.sh treats
    empty as 'unknown' rather than 'failure' — the only fatal condition
    is the HTTP request itself not returning 200.
    """
    try:
        out = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        # nvpmodel -q output line: "NV Power Mode: 15W"
        for line in out.stdout.splitlines():
            if "Power Mode" in line:
                return line.split(":", 1)[1].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


class _Handler(BaseHTTPRequestHandler):
    """One-method HTTP handler — only /healthz GET is recognized."""

    def do_GET(self):
        # Method name is fixed by the stdlib's BaseHTTPRequestHandler API
        # (it dispatches based on `do_<METHOD>`). The pep8-naming check
        # for this is suppressed at the pyproject.toml level (N802 is in
        # ruff's allowed-ignore list once the HW5-style compliance
        # refactor lands; today's ruff config doesn't select N rules at all).
        if self.path != "/healthz":
            self.send_error(404)
            return
        body = json.dumps({
            "status": "healthy",
            "model_version": MODEL_VERSION,
            "power_mode": _current_power_mode(),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        """Silence the per-request stderr spam (one line per healthcheck poll)."""
        pass


_started: Optional[threading.Thread] = None


def start_in_thread() -> Optional[threading.Thread]:
    """Start the healthz server on a daemon thread so it dies with main().

    Binds to BIND:PORT (defaults loopback:8000). With Part D's docker
    compose `network_mode: host`, loopback inside the container IS the
    host's loopback, so deploy/healthcheck.sh reaches us at
    http://localhost:8000/healthz without needing a bind-all-interfaces
    address.

    Idempotent: a second call within the same process returns the existing
    thread instead of trying to re-bind the port. Tolerates `OSError:
    Address already in use` by logging and returning None — that lets
    pytest sessions that call `main()` repeatedly not blow up, and lets
    a production container with a pre-existing sidecar on :8000 still
    boot the inference loop.
    """
    global _started
    if _started is not None and _started.is_alive():
        return _started
    try:
        server = HTTPServer((BIND, PORT), _Handler)
    except OSError as exc:
        print(f"[healthz] WARNING: could not bind {BIND}:{PORT} — {exc}. Skipping.")
        return None
    _started = threading.Thread(target=server.serve_forever, daemon=True, name="healthz")
    _started.start()
    return _started
