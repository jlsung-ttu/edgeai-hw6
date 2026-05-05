#!/usr/bin/env bash
# Copyright (c) 2026 Janlung Sung
# Tatung University — I4210 AI實務專題
# deploy/rollback.sh — revert to the previously-deployed tag in <30 s.
#
# Reads /var/lib/edgeai-hw6/deployed.txt.history (maintained by deploy.sh)
# for the previous tag, then `docker compose up`s the stack with that tag.
# If healthcheck fails, BOTH current and previous are broken — alert and
# leave the stack stopped (manual intervention required).
#
# Usage:  bash deploy/rollback.sh
# Called by:
#   - Operator: SSH to Jetson, run manually when symptoms hit (see ROLLBACK.md)
#   - deploy.sh: automatic invocation when post-deploy healthcheck fails
set -euo pipefail

STATE_DIR=/var/lib/edgeai-hw6
CURRENT_FILE="$STATE_DIR/deployed.txt"
HISTORY_FILE="$STATE_DIR/deployed.txt.history"

# 1. Find the previous tag (last line of history file — most-recently-replaced).
if [ ! -f "$HISTORY_FILE" ]; then
  echo "[rollback] FATAL: no history file at $HISTORY_FILE" >&2
  echo "[rollback] Nothing to roll back to. Was deploy.sh ever run successfully?" >&2
  exit 1
fi
PREV=$(tail -n 1 "$HISTORY_FILE")
if [ -z "$PREV" ]; then
  echo "[rollback] FATAL: history file is empty" >&2
  exit 1
fi
CURRENT=$(cat "$CURRENT_FILE" 2>/dev/null || echo "<unknown>")
echo "[rollback] Current=$CURRENT  →  Rolling back to PREV=$PREV"

# 2. Pull the previous tag (no-op if locally cached, which is the common case).
export IMAGE_TAG="$PREV"
docker compose -f deploy/docker-compose.yml pull

# 3. Recreate the container with the previous tag.
docker compose -f deploy/docker-compose.yml up -d --force-recreate

# 4. Healthcheck. If it fails, BOTH current AND previous are broken — the
#    "two broken tags" case from ROLLBACK.md. Don't update state file in
#    that case so a human can investigate without losing the audit trail.
if ! bash deploy/healthcheck.sh; then
  echo "[rollback] FATAL: previous tag $PREV is also unhealthy." >&2
  echo "[rollback] Both current and previous are broken — manual intervention required." >&2
  echo "[rollback] See docs/ROLLBACK.md → 'Two broken tags' section." >&2
  exit 1
fi

# 5. Mark the previous tag as the new current. Leave history intact so
#    future rollbacks (or audits) can step further back.
echo "$PREV" > "$CURRENT_FILE"
echo "[rollback] Successfully rolled back to $PREV in ${SECONDS}s wall time."
