#!/usr/bin/env bash
# Copyright (c) 2026 Janlung Sung
# Tatung University — I4210 AI實務專題
# deploy/deploy.sh — Pull tag → set nvpmodel → restart compose → healthcheck → rollback on fail.
#
# Usage:        DEPLOY_ENV=production bash deploy/deploy.sh <vX.Y.Z|sha-XXXXXXX>
# Triggered by: deploy.yml (Part D6) SSHes into the Jetson and runs this.
#
# Exit 0 → new tag is live and serving healthy /healthz responses.
# Exit non-zero → healthcheck failed; rollback was attempted (Part E).
set -euo pipefail

TAG="${1:?Usage: deploy.sh <vX.Y.Z|sha-XXXXXXX>}"
ENV="${DEPLOY_ENV:-production}"
STATE_DIR=/var/lib/edgeai-hw6
sudo mkdir -p "$STATE_DIR"

# 1. Resolve the configured power-mode NAME → numeric ID for THIS Jetson SKU.
#    (Original Orin Nano: 7W/15W/MAXN; Orin Nano Super: 7W/15W/25W/MAXN_SUPER.)
MODE_NAME=$(jq -r ".\"$ENV\"" deploy/power_profile.json)
MODE_ID=$(grep -oE "<\s*POWER_MODEL\s+ID=[0-9]+\s+NAME=$MODE_NAME\s*>" /etc/nvpmodel.conf \
          | grep -oE "ID=[0-9]+" | cut -d= -f2 | head -1)
if [ -z "$MODE_ID" ]; then
  echo "[deploy] ERROR: power mode '$MODE_NAME' not found in /etc/nvpmodel.conf"
  echo "         Available modes:"
  grep -oE "<\s*POWER_MODEL\s+ID=[0-9]+\s+NAME=\S+\s*>" /etc/nvpmodel.conf
  exit 1
fi
echo "[deploy] Setting nvpmodel to $MODE_NAME (ID=$MODE_ID) for env=$ENV"
sudo nvpmodel -m "$MODE_ID"
sudo jetson_clocks
sleep 2

# 2. Save the currently-deployed tag for Part E's rollback.sh.
#    `sudo tee` because /var/lib/edgeai-hw6/ is owned by root — bare `>`
#    would fail with permission denied even under sudo, since the redirect
#    happens in the parent shell before sudo elevates.
if [ -f "$STATE_DIR/deployed.txt" ]; then
  PREV=$(sudo cat "$STATE_DIR/deployed.txt")
  echo "$PREV" | sudo tee -a "$STATE_DIR/deployed.txt.history" >/dev/null
  echo "[deploy] Previous tag was $PREV (saved for rollback)"
fi

# 3. Pull the requested tag, recreate the inference container.
export IMAGE_TAG="$TAG"
docker compose -f deploy/docker-compose.yml pull
docker compose -f deploy/docker-compose.yml up -d --force-recreate

# 4. Wait for health (D3); roll back on fail (Part E hooks here).
if ! bash deploy/healthcheck.sh; then
  echo "[deploy] Healthcheck failed — rolling back"
  if [ -x deploy/rollback.sh ]; then
    bash deploy/rollback.sh
  else
    echo "[deploy] WARNING: deploy/rollback.sh not yet implemented (Part E)"
  fi
  exit 1
fi

# 5. Mark this tag as the new current.
echo "$TAG" | sudo tee "$STATE_DIR/deployed.txt" >/dev/null
echo "[deploy] Deployed $TAG at power mode $MODE_NAME"
