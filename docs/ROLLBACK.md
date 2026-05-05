# Rollback Procedure

*Edge-AI HW6 production rollback — by Janlung Sung at Tatung University*

When the production inference container starts misbehaving, this is the
on-call playbook. Goal: revert to the previous known-good tag in **<30 s**.

## When to roll back

Roll back **immediately** if any of these symptoms appear after a deploy:

- `deploy/healthcheck.sh` fails 3 times consecutively (the deploy script
  invokes this automatically and rolls back on its own — but a human
  on-call may still see the alert)
- `/healthz` returns non-200 OR `status != "healthy"` for more than 60 s
- Inference mAP@50 drops more than 5 points vs the deployed baseline
  (compare `calibration/accuracy_baseline.json` against fresh measurement)
- The container is in a `restarting` loop (`docker ps` shows
  `Restarting (...)` for the inference service)
- MQTT detection topic stops publishing for >2 min while the container
  is reportedly `Up`
- nvpmodel reverts to a different mode than `power_profile.json` says
  for the active environment

## The exact rollback command

SSH to the Jetson and run:

```bash
cd ~/edgeai-hw6
time bash deploy/rollback.sh
```

That's it. The script:

1. Reads the previous tag from `/var/lib/edgeai-hw6/deployed.txt.history`
2. `docker compose pull` (no-op if already cached locally)
3. `docker compose up -d --force-recreate` with the previous tag
4. Polls `/healthz` (3 consecutive successes within 60 s)
5. On success: rewrites `deployed.txt` to the previous tag, exits 0
6. On healthcheck failure: leaves the stack as-is, exits 1 (the "two
   broken tags" case below)

Expected wall time: **<10 s** when the previous image is locally cached
(the common case — the previous image was running just minutes ago).

## How to find which tag to roll back *to*

`rollback.sh` automatically picks the last entry from
`/var/lib/edgeai-hw6/deployed.txt.history` (the most-recently-replaced tag).
If you need to roll back further than one step, inspect the history:

```bash
cat /var/lib/edgeai-hw6/deployed.txt.history
```

To roll back to a *specific* older tag (e.g., skip a known-bad
intermediate version):

```bash
cd ~/edgeai-hw6
IMAGE_TAG=v1.2.3 docker compose -f deploy/docker-compose.yml up -d --force-recreate
bash deploy/healthcheck.sh
echo v1.2.3 > /var/lib/edgeai-hw6/deployed.txt   # update state manually
```

You can also list which release tags exist on GHCR:

```bash
gh release list -R jlsung-ttu/edgeai-hw6
```

## What to do if rollback also fails ("two broken tags")

If `rollback.sh` exits non-zero, both the current AND the previous tag
are broken. Neither is serving healthy `/healthz` responses. The compose
stack may be in any state. **Recovery procedure:**

1. **Stop the inference service** to remove ambiguity:
   ```bash
   docker compose -f deploy/docker-compose.yml down
   ```
2. **Pick a known-good tag from earlier in history**:
   ```bash
   cat /var/lib/edgeai-hw6/deployed.txt.history
   ```
   Look for a tag from a deploy that was confirmed-healthy at the time.
3. **Bring up that tag manually** (skip the script):
   ```bash
   cd ~/edgeai-hw6
   IMAGE_TAG=<known-good-tag> docker compose -f deploy/docker-compose.yml up -d
   sleep 15
   curl -s http://localhost:8000/healthz | jq
   ```
4. **If even old tags are unhealthy**, the failure is environmental
   (camera disconnected, /var/lib/nvpmodel/status missing, MQTT broker
   down, GHCR unreachable) — diagnose with:
   ```bash
   docker compose -f deploy/docker-compose.yml logs --tail 50
   sudo nvpmodel -q
   curl -s http://localhost:1883 || echo "mosquitto down"
   docker pull ghcr.io/jlsung-ttu/edgeai-hw6:latest 2>&1 | head -3
   ```
5. **Restore the state file by hand once a working tag is up**:
   ```bash
   echo <known-good-tag> > /var/lib/edgeai-hw6/deployed.txt
   ```

## What to communicate to the team

When you roll back, send this to the team's `#edgeai-ops` channel
(Slack / Teams / email — adapt to your team's channel):

```
🔁 Rolled back production from <BAD_TAG> to <GOOD_TAG>

Symptom: <one-line description, e.g. "healthcheck failing 60s after deploy">
Detected at: <timestamp>
Rollback completed at: <timestamp>
Wall time: <time output from `time bash deploy/rollback.sh`>
Current /healthz: <paste curl response — must show power_mode + healthy>

Root cause investigation: <link to PR / issue / log dump>
Next deploy: held until root cause is fixed and a fresh tag passes
the integration test on main.
```

If the **two-broken-tags** case fired, also escalate:

```
🚨 Both current AND previous tags broken — manual recovery in progress.
Rolled forward to <known-good-older-tag>.
Investigation: <link>
```
