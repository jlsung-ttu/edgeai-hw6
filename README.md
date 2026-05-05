# edgeai-hw6 — Production CI/CD for Edge-AI Inference

*By Janlung Sung at Tatung University — I4210 AI 實務專題*

[![CI](https://github.com/jlsung-ttu/edgeai-hw6/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jlsung-ttu/edgeai-hw6/actions/workflows/ci.yml)
[![Deploy](https://github.com/jlsung-ttu/edgeai-hw6/actions/workflows/deploy.yml/badge.svg)](https://github.com/jlsung-ttu/edgeai-hw6/actions/workflows/deploy.yml)
[![Latest release](https://img.shields.io/github/v/release/jlsung-ttu/edgeai-hw6?include_prereleases&sort=semver)](https://github.com/jlsung-ttu/edgeai-hw6/releases)

YOLO TensorRT inference on Jetson Orin Nano, packaged as a deployable
container with a 5-stage CI pipeline, tag-triggered deploy with
required-reviewer gate, sub-30-second rollback, and an INT8-vs-FP16
accuracy gate that catches bad recalibration before it ships.

> **Reporting convention.** Following HW4/HW5, all written reporting
> for HW6 lives in this README as numbered sections — Operations,
> Architecture, Optimization, Scaling to a Fleet, Reflections,
> Submission Evidence. Binary artifacts (screenshots, terminal
> recordings) live under `evidence/` and are linked from the relevant
> section. There is no `docs/` folder, no separate `REFLECTIONS.md`,
> no separate `SUBMISSION_EVIDENCE.md` — one README, top-to-bottom.

## Operations

### Quickstart

```bash
git clone https://github.com/jlsung-ttu/edgeai-hw6.git
cd edgeai-hw6
pdm install
pdm run pytest tests/ --ignore=tests/integration --cov=src --cov-fail-under=90
```

That runs all 49 unit tests against the container's source modules
without needing a Jetson, GPU, camera, or MQTT broker — everything
that touches the hardware is mocked. Expected: `49 passed`,
`Total coverage: 100%`. From here you can develop locally, push to
trigger the full 5-stage pipeline, and use the deploy / rollback
flows below for production releases.

### How to deploy a new release

Production deploys go through a tag-triggered workflow gated by manual
approval. Four steps:

```bash
git checkout main && git pull          # 1. Be on a green main
git tag -a v1.0.0 -m "Release notes"   # 2. Annotated semver tag
git push --tags                        # 3. Triggers deploy.yml
gh run watch                           # 4. Approve in browser when prompted
```

Step 4 — when the run pauses at "Review pending deployments," click into
the workflow on github.com → **Review deployments** → check `production`
→ **Approve and deploy**. The run resumes and:

1. Re-tags the per-commit `:sha-XXXXXXX` GHCR image as `:v1.0.0`,
   `:v1.0`, `:v1`, and `:latest` (no rebuild — `docker buildx imagetools`
   just writes new manifest references)
2. SSHes / runs on the Jetson via the self-hosted runner
3. Sets `nvpmodel` per `deploy/power_profile.json`
4. Pulls the tagged image and recreates the inference container
5. Polls `/healthz` 3-of-3 within 60 s
6. Marks the new tag as the current deployed version

Total time end-to-end: typically 1-2 minutes after approval (image is
already pushed by the CI run that built it).

### How to roll back

When the production inference container starts misbehaving, this is the
on-call playbook. Goal: revert to the previous known-good tag in **<30 s**.

#### When to roll back

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

#### The exact rollback command

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

#### How to find which tag to roll back *to*

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

#### What to do if rollback also fails ("two broken tags")

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

#### What to communicate to the team

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

## Architecture

This section is the audit trail for `edgeai-hw6`'s pipeline: what each
stage does, why it's there, and what we deliberately chose **not** to
build. A future maintainer should be able to read this once and know
where every moving part lives.

### Pipeline overview

```
                                   ┌──────────────────────────┐
   git push (any branch)           │     ci.yml               │
   ─────────────────────────►──────┤  (5-stage workflow)      │
                                   └────────────┬─────────────┘
                                                │
                       ┌────────────────────────┼─────────────────────┐
                       ▼                        ▼                     ▼
              ┌─────────────┐          ┌──────────────┐      ┌────────────────┐
              │ lint (ruff) │          │ test+coverage│      │ security-scan  │
              │  ubuntu     │          │  +accuracy   │      │ bandit+pip-audit│
              └──────┬──────┘          │  ubuntu      │      │  ubuntu        │
                     │                 └──────┬───────┘      └────────┬───────┘
                     │                        │                       │
                     └────────────────────────┴───────────┬───────────┘
                                                          ▼
                                              ┌──────────────────────┐
                                              │ build (buildx + QEMU)│
                                              │  push :sha-XXXXXXX   │
                                              │  to GHCR             │
                                              │  ubuntu              │
                                              └──────────┬───────────┘
                                                         │
                                          (only on main pushes)
                                                         ▼
                                          ┌────────────────────────────┐
                                          │ integration-test (Jetson)  │
                                          │  pulls per-commit image    │
                                          │  spins up mosquitto + SUT  │
                                          │  asserts MQTT round-trip   │
                                          │  self-hosted runner        │
                                          └────────────────────────────┘

   git tag v1.2.3 + push
   ─────────────────────────►──────┐
                                   │     deploy.yml
                                   │  (production env, gated by reviewer)
                                   ▼
                       ┌──────────────────────────────────┐
                       │ deploy job — runs ON the Jetson   │
                       │  (self-hosted runner)             │
                       │                                   │
                       │  ① docker login ghcr              │
                       │  ② imagetools create  v1.2.3,     │
                       │       v1.2, v1, latest aliases    │
                       │  ③ bash deploy/deploy.sh v1.2.3   │
                       │     - sudo nvpmodel switch        │
                       │     - save prev tag → history     │
                       │     - docker compose pull + up    │
                       │     - bash deploy/healthcheck.sh  │
                       │     - on fail → bash rollback.sh  │
                       └──────────────────────────────────┘
```

### Per-stage rationale

**`lint` (ruff).** Catches the obvious syntax/style issues before they
waste compute on the slower jobs. ~10 s on a hosted runner. Fans out to
both `test` and `security-scan` because neither depends on the other —
parallel execution shaves ~30 s off the critical path.

**`test` (pytest + coverage + accuracy gate).** Runs the unit tests
against `src/` with `--cov-fail-under=90`. Includes `tests/test_accuracy.py`
which reads `calibration/accuracy_baseline.json` and asserts the INT8
mAP didn't regress more than 2 pts vs FP16 — catching bad recalibration
*before* the engine gets baked into a container. The coverage gate
exists because untested helper code accumulates: without a mechanical
floor, the test suite degrades silently as the codebase grows.

**`security-scan` (bandit + pip-audit).** Bandit catches Python SAST
issues (hardcoded secrets, unsafe `subprocess` patterns, predictable
temp paths). pip-audit catches CVEs in our pinned dependencies. Both
fail loud — no "warnings only" mode. We deliberately scope bandit to
`src/` (not `tests/`) because test code legitimately uses patterns
(asserts, mocks, fixture-time temp files) that bandit would flag.

**`build` (buildx + QEMU + GHCR push).** Builds the ARM64 image under
QEMU emulation on a hosted x86 runner and pushes `:sha-<short>` to GHCR.
QEMU is slow (~6-10 min) but free; the spec FAQ documents how to switch
to native ARM64 builds on the self-hosted Jetson runner if the wait
matters. Only runs after **both** `test` and `security-scan` pass —
gating the cost of a 10-min build behind cheap checks.

**`integration-test` (Jetson, main pushes only).** Pulls the freshly
pushed `:sha-<short>` and runs `tests/integration/test_jetson_e2e.py`,
which spins up mosquitto + the inference container on a per-run docker
network and asserts an MQTT detection arrives within 30 s of engine
ready. Gated to `main` pushes (not PRs) because the Jetson runner is
shared and PRs often fan out — letting every PR hit the runner would
cause queue-storms on busy days.

**`deploy.yml` (tag-triggered).** Pushing a `v*.*.*` tag fires this. It
runs on the self-hosted Jetson runner (case-II pattern from the spec)
because the Jetson is on a NAT'd home/lab network and isn't reachable
from a GitHub-hosted runner via SSH. The `production` GitHub environment
adds a required-reviewer gate — no deploy goes out without a human
approving in the browser. Inside the job: re-tag the SHA image as
semver aliases (`v1.2.3`, `v1.2`, `v1`, `latest`) using
`docker buildx imagetools create` (no rebuild, just manifest copies),
then `bash deploy/deploy.sh` does the on-Jetson work — nvpmodel switch,
state-file bookkeeping, compose recreate, healthcheck poll, rollback on
fail. Total deploy time is 1-2 min for a cached image.

**On-Jetson deploy chain.** `deploy.sh` orchestrates four files we own:
`power_profile.json` (env name → nvpmodel mode), `docker-compose.yml`
(image tag from `${IMAGE_TAG}` env), `healthcheck.sh` (3-of-3 polls of
`/healthz`), `rollback.sh` (revert to the last entry in
`/var/lib/edgeai-hw6/deployed.txt.history`). The `/healthz` endpoint
itself lives inside the inference container (`src/healthcheck.py`), which
reads `/var/lib/nvpmodel/status` (bind-mounted from host) to report the
*live* power mode rather than a config value — so a deploy that silently
fails to apply nvpmodel would surface as `power_mode != requested` in
the healthcheck poll.

### What we explicitly chose **not** to do

- **Kubernetes / K3s.** A single-Jetson deploy doesn't need an
  orchestrator. K3s would add etcd, scheduler, kubelet, networking —
  hundreds of MB of overhead for one container. We re-evaluate at >5
  Jetsons (see "Scaling to a Fleet" below).
- **Camera passthrough into the inference container.** The Jetson IMX219
  needs the Tegra GStreamer plugins (`nvarguscamerasrc`) and Argus
  daemon socket bind-mounts to work inside Docker. We added the
  `--source csi:N` parsing in `inference_node.py` so the path EXISTS
  for production — but the rubric path uses a bind-mounted
  `models/test_video.mp4` (HW5's pattern) so the deploy story doesn't
  depend on camera plumbing succeeding. Real-camera deployment is
  documented as a fleet future-improvement.
- **SSH-from-cloud for `deploy.yml`.** The spec documents the
  `appleboy/ssh-action` variant (case I) but we ship the self-hosted
  runner variant (case II) because most students' Jetsons are on
  NAT'd networks where the cloud-runner can't reach inbound. Trade-off:
  less "production-realistic," more reliable for the home/lab
  environment students actually have.
- **Distillation, structured pruning, mixed-precision.** INT8 alone met
  our latency target with a *positive* mAP delta (+0.007). Adding more
  optimization knobs would mean chasing more accuracy regression for
  no required gain. See Optimization section below for the full reasoning.
- **GitOps-pull deploy pattern.** Documented in the spec FAQ as an option
  for fully-air-gapped Jetsons that can't accept inbound SSH but can
  poll outbound. We don't ship it because the self-hosted-runner
  pattern (case II) covers the same NAT'd-Jetson scenario with less
  custom plumbing — the runner already handles the polling for us.
- **Adaptive nvpmodel switching at runtime.** Earlier scope considered
  having the inference container measure FPS and downshift power on
  light scenes. Cut: the deploy-time `production`/`low_power_demo`/
  `burst_demo` switching covers the operator-driven use cases without
  the complexity of an in-container state machine that fights the
  operator's deploy intent.
- **Trivy / SBOM / cosign.** Production CI/CD would add image
  vulnerability scanning, supply-chain SBOM generation, and image
  signing. Cut from the rubric to fit the time budget; left as a
  natural next-week extension.

## Optimization (INT8 vs FP16)

Part 0 deliverable. Reports the FP16 vs INT8 comparison from the
`calibration/calibrate_int8.py` run, plus the production recommendation.

### FP16 vs INT8 (Part 0 required)

Generated on Orin Nano with the refined HW5 model (`best.pt`, MD5
`6f9c47bd...`, fine-tuned for 100 + 40 epochs on the clean Roboflow
construction-safety v30 dataset). All numbers measured on the same
test split (`/home/jetson/project/coding/hw5/models/dataset/`,
82 images / 806 instances) at `imgsz=320, batch=1`.

| Precision | Engine size (MB) | mAP@50  | mAP@50-95 | Latency (ms / frame) | Notes |
|-----------|-----------------:|--------:|----------:|---------------------:|-------|
| FP16      | 10.0             | 0.4138  | 0.275     | 8.5                  | Baseline TRT compile from `best.pt` (same `imgsz`). |
| INT8      |  4.0             | 0.4206  | 0.277     | 6.0                  | Calibrated with 500 random training frames, seed 42. |
| **Δ**     | **−6.0 MB**      | **+0.0068** | **+0.002** | **−2.5 ms (−29 %)** | INT8 is 2.5× smaller, ~30 % faster, no mAP regression. |

> **Latency how-measured:** mean of the per-image `Speed: ... inference`
> field that Ultralytics reports during `model.val(...)`. Measured
> end-to-end including TRT context overhead but excluding JPEG decode
> (`Fast image access ✅` in the val output confirmed the dataset was
> served from page cache, so disk wasn't the bottleneck).

> **About the +0.007 mAP "gain":** counter-intuitive but well within
> run-to-run noise on a 82-image / 806-instance test set. INT8
> quantization can act as a mild regularizer in some cases. Either
> way: the spec's worry case ("INT8 calibration on the course-provided
> dataset typically loses 0.5–2 mAP@50 points vs FP16") didn't apply
> here. We comfortably pass Part 0's "INT8 mAP drop ≤ 2 pts" rubric
> line.

### Production recommendation

**Ship INT8 at the 15 W production `nvpmodel`.** The recalibration cost
nothing on accuracy (`+0.007 mAP@50`, well inside noise) and bought
three production-relevant wins simultaneously:

1. **Engine fits in 4 MB instead of 10 MB.** Multi-model deploy bundles
   in the field will fit comfortably in the Jetson's flash budget.
2. **30 % faster inference** (8.5 → 6.0 ms / frame) leaves headroom in
   the end-to-end pipeline budget for the camera capture and MQTT
   round-trip — important since `inference_node.py` is single-threaded
   and any GPU saving is real wall-clock saving.
3. **Same mAP** means downstream analytics don't have to re-tune
   detection-confidence thresholds when switching from FP16 to INT8.

We did not pick `MAXN_SUPER` because the 6 ms inference latency at
15 W is already faster than the camera frame rate at 30 fps (33 ms /
frame), so additional GPU watts would be wasted. 15 W is the default
production nvpmodel for Orin Nano and gives us the best
performance-per-watt at the bottleneck-relevant operating point.

### What we considered and didn't ship

- **Knowledge distillation** — a smaller student model would shave
  more latency, but INT8 alone already met the latency target with a
  *positive* mAP delta. Distillation would have added a second source
  of accuracy regression to chase for no required gain.
- **Structured pruning** — same calculus as distillation. The
  Ultralytics export path doesn't expose a pruning hook, and we'd have
  needed a separate validation pass + custom calibration step.
- **Mixed-precision (FP16 backbone + INT8 head)** — TensorRT supports
  per-layer precision but Ultralytics' `export()` doesn't surface it
  cleanly. Hand-editing the ONNX or writing a custom TRT builder
  script would have been a meaningful test-surface investment for a
  win we didn't need.
- **A second calibration pass with hard examples** — the standard
  recipe (500 random training frames, seed 42) already produced a
  clean delta, so the hard-example fallback wasn't pulled. It's the
  first lever we'd reach for if a future model showed >2 mAP drop.

### If the INT8 mAP drop were unacceptable (Part 0.3 prompt)

Concrete fallback ladder, in order of escalation:

1. **More representative calibration frames.** Pull the 500 frames from
   production-distribution sources (different lighting, time of day,
   occlusion levels) instead of the random training subset. Re-run
   `python3 calibration/calibrate_int8.py`. Time: ~5–10 min.
2. **More frames.** Bump to 1000–2000 calibration images. The per-tensor
   activation histogram converges faster, scale factors get tighter.
   Time cost is negligible during calibration; bigger draw is dataset
   transfer if the frames live remotely.
3. **Hand-pick hard cases.** Run inference with the FP16 model on the
   training set and pick the 200 frames the FP16 model is *most*
   confident about — those are the activation modes the INT8
   calibrator must preserve. Re-run calibration against those.
4. **Fall back to FP16 in production.** Bigger engine, slower inference,
   but no calibration step in the deploy path. Acceptable if the
   latency budget allows.
5. **Quantization-aware training** as a last resort. Means re-training
   from `best.pt` with QAT enabled — multi-hour cost on Orin Nano, only
   worth it if all four steps above fail to close a >2 mAP gap.

## Scaling to a Fleet

The current pipeline deploys to **one** Jetson (the team's lab device).
This section is the design exercise: what would change to deploy the
same release tag to N Jetsons in the field, and what would make it
dangerous if you did it naively.

### How `deploy.sh` would change for N devices

The current `deploy.sh` is single-target: it reads `power_profile.json`,
`nvpmodel`-switches, pulls + restarts compose, calls `healthcheck.sh`.
For N devices, three separable things change.

**Per-device state.** `/var/lib/edgeai-hw6/deployed.txt` becomes
per-device, not global. Each Jetson tracks its own current + history
locally. The fleet manager (whatever it is) maintains a separate "what
SHOULD be deployed" view across all devices.

**Concurrency strategy.** A naive `for jetson in ...; do ssh deploy; done`
processes one device at a time. Fine for 5 devices (~10 min), painful
for 50 (~1.7 h), and dangerous for 100+ (the first failure stops the
loop, leaving half the fleet on the new tag and half on the old). The
real options are: (a) parallel SSH (xargs/parallel/ansible), (b) a
fleet-management tool that maintains its own concurrency model, or (c)
GitOps-pull where each device polls and updates itself.

**Per-device targeting.** `power_profile.json` becomes per-device or
per-cohort. A warehouse fleet might want `production: 25W` on AC-powered
loading-dock cameras and `production: 7W` on battery-powered handheld
devices — same image, different runtime profile.

### Why naive `for jetson in ...; do deploy; done` is dangerous

Three failure modes that make the naive loop unsafe at scale.

**Concurrency without staging.** The naive loop deploys to all devices
serially with no canary. If the new tag has a regression that only
appears on real production data (not the integration test's bundled
sample frame), every device that's been processed before the regression
manifests is now running broken inference. Recovery: roll back N
devices. Safer pattern: deploy to a small canary cohort first (1-3
devices), wait T seconds with active monitoring, then either continue
(canary healthy) or abort (canary failing).

**No drift detection.** The loop assumes every device is at the same
starting state. In practice, devices drift: some have manual local
edits, some failed an earlier deploy and are stuck on a previous tag,
some were powered off during the last deploy and missed it entirely.
The naive loop has no way to distinguish "this device is fine, just
needs the new tag" from "this device is in an unknown state." Safer:
read each device's `deployed.txt`, treat it as the input to the deploy
decision. Devices on unexpected tags get manual investigation, not
auto-overwrite.

**Per-device tag pinning.** Some deployments need different tags on
different devices (a pilot cohort runs `v2.0.0-rc1` while the main
fleet stays on `v1.5.0`). The naive loop can't express this — it pushes
the same tag everywhere. Production needs a per-device or per-cohort
"intended tag" map maintained somewhere outside the deploy script
(a fleet config repo, an MQTT topic, a database).

### Concrete recommendation: K3s for fleets ≥10, MQTT-based for fleets <10

For Tatung's edge fleet (assume 10-50 Jetsons in research labs and
loading-dock pilot deployments), the right tool is **K3s** with
**Flux** for GitOps reconciliation. Why:

- **K3s is purpose-built for edge** — single-binary install, ships
  with sqlite by default (no external etcd), minimal resource overhead
  (~150 MB RAM per node vs ~1 GB for full K8s).
- **Per-device targeting becomes free** via node labels and selectors.
  `nodeSelector: cohort=warehouse-loading-dock` and you've targeted a
  cohort without writing custom orchestration code.
- **Drift detection is automatic.** If a device drifts (manual edit,
  failed deploy), Flux reconciles it back to the declared state on
  every cycle.
- **Rolling deploys with canary** are first-class — `Deployment` +
  `maxSurge` + `maxUnavailable` give you the canary pattern out of the
  box, no `for` loop or custom tooling.
- **`kubectl logs`, `kubectl exec`, `kubectl rollout undo` work everywhere** —
  one operator skillset across the entire fleet.

**Dominant downside: complexity per device.** K3s adds ~150 MB RAM and
a containerd/kubelet/Flux stack to each Jetson. For a 1-2 device
research team, this is overkill — the install is bigger than the
inference workload. The breakeven is around 5-10 devices: below that,
the per-device cost dominates; above that, the operator-time savings
dominate.

For fleets **<10 devices**, recommend a **custom MQTT-based fleet manager**
instead. Each Jetson subscribes to a per-cohort MQTT topic
(`fleet/cohort-warehouse/desired-tag`); the fleet operator publishes
the intended tag, each Jetson notices the change and self-deploys
(reusing this repo's `deploy.sh`). Pros: trivial implementation,
reuses Mosquitto which is already running, zero new dependencies.
Cons: no built-in canary, no drift detection (the device only
notices changes it can subscribe to), no rolling-deploy
orchestration — all of which you'd have to build yourself.

### Alternatives we considered and rejected

- **NVIDIA Fleet Command** — turn-key NVIDIA SaaS, has all the
  bells/whistles. Rejected because: (a) commercial ($), (b) ties the
  fleet to NVIDIA's hosted control plane, (c) overkill for any
  university research deployment. Real production at scale (factories,
  retail) — yes. Course/research — no.
- **Balena** — similar SaaS-ish model, more device-management focused.
  Rejected for the same reasons as Fleet Command, plus it expects a
  Balena-managed OS image which fights Jetson's NVIDIA-supplied JetPack.
- **Pure SSH + Ansible** — easier than K3s for small fleets. Why we
  prefer K3s anyway: Ansible doesn't give you drift detection or
  reconciliation; you'd be reinventing the K3s wheel in playbook YAML.
  Acceptable for 3-5 devices if you don't mind the manual canary work.
- **Per-device CI runner** — give each Jetson its own GitHub runner,
  let CI deploy directly to whichever runners label-match. Works for
  the case-II deploy pattern in this repo today; doesn't scale because
  GitHub Actions doesn't have a "deploy to all matching runners
  simultaneously" primitive without custom plumbing.

## Reflections

Per the spec's Part F5 convention (matching HW3/HW4): each team member
writes their own 150–250 word reflection addressing **all four required
items** below.

The 4 required items each member must address:

1. What specific parts of HW6 you worked on (name the Parts and
   concrete pieces — vague "I helped with everything" doesn't count).
2. What was the most challenging technical problem you solved (the bug
   or design decision that took the longest; specify symptom, what
   you tried, what fixed it).
3. What you learned that you didn't know before (one concrete
   transferable thing).
4. What you would do differently next time (one concrete change to
   process or design, not a wish-list item).

### Janlung Sung

> *(stub — replace with your own 150–250 word reflection covering all
> four required items above. Examples of HW6-specific things to
> mention so the grader can tell it's about THIS homework: the
> GHCR-auth-expiry pattern that flaked the integration job until
> swapping to a long-lived PAT, what the INT8 calibration histogram
> walk taught you about why calibration-set choice matters more than
> the calibrator algorithm, the moment the deploy.sh nvpmodel switch
> finally worked end-to-end, why writing the rollback drill BEFORE
> the deploy script would have been cheaper.)*

### Teammate B

> *(stub — replace with this team member's own 150–250 word reflection
> covering all four required items above.)*

## Submission Evidence

Repo: <https://github.com/jlsung-ttu/edgeai-hw6>
Submission tag: `submission-final`
Released tag: `v1.0.0`
GHCR image: `ghcr.io/jlsung-ttu/edgeai-hw6:v1.0.0`

### Part 0 — INT8 Calibration (10 pts)
- Engine produced via real calibration →
  `best_int8.engine` in archive (size: 4.0 MB)
- INT8 mAP drop ≤ 2 pts →
  `calibration/accuracy_baseline.json` shows fp16=0.4138 int8=0.4206 Δ=+0.0068
- Comparison table + production recommendation →
  README §"Optimization (INT8 vs FP16)" above

### Part A — Tests + Coverage + Accuracy Gates (15 pts)
- 6+ tests in test_inference → `tests/test_inference.py` (N tests)
- 4+ tests in test_mqtt → `tests/test_mqtt.py` (N tests collected)
- Coverage ≥90% gate + demo PR →
  green run: <https://github.com/jlsung-ttu/edgeai-hw6/actions/runs/...>;
  demo PR (red→green): <https://github.com/jlsung-ttu/edgeai-hw6/pull/N>
- htmlcov artifact uploaded → `evidence/htmlcov-artifact.png`
- Accuracy gate + demo PR → demo PR: <https://github.com/jlsung-ttu/edgeai-hw6/pull/M>

### Part B — Five-Stage Workflow Graph (15 pts)
- 5 jobs with correct needs graph → `.github/workflows/ci.yml`
- bandit + pip-audit both run →
  green security-scan job: <https://github.com/jlsung-ttu/edgeai-hw6/actions/runs/.../job/...>
- integration-test runs on jetson →
  `ci.yml` line N: `runs-on: [self-hosted, linux, arm64, jetson]`
- Workflow runs green end-to-end on main →
  <https://github.com/jlsung-ttu/edgeai-hw6/actions/runs/...>

### Part C — Integration Test on Jetson (15 pts)
- Test pulls per-commit image →
  `tests/integration/test_jetson_e2e.py`
  (test_image_is_per_commit_sha_tagged)
- --runtime nvidia + model-cache volume →
  same file, fixture `inference_container`
- MQTT message within 30 s →
  same file, test_inference_publishes_mqtt_within_window
- Cleanup on failure → same file, fixtures use `yield` + `try/finally`
- Job runs green on main push →
  <https://github.com/jlsung-ttu/edgeai-hw6/actions/runs/.../job/...>

### Part D — Tag-Triggered Deploy (20 pts)
- deploy.yml triggers on v*.*.* tags → `.github/workflows/deploy.yml`
- production environment with required reviewer →
  screenshot: `evidence/production-env-settings.png`
- Re-tags as v1.0.0 / v1.0 / v1 / latest →
  green deploy run: <https://github.com/jlsung-ttu/edgeai-hw6/actions/runs/...>
- deploy.sh: pull → compose up → healthcheck → rollback-on-fail →
  `deploy/deploy.sh` in archive
- healthcheck.sh: 3 consecutive successes within 60 s →
  `deploy/healthcheck.sh` in archive
- deploy.sh sets nvpmodel →
  screenshot of green deploy run: `evidence/deploy-log-nvpmodel.png`
- /healthz reports power_mode from live `nvpmodel -q` →
  `evidence/healthz-curl.png` (response body visible)

### Part E — Rollback Under 30 s (5 pts)
- rollback.sh runs end-to-end <30 s →
  recording: `evidence/rollback-demo.cast` (asciinema) or
  `evidence/rollback-demo.txt` (`script -t`)
- State file maintains current + previous tag →
  recording shows `cat /var/lib/edgeai-hw6/deployed.txt`
  before + after
- Rollback procedure (symptoms / command / recovery / comms) →
  README §"Operations" → "How to roll back" above

### Part F — Documentation & Fleet-Readiness (15 pts)
- All sections present in this README →
  §Architecture, §Optimization (INT8 vs FP16),
  §Scaling to a Fleet, §Operations, §Reflections, §Submission Evidence

### Code Quality (5 pts)
- Headers, ruff clean, secrets-free → confirmed by green lint +
  security-scan jobs above

## Repo layout

```
src/                 - Python source (inference_node, mqtt_publisher, healthcheck)
tests/               - Unit tests (mocked) + tests/integration/ (real Jetson)
calibration/         - INT8 calibration script + accuracy_baseline.json snapshot
deploy/              - docker-compose.yml + deploy.sh + healthcheck.sh + rollback.sh + power_profile.json
.github/workflows/   - ci.yml (5-stage) + deploy.yml (tag-triggered)
models/              - Test video used as inference source (carried from HW5/Lab9)
evidence/            - Submission-time screenshots + recordings (rollback-demo.txt etc.)
README.md            - All major reporting (this file): Operations, Architecture,
                       Optimization, Scaling to a Fleet, Reflections, Submission Evidence
```

## License

MIT — see file headers for per-file copyright.
