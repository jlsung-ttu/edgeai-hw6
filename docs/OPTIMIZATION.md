# Optimization

Part 0 deliverable. Reports the FP16 vs INT8 comparison from the
`calibration/calibrate_int8.py` run, plus the production recommendation.

## FP16 vs INT8 (Part 0 required)

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

## Production recommendation

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

## What we considered and didn't ship

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

## If the INT8 mAP drop were unacceptable (Part 0.3 prompt)

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
