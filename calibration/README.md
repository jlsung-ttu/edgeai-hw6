# INT8 Calibration

Builds `best_int8.engine` (~7 MB) from `best.pt` using ~500 representative frames sampled from the same dataset HW5 trained on. Source-of-truth scripts live here; the actual calibration runs on the Jetson because TensorRT INT8 calibration needs a real CUDA device.

## Regenerate

```bash
cd ~/edgeai-hw6
mkdir -p calibration/calibration_data

# 1. Sample ~500 calibration frames from your HW5 dataset (which is the
#    same Lab 9 construction-safety dataset, copied into hw5/models/dataset/).
#    The directory is gitignored — re-run any time.
python3 -c "
import random, shutil
from pathlib import Path

# Try HW5 path first (preferred — same dataset that trained the refined best.pt)
src = Path.home() / 'project/coding/hw5/models/dataset/train/images'
# Fallback to Lab 9 if HW5 isn't on this machine
if not src.exists():
    src = Path.home() / 'project/coding/lab9/dataset/train/images'
if not src.exists():
    src = Path.home() / 'lab9/dataset/train/images'

assert src.exists(), f'No dataset found. Tried HW5 + Lab 9 paths.'
print(f'Using dataset at {src}')

dst = Path('calibration/calibration_data')
random.seed(42)
imgs = sorted(src.glob('*.jpg'))
sample = random.sample(imgs, k=min(500, len(imgs)))
for s in sample: shutil.copy(s, dst / s.name)
print(f'Copied {len(sample)} frames')
"

# 2. Run the calibrator
python3 calibration/calibrate_int8.py

# 3. Commit the resulting best_int8.engine (the engine is small enough for git)
git add best_int8.engine
git commit -m "Calibrate INT8 engine from latest best.pt"
git push
```

## Class list in `calibration.yaml`

`calibration.yaml` already has all 25 construction-safety classes (matches `coding_fy25/hw5/configs/labels.txt` and Lab 9's `data.yaml`). If your team trained on a different dataset, edit `nc` and `names` to match before running the calibrator.
