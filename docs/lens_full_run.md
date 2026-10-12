# AdaRoboVLG + LENS full inference

Use a Python environment with this repository's requirements and the dependencies required by LENS.

```bash
cd /path/to/adarobovlg_src
PYTHONPATH=$PWD:$PWD/third_party/d3fields/d3fields \
ADAROBOVLG_CHECKPOINTS_DIR=$PWD/checkpoints \
ADAROBOVLG_MASK_BACKEND=lens \
ADAROBOVLG_LENS_REPO=/path/to/lens_src \
ADAROBOVLG_LENS_PATH=/path/to/lens_weights \
python tools/run_lens_full.py \
  --data data/examples/static/red_chips_can.pkl \
  --prompt "red chips can"
```

Verified output on 2026-10-10:

```text
prompt: red chips can
views: 4 points: 37532
mask pixels: [6543, 6892, 4372, 1407]
projected object points: 4734
grasp candidates: 5
best score: 0.9183110594749451
best translation: [0.7191216265795042, -0.15567348595112648, 0.09176479391860176]
```

The run covers LENS text-conditioned mask inference, multi-view mask fusion, 3D object-point projection, and ContactGraspNet grasp proposal generation.
