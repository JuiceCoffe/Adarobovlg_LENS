# LENS mask backend

AdaRoboVLG can use the released HUST LENS checkpoint for direct natural-language to mask inference. Set these variables before starting an AdaRoboVLG service:

```bash
export ADAROBOVLG_MASK_BACKEND=lens
export ADAROBOVLG_LENS_REPO=/path/to/lens_src
export ADAROBOVLG_LENS_PATH=/path/to/lens_weights
```

The adapter loads the Qwen2.5-VL 3B LENS model once, generates a referring-expression box and mask for each camera image, and returns the same `(N,H,W)` masks expected by the 3D projection code. `ADAROBOVLG_LENS_PATH` can point to any compatible LENS checkpoint, including the RefCOCO, ReasonSeg, or ReasonSeg-CoT variants.

To reproduce the server-side smoke test used for this integration:

```bash
cd /path/to/adarobovlg_src
PYTHONPATH=$PWD \
  python tools/run_lens_mask.py \
  --data data/examples/static/red_chips_can.pkl \
  --prompt 'red chips can' --output lens_outputs
```

The command writes `masks.npy`, one overlay PNG per view, and `boxes.txt`. The four-view sample produced masks with 6,056, 6,494, 5,204, and 2,629 foreground pixels respectively.
