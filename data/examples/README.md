# Grasping Example Data

Input observations are stored under `static/` and `dynamic/`, and every saved `prompt` is in English. Static examples can be used with the parallel-jaw gripper or any supported right-handed dexterous hand. Whether a valid grasp is produced depends on the target, hand, and workspace.

| File | Saved prompt | Client `--obs-pose` |
| --- | --- | --- |
| [static/red_chips_can.pkl](static/red_chips_can.pkl) | red chips can | `world-to-camera` |
| [static/green_bottle_cap.pkl](static/green_bottle_cap.pkl) | the cap of the green bottle | `world-to-camera` |
| [static/black_headphone_headband.pkl](static/black_headphone_headband.pkl) | the headband of the black headphones | `world-to-camera` |
| [dynamic/white_box_000.pkl](dynamic/white_box_000.pkl) | white box | `world-to-camera` |
| [dynamic/white_box_001.pkl](dynamic/white_box_001.pkl) | white box | `world-to-camera` |

Replay the dynamic files in the order `white_box_000.pkl` then `white_box_001.pkl`. They are consecutive observations of the same target, not separate tasks.

Run commands from the repository root, for example:

```bash
python client/static_grasp/client_gripper.py \
  --data data/examples/static/red_chips_can.pkl \
  --obs-pose world-to-camera --output /tmp/gripper-static.pkl

python client/dynamic_grasp/client_gripper.py \
  --data data/examples/dynamic/white_box_000.pkl \
  --next-data data/examples/dynamic/white_box_001.pkl \
  --obs-pose world-to-camera --output /tmp/gripper-track.pkl
```

Start the corresponding server as described in the project [README](../../README.md). Omit `--prompt` to use the saved English prompt, or provide it explicitly to target another object.
