# LENS demo and evaluation notes

## Visual demos

Generated with `tools/run_lens_mask.py`:

- `lens_demo_red`: `red chips can`
- `lens_demo_green`: `the cap of the green bottle`
- `lens_demo_headphones`: `the headband of the black headphones`

Each directory contains `view_00.png` ... `view_03.png`, `masks.npy`, and `boxes.txt`.

## Static mask comparison

The original SAM3 text-only backend and LENS were run on the same three recorded observations. These are diagnostic comparisons because the examples do not contain pixel-level ground-truth masks. Mean per-view IoU between the two model outputs was:

| sample | LENS foreground pixels (4 views) | SAM3 foreground pixels (4 views) | mean output IoU |
|---|---:|---:|---:|
| red chips can | 6113, 6640, 5476, 4054 | 0, 0, 0, 0 | 0.000 |
| green bottle cap | 11647, 7760, 1334, 9140 | 0, 670, 321, 472 | 0.092 |
| black headphone headband | 8687, 6063, 4908, 8559 | 0, 7573, 2640, 6950 | 0.425 |

The output IoU is not an accuracy metric; it only measures agreement between the two implementations.

## End-to-end grasp checks

LENS produced five grasp candidates on each of the following recorded observations:

- `red_chips_can.pkl`: 4,734 projected object points, best score `0.9183`
- `green_bottle_cap.pkl`: 3,838 projected object points, best score `0.9809`
- `black_headphone_headband.pkl`: 7,598 projected object points, best score `0.9616`

## Playground smoke evaluation

The official Playground runner was started with one object and one trial. After restoring the repository's missing Git LFS robosuite assets and installing the simulator dependencies, one complete 300-step episode ran through the LENS-backed ZMQ server. The sampled target was `corn`; result: `fail`, steps `300`, success rate `0/1 = 0%`.

The standard LIBERO task runner reaches simulator initialization, but the released checkout currently has incompatible benchmark assets/goal-state conventions in this environment. It therefore cannot provide a defensible paper-level success-rate comparison without aligning the exact Playground revision and asset bundle used by the paper.
