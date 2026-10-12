"""Run LENS masks on an AdaRoboVLG example and save per-view overlays."""
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location("lens_mask_generator", Path(__file__).resolve().parents[1] / "model" / "lens_mask_generator.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
LensMaskGenerator = _module.LensMaskGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="lens_outputs")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    with open(args.data, "rb") as f:
        sample = pickle.load(f)
    images = np.asarray(sample["images"])
    generator = LensMaskGenerator(checkpoint=args.checkpoint)
    masks, boxes = generator.generate_mask(images, args.prompt)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    np.save(output / "masks.npy", masks.astype(np.uint8))
    with open(output / "boxes.txt", "w", encoding="utf-8") as f:
        f.write(f"prompt: {args.prompt}\n")
        for i, box in enumerate(boxes):
            f.write(f"view {i}: {box}\n")
    for i, (image, mask, box) in enumerate(zip(images, masks, boxes)):
        vis = image.copy()
        vis[mask] = (0.55 * vis[mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        if len(box) == 4:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"LENS prompt: {args.prompt}"
        cv2.rectangle(vis, (0, 0), (min(vis.shape[1], 12 + 11 * len(label)), 32), (0, 0, 0), -1)
        cv2.putText(vis, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        original = image.copy()
        cv2.putText(original, f"input view {i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        comparison = np.concatenate([original, vis], axis=1)
        cv2.imwrite(str(output / f"view_{i:02d}.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"saved {len(masks)} masks to {output}")


if __name__ == "__main__":
    main()
