"""End-to-end AdaRoboVLG inference with the LENS mask backend."""
import argparse
import pickle

import numpy as np

from model.adarobovlg import AdaRoboVLG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--prompt", default=None)
    args = p.parse_args()
    with open(args.data, "rb") as f:
        sample = pickle.load(f)
    prompt = args.prompt or sample["prompt"]
    obs = sample["obs"]
    points = np.asarray(sample["points"], dtype=np.float32)
    colors = np.asarray(sample["colors"], dtype=np.float32)
    model = AdaRoboVLG(mask_backend="lens", track=False, num_cam=obs["color"].shape[0])
    grasps = model.forward_gripper(
        prompt, points, colors, obs, flip_z=True, top_k=5, collision_thresh=0.05
    )
    print("prompt:", prompt)
    print("views:", obs["color"].shape[0], "points:", points.shape[0])
    print("mask pixels:", model.masks.sum(axis=(1, 2)).tolist())
    print("projected object points:", len(model.obj_pts))
    print("grasp candidates:", 0 if grasps is None else len(grasps))
    if grasps is not None and len(grasps):
        print("best score:", float(grasps.scores.max()))
        print("best translation:", grasps.translations[0].tolist())


if __name__ == "__main__":
    main()
