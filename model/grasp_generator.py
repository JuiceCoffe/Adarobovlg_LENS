"""ContactGraspNet adapter with world/network coordinate conversion."""

from utils.checkpoints import checkpoint_path

import numpy as np
from pathlib import Path
from contact_graspnet.cgnet_infer import contact_grasp_init, contact_grasp_infer, query_best_points


class GraspGenerator:
    def __init__(self):
        """Load bundled ContactGraspNet configuration and weights, rejecting missing assets."""
        repo_root = Path(__file__).resolve().parents[1]
        cgnet_root = repo_root / "third_party" / "contact_graspnet" / "contact_graspnet"
        cfg_path = cgnet_root / "cfgs" / "GC6D" / "Contact_GraspNet.yaml"
        ckpt_path = checkpoint_path("contact_graspnet/gc6d.pth", "ADAROBOVLG_GRASP_CHECKPOINT")

        missing_paths = [str(path) for path in (cfg_path, ckpt_path) if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(
                "ContactGraspNet assets are missing: " + ", ".join(missing_paths)
            )

        self.cgnet_model = contact_grasp_init(str(cfg_path), str(ckpt_path))

    def get_best_points(self, pts, top_k=5, flip_z=False, debug=False):
        """Return top_k world-frame contact points after undoing network centering and optional X rotation."""
        pts = pts.copy()
        scene_center = pts.mean(axis=0)
        pts -= scene_center
        if flip_z:
            # rotate point cloud respect to x axis
            rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
            pts = pts @ rot.T
        best_points = query_best_points(self.cgnet_model, pts, top_k=top_k, debug=debug)  # Nx3
        if flip_z:
            best_points = best_points @ rot  # Nx3
        best_points += scene_center
        best_points = best_points.float()
        return best_points

    def generate_grasps(
        self, pts, color, valid_pts=None, top_k=5, collision_thresh=0.01, debug=False, flip_z=False
    ):
        """Generate world-frame GraspGroup candidates from metric points and RGB colors.

        valid_pts optionally restricts contacts; flip_z is undone on output.
        collision_thresh is passed to the network inference adapter."""
        pts = pts.copy()
        valid_pts = valid_pts.copy() if valid_pts is not None else None
        scene_center = pts.mean(axis=0)
        pts -= scene_center
        valid_pts = valid_pts - scene_center if valid_pts is not None else None

        if flip_z:
            # rotate point cloud respect to x axis
            rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
            pts = pts @ rot.T
            if valid_pts is not None:
                valid_pts = valid_pts @ rot.T
        grasps = contact_grasp_infer(
            self.cgnet_model,
            pts,
            color=color,
            valid_pts=valid_pts,
            top_k=top_k,
            collision_thresh=collision_thresh,
            debug=debug,
        )
        if flip_z:
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = rot.T
            grasps.transform(T)
        grasps.translations += scene_center
        return grasps
