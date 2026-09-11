"""Filter multi-finger grasp candidates using hand-object collision checks."""

import torch
import copy
import time
import numpy as np
import meshcat
import meshcat.geometry as g

from typing import Optional, Union
from cprint import cprint
import trimesh
from dexgraspapi.multi_finger_grasp_base import MultiFingerGraspGroupBase
from graspnetAPI.grasp import GraspGroup
from robots.hands.hand_base import create_hand_model
from robots.hands.hand_collision_base import create_hand_collision
from utils.rotation_torch import matrix_to_euler


class MultiFingerCollisionDetector:
    def __init__(
        self,
        hand_name: str,
        device: Union[str, torch.device] = "cpu",
        max_points: Optional[int] = None,
    ):
        """Initialize the multi-finger collision detector.

        Only hand model and collision model are created here.
        Point cloud and grasp groups are set via `set_scene` later.

        Args:
            hand_name: Name of the robotic hand (e.g., "AllegroHand").
            device: Device for computation ("cpu" or "cuda").
            max_points: Optional maximum number of points to use from the scene
                point cloud. If the input point cloud has more points, random
                downsampling is applied. If None, use all points.
        """
        self.device = torch.device(device)
        self.max_points = max_points

        self.tf_gg = None
        self.mf_gg = None
        self.pc_points = None
        self.ori_pc_points = None
        self.pc_colors = None

        # Hand model (visualization)
        self.hm = create_hand_model(
            hand_name=hand_name,
            use_cache=True,
        )

        # Hand collision model
        self.hc = create_hand_collision(
            hand_name=hand_name + "Collision",
            device=self.device,
            margin=0.0,
        )

    def set_scene(
        self,
        pc_points: Union[np.ndarray, torch.Tensor],
        tf_gg: GraspGroup,
        mf_gg: MultiFingerGraspGroupBase,
        pc_colors: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """Set or update scene point cloud and grasp groups.

        This can be called multiple times before `detect` / `vis_detect`.

        Args:
            pc_points: Scene point cloud as (N, 3) or (B, N, 3) numpy array
                or torch tensor (float32).
            tf_gg: GraspGroup of two-finger grasps corresponding to the
                multi-finger grasps.
            mf_gg: MultiFingerGraspGroupBase of multi-finger grasps to check.
            pc_colors: Optional colors for the point cloud as (N, 3) or
                (B, N, 3) numpy array or torch tensor (float32).
        """
        self.tf_gg = tf_gg
        self.mf_gg = mf_gg

        # Process point cloud
        if isinstance(pc_points, np.ndarray):
            pc = torch.from_numpy(pc_points.astype(np.float32))
        else:
            pc = pc_points.to(dtype=torch.float32)

        if pc.ndim == 2:
            pc = pc.unsqueeze(0)  # (1, N, 3)

        assert (
            pc.ndim == 3 and pc.shape[2] == 3
        ), f"pc_points must have shape (B, N, 3) or (N, 3), got {pc.shape}"

        pc = pc.to(self.device)  # (B, N, 3)
        self.ori_pc_points = pc.clone()

        # Optional global downsampling
        if (self.max_points is not None) and (pc.shape[1] > self.max_points):
            N = pc.shape[1]
            idx = torch.randperm(N, device=self.device)[: self.max_points]
            pc = pc[:, idx, :]  # (B, max_points, 3)

        self.pc_points = pc

        # Process optional colors
        if pc_colors is not None:
            if isinstance(pc_colors, np.ndarray):
                colors = torch.from_numpy(pc_colors.astype(np.float32))
            else:
                colors = pc_colors.to(dtype=torch.float32)

            if colors.ndim == 2:
                colors = colors.unsqueeze(0)  # (1, N, 3)

            assert (
                colors.ndim == 3 and colors.shape[2] == 3
            ), f"pc_colors must have shape (B, N, 3) or (N, 3), got {colors.shape}"
            assert (
                colors.shape[0] == self.ori_pc_points.shape[0]
                and colors.shape[1] == self.ori_pc_points.shape[1]
            ), "pc_colors must have the same batch size and number of points as pc_points."

            self.pc_colors = colors.to(self.device)  # (B, N, 3)

    def detect(
        self,
        approach_dist: float = 0.04,
        collision_thresh: int = 10,
        step: float = 0.005,
        print_result: bool = False,
    ):
        """Check collision for each multi-finger grasp along its approach path.

        Args:
            approach_dist: Distance (in meters) for the hand to move backwards
                from the final grasp pose along the approach direction.
            collision_thresh: Threshold on the number of colliding spheres
                (accumulated over all links) to declare collision at one time step.
            step: Step size (in meters) along the approach direction.
            print_result: If True, print per-grasp, per-time collision summary.

        Returns:
            dict with keys:
                "is_colliding_path":    (B,) bool tensor (after applying threshold)
                "first_collision_step": (B,) int tensor (-1 if never colliding)
                "raw_time_series_stats": full stats returned by HandCollisionBase
        """
        if self.pc_points is None or self.tf_gg is None or self.mf_gg is None:
            raise RuntimeError(
                "Point cloud and grasp groups are not set. "
                "Call `set_scene(pc_points, tf_gg, mf_gg, pc_colors)` first."
            )

        device = self.device

        # Number of grasps
        B = len(self.mf_gg)
        assert B == len(self.tf_gg), "tf_gg and mf_gg must have the same length."

        # Joint angles from multi-finger grasp group: (B, D)
        joints_per_grasp = torch.as_tensor(self.mf_gg.angles, dtype=torch.float32, device=device)
        B_, D = joints_per_grasp.shape
        assert B_ == B, f"Inconsistent batch size: B={B}, angles.shape={joints_per_grasp.shape}"

        # Determine number of steps along the approach path
        max_back = float(approach_dist)
        if max_back <= 0.0:
            raise ValueError(f"approach_dist must be positive, got {approach_dist}")
        else:
            num_steps = int(max_back / step) + 1

        # Prepare time-series tensors
        joints_bt = torch.zeros(B, num_steps, D, dtype=torch.float32, device=device)
        trans_bt = torch.zeros(B, num_steps, 3, dtype=torch.float32, device=device)
        rot_bt = torch.zeros(B, num_steps, 3, dtype=torch.float32, device=device)

        # Build time series for each grasp
        for i in range(B):
            # Base rotation from two-finger grasp
            R0_tf = torch.as_tensor(
                self.tf_gg.rotation_matrices[i], dtype=torch.float32, device=device
            ).view(3, 3)

            # Base pose from multi-finger grasp
            t0_mf = torch.as_tensor(
                self.mf_gg.translations[i], dtype=torch.float32, device=device
            )  # (3,)
            R0_mf = torch.as_tensor(
                self.mf_gg.rotation_matrices[i], dtype=torch.float32, device=device
            ).view(
                3, 3
            )  # (3, 3)

            # Approach direction (x-axis of the gripper in world frame)
            d = R0_tf[:, 0]
            d = d / (d.norm() + 1e-8)

            # Depth offset for this grasp
            dpi = torch.as_tensor(self.mf_gg.depths[i], dtype=torch.float32, device=device)

            # Distance samples along the approach path:
            # s goes from (max_back - dpi) to -dpi
            s = torch.linspace(max_back - dpi, -dpi, num_steps, device=device)  # (T,)

            # Time-series base translations:
            trans_series = t0_mf.unsqueeze(0) - s.unsqueeze(1) * d.unsqueeze(0)  # (T, 3)

            # Fill tensors
            joints_bt[i] = joints_per_grasp[i].unsqueeze(0).expand(num_steps, -1)
            trans_bt[i] = trans_series

            # Base rotation constant along the path
            euler = matrix_to_euler(R0_mf)  # (3,)
            rot_bt[i] = euler.unsqueeze(0).expand(num_steps, -1)

        # Call time-series collision on the hand collision model
        detect_batch_size = 32
        stats = {}
        for start in range(0, B, detect_batch_size):
            end = min(start + detect_batch_size, B)
            stats_batch = self.hc.compute_collision_with_pointcloud_time_series(
                point_cloud=copy.deepcopy(self.pc_points),  # (B, N, 3)
                joints=joints_bt[start:end],  # (B_batch, T, D)
                translations=trans_bt[start:end],  # (B_batch, T, 3)
                rotations=rot_bt[start:end],  # (B_batch, T, 3)
                print_result=False,
                need_per_link=False,
            )
            # Accumulate stats
            for key in stats_batch:
                if key not in stats:
                    stats[key] = []
                stats[key].append(stats_batch[key])
        for key in stats:
            if isinstance(stats[key][0], torch.Tensor):
                stats[key] = torch.cat(stats[key], dim=0)  # (B, T, ...)
            elif isinstance(stats[key][0], list):
                stats[key] = sum(stats[key], [])  # concatenate lists
            # stats = self.hc.compute_collision_with_pointcloud_time_series(
            #     point_cloud=copy.deepcopy(self.pc_points),  # (B, N, 3)
            #     joints=joints_bt,       # (B, T, D)
            #     translations=trans_bt,  # (B, T, 3)
            #     rotations=rot_bt,       # (B, T, 3)
            #     print_result=False,
            #     need_per_link=False,
            # )

        # stats["is_colliding"]: (B, T) bool (any sphere penetration)
        # stats["num_colliding_points"]: (B, T) int (colliding scene points)
        is_coll_ts = stats["is_colliding"]  # (B, T)
        num_pts_ts = stats["num_colliding_points"]  # (B, T)

        B_ts, T_ts = is_coll_ts.shape
        assert B_ts == B and T_ts == num_steps

        # Apply collision_thresh based on total number of colliding scene points at each (b, t)
        spheres_count_ts = num_pts_ts.to(torch.int32)  # (B, T) int32
        is_coll_thresh = spheres_count_ts >= collision_thresh  # (B, T) bool

        # For each grasp, collision along the path if any time step exceeds threshold
        is_coll_path = torch.any(is_coll_thresh, dim=1)  # (B,)

        # First time step where threshold-based collision happens
        first_collision_step = torch.full((B,), -1, dtype=torch.int64)
        for b in range(B):
            idx = torch.nonzero(is_coll_thresh[b], as_tuple=False)
            if idx.numel() > 0:
                first_collision_step[b] = int(idx[0].item())

        if print_result:
            for b in range(B):
                print(f"\n===== Grasp {b} =====")
                print(f"Path collision (threshold-based): {bool(is_coll_path[b].item())}")
                if first_collision_step[b] >= 0:
                    print(f"First collision step index: {first_collision_step[b].item()}")
                else:
                    print("No collision along the approach path.")
                # Optional: print per-step colliding point counts for debugging
                print("Colliding scene point counts along time steps:")
                print(spheres_count_ts[b].cpu().tolist())

        return {
            "is_colliding_path": is_coll_path,  # (B,) after threshold
            "first_collision_step": first_collision_step,  # (B,)
            "raw_time_series_stats": stats,  # full [B, T] collision stats
            "spheres_count_time_series": spheres_count_ts,  # (B, T) int
            "joints_time_series": joints_bt,  # (B, T, D)
            "translations_time_series": trans_bt,  # (B, T, 3)
            "rotations_time_series": rot_bt,  # (B, T, 3)
        }

    def vis_detect(
        self,
        approach_dist: float = 0.04,
        collision_thresh: int = 10,
        step: float = 0.005,
        fps: float = 20.0,
    ):
        """Visualize the collision detection process.

        Args:
            approach_dist: Distance (in meters) for the hand to move backwards
                from the final grasp pose along the approach direction.
            collision_thresh: Threshold on the number of colliding spheres
                (accumulated over all links) to declare collision at one time step.
            step: Step size (in meters) along the approach direction.
        """

        # 1) Run detection to get time-series data
        res = self.detect(
            approach_dist=approach_dist,
            collision_thresh=collision_thresh,
            step=step,
            print_result=False,
        )

        is_coll_path = res["is_colliding_path"]  # (B,)
        first_collision_step = res["first_collision_step"]  # (B,)
        joints_bt = res["joints_time_series"]  # (B, T, D)
        trans_bt = res["translations_time_series"]  # (B, T, 3)
        rot_bt = res["rotations_time_series"]  # (B, T, 3)

        # 2) Initialize meshcat visualizer
        vis = meshcat.Visualizer()
        vis["/Background"].set_property("top_color", [0.7, 0.7, 0.7])
        vis["/Background"].set_property("bottom_color", [0.7, 0.7, 0.7])
        vis["/Grid"].set_property("visible", False)
        vis["/Axes"].set_property("visible", False)

        # 3) Visualize scene point cloud
        points = self.ori_pc_points.squeeze(0).cpu().numpy()  # (N, 3)
        colors = (
            self.pc_colors.squeeze(0).cpu().numpy()
            if self.pc_colors is not None
            else np.ones_like(points) * 0.6
        )
        if len(points) > 0:
            point_cloud = g.Points(
                g.PointsGeometry(
                    position=points.T,  # (3, N)
                    color=colors.T,  # (3, N)
                ),
                g.PointsMaterial(size=0.002),
            )
            vis["point_cloud"].set_object(point_cloud)

        # 4) Visualize each grasp in collision detection process
        B, T, _ = joints_bt.shape
        assert B == len(self.mf_gg) and B == len(self.tf_gg)
        assert T == trans_bt.shape[1] and T == rot_bt.shape[1]

        for b in range(B):
            cprint.info(f"Visualizing grasp {b}/{B-1}...")

            # Whether this grasp has any collision along its path
            collide = bool(is_coll_path[b].item())
            # First time step index where collision happens (-1 means never)
            first_step = int(first_collision_step[b].item())

            for t in range(T):
                self.hm.update_cfg(
                    joints=joints_bt[b, t].cpu().numpy(),
                    translation=trans_bt[b, t].cpu().numpy(),
                    rotation=rot_bt[b, t].cpu().numpy(),
                )

                hm_tm = self.hm.trimesh_mesh  # hand trimesh object

                if isinstance(hm_tm.visual, trimesh.visual.texture.TextureVisuals):
                    base_colors = np.asarray(hm_tm.visual.to_color().vertex_colors)[:, :3] / 255.0
                elif isinstance(hm_tm.visual, trimesh.visual.color.ColorVisuals):
                    base_colors = np.asarray(hm_tm.visual.vertex_colors)[:, :3] / 255.0
                else:
                    base_colors = np.ones((len(hm_tm.vertices), 3)) * 0.8

                # Decide if current frame is in collision phase
                in_collision_phase = collide and (first_step >= 0) and (t >= first_step)

                if in_collision_phase:
                    vert_colors = np.zeros_like(base_colors)
                    vert_colors[:, 0] = 1.0
                else:
                    vert_colors = base_colors

                vis["hand"].set_object(
                    g.TriangularMeshGeometry(
                        hm_tm.vertices,
                        hm_tm.faces,
                        vert_colors,
                    ),
                    g.MeshBasicMaterial(vertexColors=True),
                )

                time.sleep(1.0 / fps)
