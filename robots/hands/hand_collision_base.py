import re
import torch
import logging
import numpy as np
import open3d as o3d
import matplotlib.cm as cm

from utils.rotation_torch import euler_to_matrix
from utils.torch_urdf import TorchURDF
from abc import ABC
from geometrout.primitive import Sphere
from typing import Dict, List, Optional, Union
from utils.registry import Registry
from utils.mesh import extract_collision_spheres_from_urdf
from utils.transform import transform_pointcloud_torch


class HandCollisionBase(ABC):
    def __init__(
        self,
        urdf_path: str,
        device: Union[str, torch.device] = "cpu",
        margin: float = 0.0,
        num_points: int = 32768,
    ):
        """Initializes the collosion model for the robot hand, loading
        its URDF model, configuring collision spheres with optional margin.

        Args:
            device [torch.device or str]:
                The device on which tensors and models will be loaded (e.g., 'cpu' or 'cuda').
            margin [float, optional]:
                Additional margin to add to the radius of each collision sphere. Default is 0.0.

        Return:
            None. Initializes class attributes including the hand model.
        """
        logging.getLogger("trimesh").setLevel("ERROR")

        self.device = device
        self.num_points = num_points

        self.urdf = TorchURDF.load(urdf_path, lazy_load_meshes=True, device=device)
        self.link_names = [l.name for l in self.urdf.links if len(l.visuals)]
        self.joint_names = [j.name for j in self.urdf.actuated_joints]
        self.dof = len(self.joint_names)
        self.frame_status = None

        # Initialize zero tensors for translations, rotations, and joints
        self.zero_translations = torch.zeros(3, device=device, dtype=torch.float32).unsqueeze(0)
        self.zero_rotations = torch.zeros(3, device=device, dtype=torch.float32).unsqueeze(
            0
        )  # Euler angles
        self.zero_joints = torch.zeros(self.dof, device=device, dtype=torch.float32).unsqueeze(0)

        # Set current translations, rotations, and joints to zero tensors
        self.translations = self.zero_translations
        self.rotations = self.zero_rotations
        self.joints = self.zero_joints

        self._init_spheres_(
            urdf_path,
            device=device,
            margin=margin,
            num_points=num_points,
        )

    def _init_spheres_(
        self,
        urdf_path: str,
        device: Union[str, torch.device] = "cpu",
        margin: float = 0.0,
        num_points: int = 10000,
    ):
        """
        Initializes collision spheres for robot hand and precomputes surface sample
        points for each hand link.

        Args:
            device [torch.device or str]:
                The device on which tensors and models will be loaded (e.g., 'cpu' or 'cuda').
            margin [float, optional]:
                Additional margin to add to the radius of each collision sphere. Default is 0.0.
            total_points [int, optional]:
                Total number of surface points to sample across all links. Default is 10000

        Return:
            None. Initializes class attributes including the collision spheres, and sampled surface points.
        """
        spheres_list = extract_collision_spheres_from_urdf(urdf_path)

        self.spheres = []
        for radius, point_set in spheres_list:
            sphere_centers = {
                k: torch.as_tensor(v).unsqueeze(0).to(device) for k, v in point_set.items()
            }
            if not len(sphere_centers):
                continue
            self.spheres.append(
                (
                    radius + margin,
                    sphere_centers,
                )
            )

        all_spheres = {}
        for radius, point_set in spheres_list:
            for link_name, centers in point_set.items():
                for c in centers:
                    all_spheres[link_name] = all_spheres.get(link_name, []) + [
                        Sphere(np.asarray(c), radius + margin)
                    ]

        surface_scalar_sum = sum([sum([s.radius**2 for s in v]) for v in all_spheres.values()])
        surface_scalar = num_points / surface_scalar_sum

        self.link_points = {}
        for link_name, spheres in all_spheres.items():
            self.link_points[link_name] = torch.as_tensor(
                np.concatenate(
                    [s.sample_surface(int(surface_scalar * s.radius**2)) for s in spheres],
                    axis=0,
                ),
                device=device,
            ).unsqueeze(0)

        # Build global sphere-to-link index mapping
        # self.sphere_link_indices: List[Tuple[str, int]]
        #   global_idx -> (link_name, local_idx_in_that_link)
        self.sphere_link_indices = []
        for link_name, spheres in all_spheres.items():
            for local_idx, _sphere in enumerate(spheres):
                self.sphere_link_indices.append((link_name, local_idx))
        self.num_spheres = len(self.sphere_link_indices)

    @property
    def batch_size(self) -> int:
        """Returns the current batch size based on the joints tensor.

        Returns:
            int: The batch size, determined by the first dimension of the joints tensor.
        """
        assert (
            self.joints.shape[0] == self.translations.shape[0] == self.rotations.shape[0]
        ), "Inconsistent batch sizes among joints, translations, and rotations."
        return self.joints.shape[0]

    @batch_size.setter
    def batch_size(self, batch_size: int):
        """Sets the batch size for the joints, translations, and rotations tensors.

        Args:
            batch_size (int): The desired batch size to set.
        """
        current_batch_size = self.batch_size

        if current_batch_size == batch_size:
            return

        if current_batch_size == 1:
            # expand with copy
            self.joints = self.joints.expand(batch_size, -1).clone()
            self.translations = self.translations.expand(batch_size, -1).clone()
            self.rotations = self.rotations.expand(batch_size, -1).clone()
        elif current_batch_size > 1 and batch_size < current_batch_size:
            # truncate safely
            self.joints = self.joints[:batch_size].clone()
            self.translations = self.translations[:batch_size].clone()
            self.rotations = self.rotations[:batch_size].clone()
        else:
            raise ValueError(
                f"Cannot expand from batch {current_batch_size} → {batch_size} "
                "when current batch_size > 1. Provide a template or reset first."
            )

    @property
    def base_matrix(self) -> torch.Tensor:
        """Computes the base transformation matrix as a (B, 4, 4) tensor."""
        B = self.translations.shape[0]
        R = euler_to_matrix(self.rotations)
        T = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)
        T[:, :3, :3] = R
        T[:, :3, 3] = self.translations
        return T

    def as_tensor(self, x):
        """Convert values to this geometry model's configured tensor device and dtype."""
        return torch.as_tensor(x, device=self.device, dtype=torch.float32)

    def update_status(
        self,
        joints: Optional[Union[torch.Tensor, np.ndarray]] = None,
        translations: Optional[Union[torch.Tensor, np.ndarray]] = None,
        rotations: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ):
        """Updates the status of the frame based on the given configuration (cfg).
        Supports both single input (1D) and batched input (2D).

        Args:
            joints [Tensor or np.ndarray, optional]:
                Joint configuration in Euler representation of shape (N,) or (B, N), where N is the number of joints.
            translations [Tensor or np.ndarray, optional]:
                Base translations of shape (3,) or (B, 3).
            rotations [Tensor or np.ndarray, optional]:
                Base rotations in Euler angles of shape (3,) or (B, 3).

        Return:
            frame_status [Dict[str, pk.Transform3d]]:
                A dictionary mapping link names to their corresponding Transform3d objects after applying forward kinematics.
            base_matrix [Tensor]:
                A tensor of shape (B, 4, 4) representing the base transformation matrices.
        """
        # Reset to zero configuration
        self.joints = self.zero_joints
        self.translations = self.zero_translations
        self.rotations = self.zero_rotations

        # Convert inputs to tensors and ensure consistent batch dimensions
        if joints is None and translations is None and rotations is None:
            self.frame_status = self.urdf.link_fk_batch(self.joints, use_names=True)
        elif joints is not None and translations is None and rotations is None:
            joints = self.as_tensor(joints)
            if joints.ndim == 1:
                joints = joints.unsqueeze(0)
            elif joints.ndim != 2:
                raise ValueError("joints tensor must be either 1D or 2D.")

            B = joints.shape[0]

            self.joints = joints
            self.translations = self.translations.expand(B, -1).clone()
            self.rotations = self.rotations.expand(B, -1).clone()

            self.frame_status = self.urdf.link_fk_batch(self.joints, use_names=True)
        elif joints is not None and translations is not None and rotations is not None:
            joints = self.as_tensor(joints)
            translations = self.as_tensor(translations)
            rotations = self.as_tensor(rotations)

            assert (
                joints.ndim == translations.ndim == rotations.ndim
            ), "joints, translations, and rotations must have the same number of dimensions."

            if joints.ndim == 1:
                self.joints = joints.unsqueeze(0)
                self.translations = translations.unsqueeze(0)
                self.rotations = rotations.unsqueeze(0)
            elif joints.ndim == 2:
                assert (
                    joints.shape[0] == translations.shape[0] == rotations.shape[0]
                ), "joints, translations, and rotations must have the same batch size."
                self.joints = joints
                self.translations = translations
                self.rotations = rotations
            else:
                raise ValueError("joints tensor must be either 1D or 2D.")

            self.frame_status = self.urdf.link_fk_batch(self.joints, use_names=True)
        else:
            raise ValueError(
                "Either provide all of joints, translations, and rotations, or just joints, or none."
            )

    def get_sampled_pc(
        self,
        joints: Optional[Union[torch.Tensor, np.ndarray]] = None,
        translations: Optional[Union[torch.Tensor, np.ndarray]] = None,
        rotations: Optional[Union[torch.Tensor, np.ndarray]] = None,
        num_points: int = None,
    ) -> torch.Tensor:
        """Generates a point cloud by sampling points from the robot's links given a configuration.
        The function first ensures the configuration tensor has the correct shape, concatenates
        default prismatic values, computes forward kinematics for each link, transforms the
        predefined link points, and finally samples 'num_points' points from the resulting point cloud.

        Args:
            joints [Tensor or np.ndarray, optional]:
                Joint configuration in Euler representation of shape (N,) or (B, N), where N is the number of joints.
            translations [Tensor or np.ndarray, optional]:
                Base translations of shape (3,) or (B, 3).
            rotations [Tensor or np.ndarray, optional]:
                Base rotations in Euler angles of shape (3,) or (B, 3).
            num_points [int, optional]:
                Number of points to sample from the point cloud. If None, defaults to self.num_points attribute.
        Return:
            Tensor:
                A tensor of shape (B, num_points, 3) representing the sampled point cloud.
        """
        assert (
            num_points <= self.num_points
        ), f"Requested number of points {num_points} exceeds available points {self.num_points}."

        self.update_status(joints, translations, rotations)

        pointcloud = []
        for l_name, points in self.link_points.items():
            pc = transform_pointcloud_torch(
                points.float().expand(self.frame_status[l_name].shape[0], -1, -1),
                torch.matmul(self.base_matrix, self.frame_status[l_name]),
                in_place=False,
            )
            pointcloud.append(pc)
        pc = torch.cat(pointcloud, dim=1)
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]

    def compute_transformed_spheres(
        self,
        joints: Optional[Union[torch.Tensor, np.ndarray]] = None,
        translations: Optional[Union[torch.Tensor, np.ndarray]] = None,
        rotations: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ):
        """Computes the transformed positions of predefined spheres attached to hand
        links for a given configuration.
        The function processes each sphere group, applies the corresponding forward
        kinematics transformation, and returns the transformed points.

        Args:
            joints [Tensor or np.ndarray, optional]:
                Joint configuration in Euler representation of shape (N,) or (B, N), where N is the number of joints.
            translations [Tensor or np.ndarray, optional]:
                Base translations of shape (3,) or (B, 3).
            rotations [Tensor or np.ndarray, optional]:
                Base rotations in Euler angles of shape (3,) or (B, 3).
        Return:
            List[Tuple[float, Tensor]]:
                A list where each element is a tuple containing the sphere radius and a tensor of shape (B, M, 3)
                representing the transformed sphere centers for that radius, with M being the total number of spheres of that radius.
        """
        self.update_status(joints, translations, rotations)

        spheres = []
        for radius, centers in self.spheres:
            s_centers = []
            for l_name in centers:
                pc = transform_pointcloud_torch(
                    centers[l_name].float().expand(self.frame_status[l_name].shape[0], -1, -1),
                    torch.matmul(self.base_matrix, self.frame_status[l_name]),
                    in_place=False,
                )
                s_centers.append(pc)
            spheres.append((radius, torch.cat(s_centers, dim=1)))
        return spheres

    def compute_collision_with_pointcloud(
        self,
        point_cloud: Union[np.ndarray, torch.Tensor],
        joints: Optional[Union[torch.Tensor, np.ndarray]] = None,
        translations: Optional[Union[torch.Tensor, np.ndarray]] = None,
        rotations: Optional[Union[torch.Tensor, np.ndarray]] = None,
        print_result: bool = False,
        need_per_link: bool = False,
    ):
        """Given a scene point cloud and hand configuration, compute for each batch:
          1) whether any collision happens,
          2) maximal penetration depth,
          3) minimal penetration depth (over colliding spheres),
          4) mean penetration depth (over colliding spheres),
          5) per-link, per-sphere penetration depths.

        Args:
            point_cloud: Scene point cloud, shape (N, 3) or (B, N, 3),
                numpy array or torch tensor.
            joints: Joint values, shape (D,) or (B, D).
            translations: Base translations, shape (3,) or (B, 3).
            rotations: Base Euler rotations, shape (3,) or (B, 3).
            print_result: If True, print all statistics to stdout.
            need_per_link: If False, skip building per-link/sphere lists
                to save Python-side overhead.

        Returns:
            dict with keys:
                "is_colliding":          (B,) bool tensor,
                "max_penetration":       (B,) float tensor,
                "min_penetration":       (B,) float tensor,
                "mean_penetration":      (B,) float tensor,
                "per_link_per_sphere":   List[Dict[str, List[float]]]
                    # length B, for each batch:
                    #   link_name -> list of penetration depths of its spheres
                "num_colliding_points": (B,) int tensor
        """

        # ---- 1. Normalize scene points to (B, N, 3) on the correct device ----
        if isinstance(point_cloud, np.ndarray):
            pc = torch.from_numpy(point_cloud.astype(np.float32))
        else:
            pc = point_cloud.to(dtype=torch.float32)

        if pc.ndim == 2:
            pc = pc.unsqueeze(0)  # (1, N, 3)

        assert (
            pc.ndim == 3 and pc.shape[2] == 3
        ), f"point_cloud must have shape (B, N, 3) or (N, 3), got {pc.shape}"

        pc = pc.to(self.device)
        B, N, _ = pc.shape

        # ---- 2. Get transformed sphere centers for all batches ----
        transformed = self.compute_transformed_spheres(
            joints=joints,
            translations=translations,
            rotations=rotations,
        )
        # transformed: List[(radius, centers)], centers: (B, M_i, 3)

        centers_per_batch = []
        radii_list = []
        for radius, centers in transformed:
            centers_per_batch.append(centers)  # (B, M_i, 3)
            radii_list.extend([float(radius)] * centers.shape[1])

        if len(centers_per_batch) == 0:
            # No spheres, return zeros
            zero_bool = torch.zeros(B, dtype=torch.bool, device=self.device)
            zero_float = torch.zeros(B, dtype=torch.float32, device=self.device)
            result = {
                "is_colliding": zero_bool,
                "max_penetration": zero_float,
                "min_penetration": zero_float,
                "mean_penetration": zero_float,
                "per_link_per_sphere": [dict() for _ in range(B)],
            }
            if print_result:
                print("No collision spheres are defined; all stats are zero.")
            return result

        centers_cat = torch.cat(centers_per_batch, dim=1)  # (B, K, 3)
        K = centers_cat.shape[1]

        assert hasattr(self, "num_spheres") and hasattr(self, "sphere_link_indices"), (
            "sphere_link_indices / num_spheres not found. "
            "Please ensure _init_spheres_ builds them."
        )

        assert (
            K == self.num_spheres
        ), f"Inconsistent number of spheres: K={K}, num_spheres={self.num_spheres}"

        radii = torch.tensor(radii_list, device=self.device, dtype=torch.float32)  # (K,)

        # ---- 3. Compute minimal distance from each sphere center to scene points ----
        # centers_cat: (B, K, 3) -> (B, K, 1, 3)
        # pc:          (B, N, 3) -> (B, 1, N, 3)
        centers_expanded = centers_cat.unsqueeze(2)  # (B, K, 1, 3)
        pts_expanded = pc.unsqueeze(1)  # (B, 1, N, 3)

        dists = torch.norm(centers_expanded - pts_expanded, dim=-1)  # (B, K, N)
        min_dists, _ = torch.min(dists, dim=-1)  # (B, K)

        # ---- 4. Compute penetration depth for each sphere ----
        radii_expanded = radii.unsqueeze(0)  # (1, K)
        penetration = radii_expanded - min_dists  # (B, K)
        colliding_mask = penetration > 0.0  # (B, K)

        # ---- 5. Per-batch global statistics ----
        # is_colliding: any sphere colliding in that batch
        is_colliding = torch.any(colliding_mask, dim=1)  # (B,)

        # For global max / min / mean, only consider colliding spheres
        max_penetration = torch.zeros(B, dtype=torch.float32, device=self.device)
        min_penetration = torch.zeros(B, dtype=torch.float32, device=self.device)
        mean_penetration = torch.zeros(B, dtype=torch.float32, device=self.device)

        # number of colliding scene points per batch
        num_colliding_points = torch.zeros(B, dtype=torch.int64, device=self.device)

        # ---- 6. Per-batch, per-link, per-sphere penetration ----
        # per_link_per_sphere[b][link_name] = [penetration_depth_of_each_sphere]
        if need_per_link:
            per_link_per_sphere: List[Dict[str, List[float]]] = []
            all_link_names = sorted({name for name, _ in self.sphere_link_indices})
        else:
            per_link_per_sphere = [dict() for _ in range(B)]

        for b in range(B):
            batch_penetrations_all: List[float] = []
            # track which scene points are inside at least one colliding sphere
            colliding_points_mask = torch.zeros(N, dtype=torch.bool, device=self.device)

            if need_per_link:
                batch_link_to_list: Dict[str, List[float]] = {ln: [] for ln in all_link_names}

            for k in range(K):
                p = float(penetration[b, k].item())
                if p <= 0.0:
                    continue

                batch_penetrations_all.append(p)

                if need_per_link:
                    link_name, _local_idx = self.sphere_link_indices[k]
                    batch_link_to_list[link_name].append(p)

                # update colliding scene points for this batch
                center = centers_cat[b, k]  # (3,)
                diffs = pc[b] - center.unsqueeze(0)  # (N, 3)
                dists_k = torch.norm(diffs, dim=-1)  # (N,)
                # points within radius are considered colliding
                colliding_points_mask |= dists_k < radii[k]

            if need_per_link:
                per_link_per_sphere.append(batch_link_to_list)

            # number of scene points that lie inside at least one colliding sphere
            num_colliding_points[b] = int(colliding_points_mask.sum().item())

            if len(batch_penetrations_all) == 0:
                max_penetration[b] = 0.0
                min_penetration[b] = 0.0
                mean_penetration[b] = 0.0
            else:
                max_penetration[b] = max(batch_penetrations_all)
                min_penetration[b] = min(batch_penetrations_all)
                mean_penetration[b] = float(
                    sum(batch_penetrations_all) / len(batch_penetrations_all)
                )

        result = {
            "is_colliding": is_colliding,  # (B,) bool
            "max_penetration": max_penetration,  # (B,)
            "min_penetration": min_penetration,  # (B,)
            "mean_penetration": mean_penetration,  # (B,)
            "per_link_per_sphere": per_link_per_sphere,  # len B
            "num_colliding_points": num_colliding_points,  # (B,) int
        }

        # ---- 7. Optional printing ----
        if print_result:
            B_int = B
            for b in range(B_int):
                print(f"\n=== Batch {b} ===")
                print(f"Colliding: {bool(is_colliding[b].item())}")
                print(f"Max penetration:  {float(max_penetration[b].item()):.6f}")
                print(f"Min penetration:  {float(min_penetration[b].item()):.6f}")
                print(f"Mean penetration: {float(mean_penetration[b].item()):.6f}")

                if need_per_link:
                    link_to_list = per_link_per_sphere[b]
                    for ln, depths in link_to_list.items():
                        if len(depths) == 0:
                            continue
                        print(f"  Link: {ln}")
                        for i, d in enumerate(depths):
                            # print link name and per-sphere depth
                            print(f"    Link: {ln}, Sphere {i}: penetration = {d:.6f}")

        return result

    def compute_collision_with_pointcloud_time_series(
        self,
        point_cloud: Union[np.ndarray, torch.Tensor],
        joints: Union[np.ndarray, torch.Tensor],
        translations: Optional[Union[np.ndarray, torch.Tensor]] = None,
        rotations: Optional[Union[np.ndarray, torch.Tensor]] = None,
        print_result: bool = False,
        need_per_link: bool = False,
    ):
        """Given a time series of hand configurations and a (fixed or time-varying)
        scene point cloud, compute collision statistics for each batch and each time step.

        Expected shapes:
            - point_cloud:
                (B, T, N, 3): different point cloud per (batch, time), or
                (N, 3): single static point cloud shared by all (B, T)
            - joints: (B, T, D)
            - translations: (B, T, 3) or None (zeros)
            - rotations: (B, T, 3) or None (zeros)

        This function flattens (B, T, ...) to (B*T, ...) internally, calls the
        existing collision pipeline, then reshapes results back to (B, T, ...).

        Returns:
            dict with keys:
                "is_colliding":        (B, T) bool tensor
                "max_penetration":     (B, T) float tensor
                "min_penetration":     (B, T) float tensor
                "mean_penetration":    (B, T) float tensor
                "per_link_per_sphere": List[List[Dict[str, List[float]]]]
                    # length B, each item is a list of length T:
                    #   per_link_per_sphere[b][t][link_name] = list of depths
                "num_colliding_points": (B, T) int tensor
        """
        # ---- 0. Normalize joints / translations / rotations to torch and check shapes ----
        joints_t = torch.as_tensor(joints, dtype=torch.float32, device=self.device)
        assert joints_t.ndim == 3, f"joints must have shape (B, T, D), got {joints_t.shape}"
        B, T, D = joints_t.shape

        if translations is None:
            translations_t = torch.zeros(B, T, 3, dtype=torch.float32, device=self.device)
        else:
            translations_t = torch.as_tensor(translations, dtype=torch.float32, device=self.device)
            assert translations_t.shape == (
                B,
                T,
                3,
            ), f"translations must have shape (B, T, 3), got {translations_t.shape}"

        if rotations is None:
            rotations_t = torch.zeros(B, T, 3, dtype=torch.float32, device=self.device)
        else:
            rotations_t = torch.as_tensor(rotations, dtype=torch.float32, device=self.device)
            assert rotations_t.shape == (
                B,
                T,
                3,
            ), f"rotations must have shape (B, T, 3), got {rotations_t.shape}"

        # ---- 1. Normalize point cloud to (B*T, N, 3) ----
        if isinstance(point_cloud, np.ndarray):
            pc = torch.from_numpy(point_cloud.astype(np.float32)).to(self.device)
        else:
            pc = point_cloud.to(dtype=torch.float32, device=self.device)

        if pc.ndim == 2:
            # Static scene point cloud, shape (N, 3): repeat for all (B*T)
            N = pc.shape[0]
            pc = pc.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 3)
            pc = pc.expand(B, T, N, 3)  # (B, T, N, 3)
        elif pc.ndim == 3:
            if pc.shape[0] == 1:
                # Static scene point cloud, shape (1, N, 3): repeat for all (B*T)
                N = pc.shape[1]
                pc = pc.unsqueeze(0)  # (1, 1, N, 3)
                pc = pc.expand(B, T, N, 3)  # (B, T, N, 3)
            elif pc.shape[0] == B:
                # Static scene point cloud per batch, shape (B, N, 3): repeat for T
                N = pc.shape[1]
                pc = pc.unsqueeze(1)  # (B, 1, N, 3)
                pc = pc.expand(B, T, N, 3)  # (B, T, N, 3)
            else:
                raise ValueError(
                    f"point_cloud must have shape (B, T, N, 3) or (N, 3), got {pc.shape}"
                )
        elif pc.ndim == 4:
            # Time-varying point cloud, shape (B, T, N, 3)
            assert (
                pc.shape[0] == B and pc.shape[1] == T and pc.shape[3] == 3
            ), f"point_cloud must have shape (B, T, N, 3) or (N, 3), got {pc.shape}"
        else:
            raise ValueError(f"point_cloud must have shape (B, T, N, 3) or (N, 3), got {pc.shape}")

        B_pc, T_pc, N, _ = pc.shape
        assert B_pc == B and T_pc == T

        # Flatten (B, T, ...) -> (BT, ...)
        BT = B * T
        pc_flat = pc.view(BT, N, 3)  # (BT, N, 3)
        joints_flat = joints_t.view(BT, D)  # (BT, D)
        translations_flat = translations_t.view(BT, 3)  # (BT, 3)
        rotations_flat = rotations_t.view(BT, 3)  # (BT, 3)

        # ---- 2. Use existing compute_collision_with_pointcloud on flattened data ----
        # This returns results for a batch of size BT.

        base_result = self.compute_collision_with_pointcloud(
            point_cloud=pc_flat,  # (BT, N, 3)
            joints=joints_flat,  # (BT, D)
            translations=translations_flat,
            rotations=rotations_flat,
            print_result=False,
            need_per_link=need_per_link,
        )

        is_coll_bt = base_result["is_colliding"]  # (BT,)
        max_pen_bt = base_result["max_penetration"]  # (BT,)
        min_pen_bt = base_result["min_penetration"]  # (BT,)
        mean_pen_bt = base_result["mean_penetration"]  # (BT,)
        per_link_bt = base_result["per_link_per_sphere"]  # length BT, list[Dict[str, List[float]]]
        num_points_bt = base_result["num_colliding_points"]  # (BT,)

        # ---- 3. Reshape back to (B, T, ...) ----
        is_coll = is_coll_bt.view(B, T)
        max_pen = max_pen_bt.view(B, T)
        min_pen = min_pen_bt.view(B, T)
        mean_pen = mean_pen_bt.view(B, T)
        num_points = num_points_bt.view(B, T)

        # per_link_per_sphere: convert flat list of length BT to [B][T]
        per_link_time: List[List[Dict[str, List[float]]]] = []
        for b in range(B):
            row: List[Dict[str, List[float]]] = []
            for t in range(T):
                idx = b * T + t
                row.append(per_link_bt[idx])
            per_link_time.append(row)

        result = {
            "is_colliding": is_coll,  # (B, T)
            "max_penetration": max_pen,  # (B, T)
            "min_penetration": min_pen,  # (B, T)
            "mean_penetration": mean_pen,  # (B, T)
            "per_link_per_sphere": per_link_time,
            "num_colliding_points": num_points,  # (B, T)
        }

        # ---- 4. Optional printing in [B, T] style ----
        if print_result:
            all_link_names = sorted({name for name, _ in self.sphere_link_indices})

            for b in range(B):
                print(f"\n===== Batch {b} =====")
                for t in range(T):
                    print(f"\n--- Time {t} ---")
                    coll = bool(is_coll[b, t].item())
                    print(f"Colliding: {coll}")
                    print(f"Max penetration:  {float(max_pen[b, t].item()):.6f}")
                    print(f"Min penetration:  {float(min_pen[b, t].item()):.6f}")
                    print(f"Mean penetration: {float(mean_pen[b, t].item()):.6f}")

                    if need_per_link:
                        link_to_list = per_link_time[b][t]
                        for ln in all_link_names:
                            depths = link_to_list.get(ln, [])
                            if len(depths) == 0:
                                continue
                            print(f"  Link: {ln}")
                            for i, d in enumerate(depths):
                                print(f"    Link: {ln}, Sphere {i}: penetration = {d:.6f}")

        return result

    def visualize_sampled_pc(
        self,
        joints: Optional[Union[torch.Tensor, np.ndarray]] = None,
        translations: Optional[Union[torch.Tensor, np.ndarray]] = None,
        rotations: Optional[Union[torch.Tensor, np.ndarray]] = None,
        num_points: int = None,
    ):
        """Visualize the sampled point cloud using Open3D.

        The sampled point cloud may have one of the following shapes:
            - (N, M, 3): batched groups of point sets
            - (M, 3): a single point cloud
            - (3,): a single 3D point

        All formats will be reshaped into a final (num_points, 3) array.

        Args:
            cfg (Optional[torch.Tensor], optional):
                Robot configuration. Can be a single configuration (1D) or a batch of configurations (2D). Defaults to None.
            num_points (int, optional):
                Number of points to sample. Passed into `get_sampled_pc`. Defaults to None.
        """
        hand_sampled_pc = self.get_sampled_pc(joints, translations, rotations, num_points)[..., :3]

        if isinstance(hand_sampled_pc, torch.Tensor):
            pc = hand_sampled_pc.detach().cpu().numpy()
        else:
            pc = np.asarray(hand_sampled_pc)

        pc = np.asarray(pc, dtype=np.float64)

        if pc.ndim == 3:  # e.g. (N, M, 3) -> (N*M, 3)
            pc = pc.reshape(-1, pc.shape[-1])
        elif pc.ndim == 2:  # e.g. (M, 3) -> OK
            pass
        elif pc.ndim == 1:  # e.g. (3,) -> (1, 3)
            if pc.shape[0] == 3:
                pc = pc.reshape(1, 3)
            else:
                raise ValueError(f"Unexpected 1D shape for point cloud: {pc.shape}")
        else:
            raise ValueError(f"Unexpected point cloud shape: {pc.shape}")

        if pc.shape[1] != 3:
            raise ValueError(f"Point cloud last dimension must be 3, got {pc.shape}")

        # print("pc type:", type(pc), "shape:", pc.shape, "dtype:", pc.dtype)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)

        o3d.visualization.draw_geometries([pcd, frame])

    def visualize_transformed_spheres(
        self,
        joints: Optional[Union[torch.Tensor, np.ndarray]] = None,
        translations: Optional[Union[torch.Tensor, np.ndarray]] = None,
        rotations: Optional[Union[torch.Tensor, np.ndarray]] = None,
        batch_idx: int = 0,
        radius_scale: float = 1.0,
    ):
        """Visualize transformed collision spheres using Open3D.

        Args:
            joints [Tensor or np.ndarray, optional]:
                Joint configuration in Euler representation of shape (N,) or (B, N), where N is the number of joints.
            translations [Tensor or np.ndarray, optional]:
                Base translations of shape (3,) or (B, 3).
            rotations [Tensor or np.ndarray, optional]:
                Base rotations in Euler angles of shape (3,) or (B, 3).
            batch_idx (int, optional):
                Index of the batch to visualize when using batched input. Defaults to 0.
            radius_scale (float, optional):
                Scaling factor for the sphere radii. Defaults to 1.0.
        """
        transformed_spheres = self.compute_transformed_spheres(joints, translations, rotations)
        cmap = cm.get_cmap("viridis", len(transformed_spheres))
        assert (
            0 <= batch_idx < self.joints.shape[0]
        ), f"batch_idx {batch_idx} out of range [0, {self.joints.shape[0]-1}]"

        geometries = []
        for i, (radius, centers) in enumerate(transformed_spheres):
            centers_np = centers[batch_idx].detach().cpu().numpy()

            # get color from viridis colormap
            color = cmap(i)[:3]  # rgb only

            for center in centers_np:
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius * radius_scale))
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color(color)
                mesh.translate(center.tolist())
                geometries.append(mesh)

        # Add coordinate frame
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
        o3d.visualization.draw_geometries(geometries)


HANDCOLLISION = Registry("HandCollision")


def create_hand_collision(
    hand_name: str,
    **kwargs: Dict,
) -> HandCollisionBase:
    """Factory method to create hand collision detector instances."""
    return HANDCOLLISION.get(hand_name)(**kwargs)
