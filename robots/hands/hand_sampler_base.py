import re
import torch
import trimesh
import logging
import numpy as np
import open3d as o3d

from utils.rotation_torch import euler_to_matrix
from utils.torch_urdf import TorchURDF
from pathlib import Path
from utils.registry import Registry
from utils.io import mkdir_if_not_exists
from utils.mesh import load_link_geometries
from typing import Dict, List, Optional, Tuple, Union
from utils.transform import transform_pointcloud_torch
from cprint import cprint


class HandSamplerBase(object):
    """This class allows for fast pointcloud sampling from the surface of a hand.
    At initialization, it loads a URDF and samples points from the mesh of each link.
    The points per link are based on the (very approximate) surface area of the link.

    Then, after instantiation, the sample method takes in a batch of configurations
    and produces pointclouds for each configuration by running fk on a subsample
    of the per-link pointclouds that are established at initialization.
    """

    def __init__(
        self,
        urdf_path: str,
        device: Union[str, torch.device] = "cpu",
        num_fixed_points: Optional[int] = None,
        use_cache: bool = False,
        pointcloud_cache: str = None,
        max_points: int = 4096,
    ):
        """Initializes the sampler object, sets the logging level for trimesh,
        stores the number of fixed points, and calls the internal initialization method.

        Args:
            urdf_path [str]:
                The file path to the URDF of the robot hand.
            device [torch.device or str]:
                The device to be used for computation (e.g., 'cpu' or 'cuda').
            num_fixed_points [int, optional]:
                The number of fixed points to use. Defaults to None.
            use_cache [bool, optional]:
                Whether to use caching for internal operations. Defaults to False.
            pointcloud_cache [str, optional]:
                The directory path for caching pointcloud data. Defaults to None.
            max_points [int, optional]:
                The maximum number of points allowed. Defaults to 4096.

        Return:
            None
        """
        logging.getLogger("trimesh").setLevel("ERROR")
        self.num_fixed_points = num_fixed_points
        self._init_internal_(
            device,
            urdf_path,
            use_cache,
            pointcloud_cache,
            max_points,
        )

    def _init_internal_(
        self,
        device: Union[str, torch.device],
        urdf_path: str,
        use_cache: bool,
        pointcloud_cache: Optional[str],
        max_points: int,
    ):
        """Initializes internal data structures for the robot hand model, loads meshes or creates geometric primitives for each link,
        samples surface points on each link proportional to their surface area, and optionally caches the sampled points for future use.

        Args:
            urdf_path [str]:
                The file path to the URDF of the robot hand.
            device [torch.device or str]:
                The device on which tensors and robot data should be loaded (e.g., 'cpu' or 'cuda').
            use_cache [bool]:
                Whether to use cached sampled points if available, or generate and cache new ones.
            pointcloud_cache [str]:
                The directory path for caching pointcloud data.
            max_points [int]:
                The maximum total number of points to sample across all robot links if num_fixed_points is not set.

        Return:
            None. The function sets up internal attributes such as self.point_cloud and may save sampled points to a cache file.
        """
        self.hand_name = None
        self.pointcloud_cache = pointcloud_cache
        self.device = device
        self.max_points = max_points

        self.urdf = TorchURDF.load(urdf_path, lazy_load_meshes=True, device=device)
        self.link_names = [l.name for l in self.urdf.links if len(l.visuals)]
        self.joint_names = [j.name for j in self.urdf.actuated_joints]
        self.dof = len(self.joint_names)
        self.frame_status = None
        self.meshes = load_link_geometries(urdf_path, self.link_names)

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

        # if use_cache and self._init_from_cache_(device):
        #     return
        if use_cache:
            try:
                if self._init_from_cache_(device):
                    return
            except Exception as e:
                cprint.info(f"Failed to load cache due to {e}, regenerating point cloud.")

        areas = [m.bounding_box_oriented.area for _, m in self.meshes.items()]
        if self.num_fixed_points is not None:
            num_points = np.round(self.num_fixed_points * np.array(areas) / np.sum(areas))
            num_points[0] += self.num_fixed_points - np.sum(num_points)
            assert np.sum(num_points) == self.num_fixed_points
        else:
            num_points = np.round(max_points * np.array(areas) / np.sum(areas))

        self.point_cloud = {}
        for id, (l_name, l_mesh) in enumerate(self.meshes.items()):
            # sample points on the mesh surface
            points, face_idx = trimesh.sample.sample_surface(l_mesh, int(num_points[id]))

            # calculate point normals using barycentric interpolation ---
            triangles = l_mesh.triangles[face_idx]
            bary_value = trimesh.triangles.points_to_barycentric(triangles, points)
            face_vertex_normals = l_mesh.vertex_normals[l_mesh.faces[face_idx]]

            # interpolate the vertex normals using the barycentric coordinates
            normals = (face_vertex_normals * bary_value[..., None]).sum(axis=1)
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)

            # combine the point coordinates and normals into a single tensor
            self.point_cloud[l_name] = torch.as_tensor(
                np.concatenate([points, normals], axis=-1), device=device
            ).unsqueeze(0)

        # If we made it all the way here with the use_cache flag set,
        # then we should be creating new cache files locally
        if use_cache:
            points_to_save = {
                k: tensor.squeeze(0).cpu().numpy() for k, tensor in self.point_cloud.items()
            }
            file_name = self._get_cache_file_name_()
            print(f"Saving new file to cache: {file_name}")
            np.save(file_name, points_to_save)

    def _init_from_cache_(self, device: Union[str, torch.device]):
        """Loads cached point cloud data from a file if it exists, and initializes the
        `self.point_cloud` attribute with the loaded data as PyTorch tensors on the specified device.

        Args:
            device [torch.device or str]:
                The device on which the loaded tensors should be allocated.

        Return:
            bool:
                Returns True if the cache file exists and the data is successfully loaded; otherwise, returns False.
        """
        file_name = self._get_cache_file_name_()
        if not file_name.is_file():
            return False

        points = np.load(
            file_name,
            allow_pickle=True,
        )
        self.point_cloud = {
            key: torch.as_tensor(pc, device=device).unsqueeze(0)
            for key, pc in points.item().items()
        }
        return True

    def _get_cache_file_name_(self) -> Path:
        """Returns the file path for the cached point cloud data based on the number of fixed points.
        If the number of fixed points is specified, returns the corresponding fixed point cloud cache file path.
        Otherwise, returns the full point cloud cache file path.

        Args:
            self: Instance of the class containing configuration and cache directory information.

        Return:
            Path: The file path to the cached point cloud data as a Path object.
        """
        mkdir_if_not_exists(self.pointcloud_cache, True)
        if self.num_fixed_points is not None:
            return Path(self.pointcloud_cache) / f"fixed_point_cloud_{self.num_fixed_points}.npy"
        else:
            return Path(self.pointcloud_cache) / "full_point_cloud.npy"

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples points from the surface of the hand by calling fk, and rotates both points and normals.

        Args:
            joints [Tensor or np.ndarray, optional]:
                Joint configuration in Euler representation of shape (N,) or (B, N), where N is the number of joints.
            translations [Tensor or np.ndarray, optional]:
                Base translations of shape (3,) or (B, 3).
            rotations [Tensor or np.ndarray, optional]:
                Base rotations in Euler angles of shape (3,) or (B, 3).
            num_points [int, optional]:
                Number of points to sample. If None, returns all points.
        Returns:
            sampled_pc [Tensor]:
                Sampled point cloud of shape (B, num_points, 3) if num_points is specified, else (B, total_points, 3).
            sampled_normals [Tensor]:
                Corresponding normals of shape (B, num_points, 3) if num_points is specified, else (B, total_points, 3).
        """
        assert bool(self.num_fixed_points is not None) or bool(num_points is not None)

        self.update_status(joints, translations, rotations)

        fk_points, fk_normals = [], []
        for l_name in self.meshes.keys():

            # transform points
            points = transform_pointcloud_torch(
                self.point_cloud[l_name][:, :, :3]
                .float()
                .expand(self.frame_status[l_name].shape[0], -1, -1),
                torch.matmul(self.base_matrix, self.frame_status[l_name]),
                in_place=False,
            )
            fk_points.append(points)

            # transform normals (rotate only, do not translate)
            normals = torch.matmul(
                self.point_cloud[l_name][:, :, 3:]
                .float()
                .expand(self.frame_status[l_name].shape[0], -1, -1),
                torch.matmul(self.base_matrix, self.frame_status[l_name])[:, :3, :3].mT,
            )

            fk_normals.append(normals)

        # concatenate all the transformed points and normals
        pc = torch.cat(fk_points, dim=1)  # (N, num_points, 3)
        normals = torch.cat(fk_normals, dim=1)  # (N, num_points, 3)

        if num_points is None:
            return pc, normals  # return both points and normals

        # if num_points is specified, sample random points
        idx = np.random.choice(pc.shape[1], num_points, replace=False)
        sampled_pc = pc[:, idx, :]
        sampled_normals = normals[:, idx, :]

        return sampled_pc, sampled_normals

    def visualize_sampled_pc(
        self,
        joints: Optional[Union[torch.Tensor, np.ndarray]] = None,
        translations: Optional[Union[torch.Tensor, np.ndarray]] = None,
        rotations: Optional[Union[torch.Tensor, np.ndarray]] = None,
        num_points: int = None,
        batch_idx: int = 0,
        visualize_normals: bool = False,
    ):
        """Visualizes the sampled point cloud using Open3D. Optionally visualizes normals as arrows.

        Args:
            joints [Tensor or np.ndarray, optional]:
                Joint configuration in Euler representation of shape (N,) or (B, N), where N is the number of joints.
            translations [Tensor or np.ndarray, optional]:
                Base translations of shape (3,) or (B, 3).
            rotations [Tensor or np.ndarray, optional]:
                Base rotations in Euler angles of shape (3,) or (B, 3).
            num_points [int, optional]:
                Number of points to sample. If None, returns all points.
            batch_idx [int, optional]:
                Index of the batch element to visualize. Defaults to 0.
            visualize_normals [bool, optional]:
                Whether to visualize normals as arrows. Defaults to False.
        """
        # sample point cloud
        sampled_points, sampled_normals = self.get_sampled_pc(
            joints,
            translations,
            rotations,
            num_points,
        )

        points = sampled_points.detach().cpu().numpy()
        normals = sampled_normals.detach().cpu().numpy()

        B = points.shape[0]
        if not (0 <= batch_idx < B):
            raise ValueError(f"batch_idx {batch_idx} out of range [0, {B - 1}]")
        points, normals = points[batch_idx], normals[batch_idx]

        if visualize_normals:
            # normalize and scale normals (control arrow length)
            normal_length = 0.02  # arrow length, adjustable based on point cloud scale
            normals_normalized = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
            normals_scaled = normals_normalized * normal_length

            # construct start and end points for lines
            points_start = points
            points_end = points + normals_scaled

            # combine start and end points for all lines
            line_points = np.vstack([points_start, points_end])

            # create lines: each line connects the i-th start point to the i-th end point
            lines = np.array([[i, i + len(points)] for i in range(len(points))])

            # create LineSet
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)

            # color the lines (e.g., red)
            colors = [[1, 0, 0] for _ in range(len(lines))]  # red color
            line_set.colors = o3d.utility.Vector3dVector(colors)

            # add line set to the list of geometries
            geometries = [line_set]
        else:
            geometries = []

        # original point cloud (without normals since we're drawing them ourselves)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

        # visualization: show point cloud, frame, and optionally line set (normals)
        o3d.visualization.draw_geometries([pcd, frame] + geometries)

    def get_canonical_joints(self):
        """Generate a canonical joint configuration for visualization.
        This configuration is not intended for control or planning purposes.
        It simply provides a mid-range pose of the robot hand within its
        joint limits, which is useful for displaying meshes or checking
        kinematic structure.

        Returns:
            torch.Tensor:
                Canonical joint configuration of shape (dof, ).
        """
        lowers, uppers = [], []

        for j in self.urdf.actuated_joints:
            if j.limit is not None:
                lowers.append(j.limit.lower)
                uppers.append(j.limit.upper)
            else:
                lowers.append(-np.pi)
                uppers.append(np.pi)

        lower = torch.as_tensor(lowers, dtype=torch.float32)
        upper = torch.as_tensor(uppers, dtype=torch.float32)

        canonical_joints = lower * 0.75 + upper * 0.25
        return canonical_joints

    def get_trimesh_cfg(
        self,
        joints: Optional[Union[torch.Tensor, np.ndarray]] = None,
        translations: Optional[Union[torch.Tensor, np.ndarray]] = None,
        rotations: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> List[Dict[str, trimesh.Trimesh]]:
        """Return the hand trimesh objects corresponding to the given joint configuration(s)."""
        # Update forward kinematics; this should set self.frame_status appropriately
        self.update_status(joints, translations, rotations)

        batch_size = self.batch_size
        results: List[Dict[str, trimesh.Trimesh]] = []

        for b in range(batch_size):
            # Build a scene for this batch element
            scene = trimesh.Scene()

            for link_name in self.link_names:
                # Get transform matrix for this link and this batch index
                mat = torch.matmul(self.base_matrix, self.frame_status[link_name])

                if mat.ndim == 3:
                    T = mat[b]
                elif mat.ndim == 2:
                    # Same transform for all batch items
                    T = mat
                else:
                    raise ValueError(
                        f"Unexpected transform matrix shape {mat.shape} for link '{link_name}'."
                    )

                T = T.cpu().numpy()
                mesh = self.meshes[link_name].copy()
                mesh.apply_transform(T)
                scene.add_geometry(mesh)

            # Merge all geometries into a single visual mesh
            vertices = []
            faces = []
            vertex_offset = 0

            for geom in scene.geometry.values():
                if isinstance(geom, trimesh.Trimesh):
                    v = np.asarray(geom.vertices)
                    f = np.asarray(geom.faces)
                    vertices.append(v)
                    faces.append(f + vertex_offset)
                    vertex_offset += len(v)

            all_vertices = np.vstack(vertices) if len(vertices) > 0 else np.zeros((0, 3))
            all_faces = np.vstack(faces) if len(faces) > 0 else np.zeros((0, 3), dtype=np.int64)

            # Build per-part meshes
            parts: Dict[str, trimesh.Trimesh] = {}
            for link_name in self.meshes:
                mat = self.frame_status[link_name]

                if mat.ndim == 3:
                    T = mat[b]
                elif mat.ndim == 2:
                    T = mat
                else:
                    raise ValueError(
                        f"Unexpected transform matrix shape {mat.shape} for link '{link_name}'."
                    )

                T = T.cpu().numpy()
                part_mesh = self.meshes[link_name].copy()
                part_mesh.apply_transform(T)
                parts[link_name] = part_mesh

            result = {
                "visual": trimesh.Trimesh(vertices=all_vertices, faces=all_faces),
                "parts": parts,
            }
            results.append(result)

        return results


HANDSAMPLER = Registry("HandSampler")


def create_hand_sampler(
    hand_name: str,
    **kwargs: Dict,
) -> HandSamplerBase:
    """
    Factory method to create hand sampler model instances.
    """
    return HANDSAMPLER.get(hand_name)(**kwargs)
