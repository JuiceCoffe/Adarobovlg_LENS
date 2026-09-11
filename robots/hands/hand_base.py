import sys
import trimesh
import numpy as np
import open3d as o3d
import os

from yourdfpy import URDF
from abc import abstractmethod, ABC
from utils.registry import Registry
from utils.mesh import trimesh_to_open3d
from typing import Dict, List, Optional, Tuple, Union
from utils.rotation_numpy import euler_to_matrix, matrix_to_euler, rot6d_to_euler


class HandBase(ABC):
    def __init__(
        self,
        urdf_path: str,
        use_cache: bool = False,
    ):
        """Load the URDF, independent joint order, grasp taxonomy, and joint synergy basis."""
        self.hand_name = None
        self.urdf = URDF.load(
            fname_or_file=urdf_path,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
        )
        self._link_map = self.urdf.link_map
        self._joint_map = self.urdf.joint_map
        self.link_names = [l.name for (_, l) in self._link_map.items()]
        # NOTE: exclude fixed and mimic joints
        self.joint_names = [
            j.name for j in self._joint_map.values() if j.type != "fixed" and j.mimic is None
        ]

        self.dof = len(self.joint_names)
        self.grasp_taxonomy = None
        self.grasp_all_info = None

        # set zero configuration and base pose
        self.zero_joints = self.urdf.zero_cfg
        self.zero_translation = np.zeros(3)
        self.zero_rotation = np.zeros(3)  # Euler angles

        self.joints = self.zero_joints
        self.translation = self.zero_translation
        self.rotation = self.zero_rotation  # Euler angles

        # synergy basis attributes
        self.mu, self.M, self.m, self.cum_energy = None, None, None, None

        self._init_internal_(use_cache)
        self._compute_synergy_basis(threshold=0.9)

    @property
    def current_cfg(self) -> Dict[str, float]:
        """Get the current configuration of the hand model,
        including translation(3), rotation(3), and joint angles(DoF).
        """
        assert np.isclose(
            self.urdf.cfg, self.joints, atol=1e-3
        ), "URDF configuration does not match the current joint configuration."
        return self.translation, self.rotation, self.urdf.cfg

    @property
    def zero_cfg(self) -> Dict[str, float]:
        """Get the zero configuration of the hand model,
        including zero translation(3), zero rotation(3), and zero joint angles(DoF).
        """
        return self.zero_translation, self.zero_rotation, self.zero_joints

    @abstractmethod
    def _init_internal_(self, use_cache: bool = False) -> None:
        """Initialize internal attributes specific to the hand model."""
        pass

    @abstractmethod
    def _init_from_cache_(self) -> bool:
        """Initialize the hand model from cached data if available."""
        pass

    @abstractmethod
    def _load_grasp_taxonomy(
        self,
        grasp_taxonomy_path: str,
    ) -> List:
        """Loads the grasp taxonomy from the specified file."""
        pass

    @abstractmethod
    def _get_predefined_joints(
        self,
        grasp_type: Union[int, str],
        grasp_width: float,
    ) -> np.array:
        """Retrieves a predefined joint configuration for a specified grasp type and width.
        The desired grasp width in centimeters (cm).
        """
        pass

    @abstractmethod
    def _get_predefined_pose(
        self,
        grasp_type: Union[int, str],
        grasp_width: float,
    ) -> np.array:
        """Retrieve the predefined principle closing pose (translation and rotation) for a robot hand."""
        pass

    @abstractmethod
    def _validate_grasp_type_and_width(
        self, grasp_type: Union[int, str], grasp_width: float
    ) -> None:
        """Validate whether the given grasp type and grasp width fall within
        the allowed range defined in the hand model's grasp taxonomy.
        """
        pass

    @property
    def grasp_type_ids(self) -> List[int]:
        """Get a list of all available grasp type IDs defined in the hand model's
        grasp taxonomy.
        """
        return range(1, len(self.grasp_all_info.keys()) + 1)

    def _compute_synergy_basis(
        self, threshold: float = 0.80
    ) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """Compute the synergy basis using PCA and automatically select the number
        of principal components (synergies) based on a cumulative variance threshold.

        Args:
            threshold (float):
                Minimum cumulative variance required to select synergy dimension m.
                Default: 0.80 (80%).

        Returns:
            mu (np.ndarray):
                Mean joint vector of shape (n,).
            M (np.ndarray):
                Synergy basis matrix of shape (n, m), formed by the first m principal components.
            m (int):
                Automatically selected synergy dimension.
            cum_energy (np.ndarray):
                Cumulative variance ratio curve of shape (n,).
        """

        all_joints_data = []

        # extract all the joint configuration data from 'self.grasp_all_info'
        for _, width_dict in self.grasp_all_info.items():
            for _, grasp_info in width_dict.items():
                joints = grasp_info["hand_angles"]
                all_joints_data.append(joints)
        all_joints_data = np.array(all_joints_data)  # (N, DoF)

        assert all_joints_data.ndim == 2, "all_joints_data must be of shape (K, n)."
        assert 0 < threshold <= 1.0, "threshold must be between 0 and 1."

        mu = all_joints_data.mean(axis=0)  # compute mean joint configuration

        X = all_joints_data - mu  # center the data

        U, S, Vt = np.linalg.svd(X, full_matrices=False)  # PCA using SVD

        energy = S**2
        cum_energy = np.cumsum(energy) / np.sum(energy)  # compute cumulative variance ratio

        # select m as the smallest number of components reaching the threshold
        m = int(np.searchsorted(cum_energy, threshold) + 1)

        M = Vt[:m].T  # select the first m principal components

        self.mu, self.M, self.m, self.cum_energy = mu, M, m, cum_energy

    @abstractmethod
    def get_predefined_joints(
        self,
        grasp_type: Union[int, str],
        grasp_width: float,
    ) -> np.array:
        """Retrieves a predefined joint configuration for a specified grasp type and width.
        The desired grasp width in centimeters (cm).
        """
        pass

    @abstractmethod
    def get_predefined_pose(
        self,
        grasp_type: Union[int, str],
        grasp_width: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve the predefined principle closing pose (translation and rotation) for a robot hand."""
        pass

    @abstractmethod
    def vis_predefined_grasp_type(
        self,
        grasp_type: Union[int, str],
        grasp_width: float,
        is_visualize: bool = True,
        is_export: bool = False,
        export_path: Optional[str] = None,
        export_format: str = "obj",
    ) -> None:
        """Visualizes a predefined joint configuration for a specified grasp type and width."""
        pass

    def vis_joints(
        self,
        joints: Optional[Union[Dict[str, float], np.ndarray, List[float]]] = None,
        collision_geometry: bool = False,
    ) -> None:
        """Visualizes the hand model using a specified joint configuration.
        Note that this method temporarily updates the hand model's joint configuration
        for visualization purposes and then resets it back to the original configuration

        Args:
            joints (dict, np.array, or List[float]): Joint configuration to visualize.
            collision_geometry (bool): If True, visualize the collision geometry instead
            of the visual geometry.
        """
        current_joints = self.joints
        self.update_cfg(joints=joints)
        self.urdf.show(collision_geometry)
        self.update_cfg(joints=current_joints)  # reset to previous configuration

    def vis_cfg(
        self,
        translation: Optional[Union[np.ndarray, List[float]]] = None,
        rotation: Optional[Union[np.ndarray, List[float]]] = None,
        joints: Optional[Union[Dict[str, float], np.ndarray, List[float]]] = None,
        collision_geometry: bool = False,
    ) -> None:
        """Visualizes the hand model using a specified configuration.
        Note that this method temporarily updates the hand model's configuration
        for visualization purposes and then resets it back to the original configuration

        Args:
            translation (np.ndarray or List[float], optional): Translation vector (3,).
            rotation (np.ndarray or List[float], optional): Rotation representation.
                Can be Euler angles (3,), rot6d (6,), or rotation matrix (9,).
            joints (dict, np.array, or List[float], optional): Joint configuration to visualize.
            collision_geometry (bool): If True, visualize the collision geometry instead
            of the visual geometry.
        """
        current_translation = self.translation
        current_rotation = self.rotation
        current_joints = self.joints

        self.update_cfg(translation=translation, rotation=rotation, joints=joints)

        if collision_geometry:
            self.trimesh_collision_mesh.show()
        else:
            self.trimesh_mesh.show()

        # reset to previous configuration
        self.update_cfg(
            translation=current_translation, rotation=current_rotation, joints=current_joints
        )

    def update_cfg(
        self,
        joints: Optional[Union[Dict[str, float], np.ndarray, List[float]]] = None,
        translation: Optional[Union[np.ndarray, List[float]]] = None,
        rotation: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> None:
        """Update robot configuration of the hand model. If the arguments is None, the corresponding part of the configuration is set to zero.

        Args:
            joints (dict, np.ndarray, or List[float], optional): New joint configurations.
                If dict, should map joint names to angles. If array or list, should be of length DoF.
            translation (np.ndarray or List[float], optional): New translation vector (3,).
            rotation (np.ndarray or List[float], optional): New rotation representation.
                Can be Euler angles (3,), rot6d (6,), or rotation matrix (9,).
        """
        # reset to zero configuration
        self.translation, self.rotation, self.joints = self.zero_cfg

        # update translation if provided
        if translation is not None:
            assert len(translation) == 3, "Translation must be a 3D vector."
            self.translation = np.asarray(translation, dtype=np.float32)

        # update rotation if provided
        if rotation is not None:
            if len(rotation) == 3:
                self.rotation = np.asarray(rotation, dtype=np.float32)
            elif len(rotation) == 6:
                self.rotation = rot6d_to_euler(rotation)
            elif len(rotation) == 9:
                self.rotation = matrix_to_euler(rotation.reshape(3, 3))
            else:
                raise ValueError(
                    f"Invalid rotation length {len(rotation)}: expected "
                    f"3 (Euler), 6 (rot6d), or 9 (rotation matrix)."
                )

        # update joint configurations if provided
        if joints is not None:
            angles = {}
            for idx in range(self.dof):
                angles[self.joint_names[idx]] = float(joints[idx])
            self.joints = joints

            self.urdf.update_cfg(angles)

    def set_zero_cfg(self) -> None:
        """Reset the hand model to its zero configuration."""
        self.translation, self.rotation, self.joints = self.zero_cfg
        self.urdf.update_cfg(self.joints)

    def joints_to_synergy(
        self,
        joints: Union[np.ndarray, float],
    ) -> np.ndarray:
        """Convert joint configurations into synergy coordinates.

        Args:
            joints (np.ndarray or float): Joint configurations (..., n).

        Returns:
            np.ndarray: Synergy coordinates (..., m).
        """
        joints = np.asarray(joints)
        return (joints - self.mu) @ self.M

    def synergy_to_joints(
        self,
        c: Union[np.ndarray, float],
    ) -> np.ndarray:
        """Convert synergy coordinates back into joint configurations.

        Args:
            c (np.ndarray or float): Synergy coordinates (..., m).

        Returns:
            np.ndarray: Joint configurations (..., n).
        """
        return self.mu + c @ self.M.mT

    @property
    def trimesh_mesh(self) -> trimesh.Trimesh:
        """Combines all visual meshes of the hand into a single Trimesh object."""
        base_matrix = np.vstack(
            (
                np.hstack(
                    (euler_to_matrix(self.rotation).reshape(3, 3), self.translation.reshape(3, 1))
                ),
                np.array((0, 0, 0, 1)),
            )
        )
        return self.urdf.scene.dump(concatenate=True).apply_transform(base_matrix)

    @property
    def trimesh_collision_mesh(self) -> trimesh.Trimesh:
        """Combines all collision meshes of the hand into a single Trimesh object."""
        base_matrix = np.vstack(
            (
                np.hstack(
                    (euler_to_matrix(self.rotation).reshape(3, 3), self.translation.reshape(3, 1))
                ),
                np.array((0, 0, 0, 1)),
            )
        )
        return self.urdf.collision_scene.dump(concatenate=True).apply_transform(base_matrix)

    @property
    def open3d_mesh(self) -> o3d.geometry.TriangleMesh:
        """Converts a Trimesh object to an Open3D TriangleMesh.

        Returns:
            o3d.geometry.TriangleMesh: Open3D TriangleMesh with vertices, faces, and
            optionally vertex colors.
        """
        return trimesh_to_open3d(self.trimesh_mesh)

    @property
    def open3d_collision_mesh(self) -> o3d.geometry.TriangleMesh:
        """Converts a Trimesh collision object to an Open3D TriangleMesh.

        Returns:
            o3d.geometry.TriangleMesh: Open3D TriangleMesh with vertices, faces, and
            optionally vertex colors.
        """
        return trimesh_to_open3d(self.trimesh_collision_mesh)


HAND = Registry("Hand")


def create_hand_model(
    hand_name: str,
    **kwargs: Dict,
) -> HandBase:
    """Factory method to create hand model instances."""
    return HAND.get(hand_name)(**kwargs)
