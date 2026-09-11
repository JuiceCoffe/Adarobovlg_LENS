"""Define shared multi-finger grasp and grasp-group representations."""

from threading import Lock

import numpy as np

from graspnetAPI import Grasp
from typing import Dict, Union, List
from utils.registry import Registry
from abc import abstractmethod, ABC
from graspnetAPI.grasp import GraspGroup


class _LazyHandModel:
    """Load and cache one hand model on first class or instance attribute access."""

    def __init__(self, hand_name: str):
        """Record the hand type without reading URDF, meshes, or taxonomy assets."""
        self.hand_name = hand_name
        self._model = None
        self._lock = Lock()

    def __get__(self, instance, owner=None):
        """Return the shared model; failed initialization remains retryable."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from robots.hands.hand_base import create_hand_model

                    self._model = create_hand_model(self.hand_name, use_cache=True)
        return self._model


class MultiFingerGraspBase(ABC):
    def __init__(self, *args):
        """Initialize MultiFingerGrasp object."""
        self.mf_garray = None
        self.mf_gtax = None

    @abstractmethod
    def __repr__(self):
        """Print the grasp information."""
        return f"{self.__class__.__name__}()"

    @property
    @abstractmethod
    def score(self) -> float:
        """Grasp score."""
        pass

    @score.setter
    @abstractmethod
    def score(self, score: float):
        """Set grasp score."""
        pass

    @property
    @abstractmethod
    def depth(self) -> float:
        """Grasp depth."""
        pass

    @depth.setter
    @abstractmethod
    def depth(self, depth: float):
        """Set grasp depth."""
        pass

    @property
    @abstractmethod
    def grasp_type(self) -> int:
        """Grasp type."""
        pass

    @grasp_type.setter
    @abstractmethod
    def grasp_type(self, grasp_type: int):
        """Set grasp type."""
        pass

    @property
    @abstractmethod
    def rotation_matrix(self) -> np.ndarray:
        """Rotation matrix of the grasp pose."""
        pass

    @rotation_matrix.setter
    @abstractmethod
    def rotation_matrix(self, *args):
        """Set rotation matrix of the grasp pose."""
        pass

    @property
    @abstractmethod
    def translation(self) -> np.ndarray:
        """Translation of the grasp pose."""
        pass

    @translation.setter
    @abstractmethod
    def translation(self, *args):
        """Set translation of the grasp pose."""
        pass

    @property
    @abstractmethod
    def angle(self) -> np.ndarray:
        """Joint angles of the hand."""
        pass

    @angle.setter
    @abstractmethod
    def angle(self, *args):
        """Set joint angles of the hand."""
        pass

    @property
    @abstractmethod
    def object_id(self) -> int:
        """Object id associated with this grasp."""
        pass

    @object_id.setter
    @abstractmethod
    def object_id(self, object_id: int):
        """Set object id."""
        pass

    @property
    @abstractmethod
    def width(self) -> float:
        """Grasp width (m)."""
        pass

    @width.setter
    @abstractmethod
    def width(self, width: float):
        """Set grasp width (clipped to valid range)."""
        pass

    @abstractmethod
    def get_array_grasp(self) -> np.ndarray:
        """Get grasp as numpy array."""
        pass

    @abstractmethod
    def get_grasp_type_with_finger_name(self) -> str:
        """Get grasp type with finger names."""
        pass

    @abstractmethod
    def _tfTR_2_mfTR(self, two_finger_grasp: Grasp):
        """Convert two-finger grasp transform to multi-finger grasp transform."""
        pass

    @abstractmethod
    def from_grasp(self, two_finger_grasp: Grasp, grasp_type: int):
        """Grasp to MultiFingerGrasp Transformation."""
        pass

    @abstractmethod
    def to_open3d_geometry(self):
        """Convert the grasp group to Open3D geometry for visualization.

        Returns:
            list of open3d.geometry.Geometry: The list of Open3D geometries representing the grasps.
        """
        pass


class MultiFingerGraspGroupBase(ABC):
    def __init__(self, *args):
        """Initialize MultiFingerGraspGroup object."""
        self.mf_ggarray = None
        self.mf_gtax = None

    def __len__(self):
        """Get number of grasps in the group."""
        return len(self.mf_ggarray)

    @abstractmethod
    def __repr__(self):
        """Print the grasp group information."""
        pass

    @abstractmethod
    def __getitem__(
        self, index: Union[int, slice, np.ndarray, list]
    ) -> Union[MultiFingerGraspBase, "MultiFingerGraspGroupBase"]:
        """Get grasp at index."""
        pass

    @property
    @abstractmethod
    def scores(self) -> np.ndarray:
        """Get scores of the grasps in the group.

        Returns:
            numpy array of shape (-1, ) of the scores.
        """
        pass

    @scores.setter
    @abstractmethod
    def scores(self, scores: Union[np.ndarray, list]):
        """Set scores of the grasps in the group.

        Args:
            scores(numpy array of shape (-1, )): the scores.
        """
        pass

    @property
    @abstractmethod
    def depths(self) -> np.ndarray:
        """Get depths of the grasps in the group.

        Returns:
            numpy array of shape (-1, ) of the depths.
        """
        pass

    @depths.setter
    @abstractmethod
    def depths(self, depths: Union[np.ndarray, list]):
        """Set depths of the grasps in the group.

        Args:
            depths(numpy array of shape (-1, )): the depths.
        """
        pass

    @property
    @abstractmethod
    def grasp_types(self) -> np.ndarray:
        """Get grasp types of the grasps in the group.

        Returns:
            numpy array of shape (-1, ) of the grasp types.
        """
        pass

    @grasp_types.setter
    @abstractmethod
    def grasp_types(self, grasp_types: Union[np.ndarray, list]):
        """Set grasp types of the grasps in the group.

        Args:
            grasp_types(numpy array of shape (-1, )): the grasp types.
        """
        pass

    @property
    @abstractmethod
    def rotation_matrices(self) -> np.ndarray:
        """Get rotation_matrices of the grasps in the group.

        Returns:
            np.array of shape (-1, 3, 3) of the rotation_matrices
        """
        pass

    @rotation_matrices.setter
    @abstractmethod
    def rotation_matrices(self, rotation_matrices: Union[np.ndarray, list]):
        """Set rotation_matrices of the grasps in the group.

        Args:
            rotation_matrices(numpy array or list of shape (-1, 3, 3)): the rotation_matrices.
        """
        pass

    @property
    @abstractmethod
    def translations(self) -> np.ndarray:
        """Get translations of the grasps in the group.

        Returns:
            np.array of shape (-1, 3) of the translations.
        """
        pass

    @translations.setter
    @abstractmethod
    def translations(self, translations: Union[np.ndarray, list]):
        """Set translations of the grasps in the group.

        Args:
            translations(numpy array or list of shape (-1, 3)): the translations.
        """
        pass

    @property
    @abstractmethod
    def angles(self) -> np.ndarray:
        """Get joint angles of the grasps in the group.

        Returns:
            numpy array of shape (-1, 16) of the angles.
        """
        pass

    @angles.setter
    @abstractmethod
    def angles(self, angles: Union[np.ndarray, list]):
        """Set joint angles of the grasps in the group.

        Args:
            angles(numpy array or list of shape (-1, 16)): the angles.
        """
        pass

    @property
    @abstractmethod
    def object_ids(self) -> np.ndarray:
        """Get object_ids of the grasps in the group.

        Returns:
            numpy array of shape (-1, ) of the object_ids.
        """
        pass

    @object_ids.setter
    @abstractmethod
    def object_ids(self, object_ids: Union[np.ndarray, list]):
        """Set object_ids of the grasps in the group.

        Args:
            object_ids(numpy array or list of shape (-1, )): the object_ids.
        """
        pass

    @property
    @abstractmethod
    def widths(self) -> np.ndarray:
        """Get widths of the grasps in the group.

        Returns:
            numpy array of shape (-1, ) of the widths.
        """
        pass

    @widths.setter
    @abstractmethod
    def widths(self, widths: Union[np.ndarray, list]):
        """Set widths of the grasps in the group.

        Args:
            widths(numpy array or list of shape (-1, )): the widths.
        """
        pass

    @abstractmethod
    def get_graspgroup_types_with_finger_names(self):
        """Get graspgroup types with finger names. Converts the grasp type ids to finger names.

        Returns:
            list of str: the graspgroup types with finger names.
        """
        pass

    @abstractmethod
    def _tfGroupTR_2_mfGroupTR(self, tf_graspgroup: GraspGroup):
        """Convert two-finger grasp group transform to multi-finger grasp group transform.

        Args:
            tf_graspgroup (graspnetAPI.GraspGroup): Two-finger grasp group object.
        """
        pass

    @abstractmethod
    def from_graspgroup(
        self,
        graspgroup: GraspGroup,
        grasp_types: Union[np.ndarray, list],
    ) -> "MultiFingerGraspGroupBase":
        """Convert GraspGroup to MultiFingerGraspGroupBase.

        Args:
            graspgroup (graspnetAPI.GraspGroup): Two-finger grasp group object.
            grasp_types (numpy array or list of shape (-1, )): the grasp types for each grasp in the group.

        Returns:
            MultiFingerGraspGroupBase: The converted MultiFingerGraspGroupBase object.
        """
        pass

    @abstractmethod
    def sort_by_score(self, reverse: bool = False):
        """Sort the grasp group by score.

        Args:
            reverse (bool): If True, sort in ascending order. Default is False (descending order).

        Returns:
            MultiFingerGraspGroupBase: The sorted grasp group.
        """
        pass

    @abstractmethod
    def update_by_id(self, index: Union[list, np.ndarray, slice]):
        """Update the grasp group by index.

        Args:
            index (list, np.ndarray, slice): The indices to update the grasp group.
        """
        pass


MULTIFINGERGRASP = Registry("MultiFingerGrasp")


def create_hand_grasp(
    hand_name: str,
    **kwargs: Dict,
) -> MultiFingerGraspBase:
    """Factory method to create multi-finger hand grasp instances."""
    return MULTIFINGERGRASP.get(hand_name)(**kwargs)


MULTIFINGERGRASPGROUP = Registry("MultiFingerGraspGroup")


def create_hand_graspgroup(
    hand_name: str,
    *args: List,
) -> MultiFingerGraspGroupBase:
    """Factory method to create multi-finger hand grasp group instances."""
    return MULTIFINGERGRASPGROUP.get(hand_name)(*args)
