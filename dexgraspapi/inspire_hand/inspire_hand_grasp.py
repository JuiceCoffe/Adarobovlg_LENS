"""Define Inspire hand grasp and grasp-group representations."""

# from __future__ import annotations
import os
import json
import copy
import numpy as np

from typing import Union
from graspnetAPI import Grasp
from graspnetAPI.grasp import GraspGroup
from robots.hands.grasp_taxonomy import InspireGraspTaxonomy
from dexgraspapi.multi_finger_grasp_base import (
    _LazyHandModel,
    MULTIFINGERGRASP,
    MULTIFINGERGRASPGROUP,
    MultiFingerGraspBase,
    MultiFingerGraspGroupBase,
)

MAX_GRASP_WIDTH = 0.10
MIN_GRASP_WIDTH = 0.01
INSPIRE_DEFAULT_DEPTH = 0.0
INSPIRE_ARRAY_LEN = 23


@MULTIFINGERGRASP.register()
class InspireHandGrasp(MultiFingerGraspBase):
    mf_model = _LazyHandModel("InspireHand")

    def __init__(self, *args):
        """Initialize InspireHandGrasp object.

        Args:
            args:
                - No argument: initializes with default values.
                - One argument (np.ndarray): initializes with the given grasp array.

        The format of numpy array is [score, depth, grasp_type, rot_matrix(9), translation(3), joint_angles(6), obj_id, width]

        Note: width is in meters.
        """
        super().__init__(self, *args)
        if len(args) == 0:
            self.mf_garray = np.array(
                [0, 0, 1]  # score, depth, grasp_type
                + [1, 0, 0, 0, 1, 0, 0, 0, 1]  # rotation matrix
                + [0] * 3  # translation
                + [0] * 6  # joint angles
                + [-1, 0],  # obj_id, width
                dtype=np.float64,
            )
        elif len(args) == 1:
            if type(args[0]) == np.ndarray:
                self.mf_garray = copy.deepcopy(args[0])
            elif type(args[0]) == list:
                self.mf_garray = np.array(args[0], dtype=np.float64)
            else:
                raise TypeError("Argument must be a numpy array or a list.")
            if self.mf_garray.shape[0] != INSPIRE_ARRAY_LEN:
                raise ValueError("Grasp array must have length of {}.".format(INSPIRE_ARRAY_LEN))
        else:
            raise ValueError("Only zero or one argument is allowed.")

        self.mf_gtax = InspireGraspTaxonomy

    def __repr__(self):
        """Return a readable representation of the stored configuration."""
        return (
            "InspireHandGrasp: "
            "score:{}, depth:{}, grasp_type:{}, translation:{}\n"
            "rotation:\n{}\n"
            "object_id:{}, angle:{}, width:{}"
        ).format(
            self.score,
            self.depth,
            self.grasp_type,
            self.translation,
            self.rotation_matrix,
            self.object_id,
            self.angle,
            self.width,
        )

    @property
    def score(self) -> float:
        """Grasp score.

        Returns:
            float: Score of the grasp.
        """
        return self.mf_garray[0]

    @score.setter
    def score(self, score: float):
        """Set grasp score.

        Args:
            score (float): Score of the grasp.
        """
        self.mf_garray[0] = copy.deepcopy(score)

    @property
    def depth(self) -> float:
        """Grasp depth.

        Returns:
            float: Depth of the grasp.
        """
        return self.mf_garray[1]

    @depth.setter
    def depth(self, depth: float):
        """Set grasp depth.

        Args:
            depth (float): Depth of the grasp.
        """
        self.mf_garray[1] = copy.deepcopy(depth)

    @property
    def grasp_type(self) -> int:
        """Grasp type.

        Returns:
            int or float: Type of the grasp.
        """
        return self.mf_garray[2]

    @grasp_type.setter
    def grasp_type(self, grasp_type: int):
        """Set grasp type.

        Args:
            grasp_type (int or float): Type of the grasp.
        """
        self.mf_garray[2] = copy.deepcopy(grasp_type)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Rotation matrix of the grasp pose.

        Returns:
            np.ndarray: Rotation matrix of shape (3, 3).
        """
        return self.mf_garray[3:12].reshape((3, 3))

    @rotation_matrix.setter
    def rotation_matrix(self, *args):
        """Set rotation matrix of the grasp pose.

        Args:
            If len(args) == 1: array-like of shape (3, 3) or (9,).
            If len(args) == 9: nine float values of the matrix.
        """
        if len(args) == 1:
            self.mf_garray[3:12] = np.array(args[0], dtype=np.float64).reshape(9)
        elif len(args) == 9:
            self.mf_garray[3:12] = np.array(args, dtype=np.float64)

    @property
    def translation(self) -> np.ndarray:
        """Translation of the grasp pose.

        Returns:
            np.ndarray: Translation vector of shape (3,).
        """
        return self.mf_garray[12:15]

    @translation.setter
    def translation(self, *args):
        """Set translation of the grasp pose.

        Args:
            If len(args) == 1: array-like (x, y, z).
            If len(args) == 3: three float values x, y, z.
        """
        if len(args) == 1:
            self.mf_garray[12:15] = np.array(args[0], dtype=np.float64)
        elif len(args) == 3:
            self.mf_garray[12:15] = np.array(args, dtype=np.float64)

    @property
    def angle(self) -> np.ndarray:
        """Joint angles of the hand.

        Returns:
            np.ndarray: Joint angles of shape (6,).
        """
        return self.mf_garray[15:21]

    @angle.setter
    def angle(self, *args):
        """Set joint angles of the hand.

        Args:
            If len(args) == 1: array-like of 6 angles.
            If len(args) == 6: six float angle values.
        """
        if len(args) == 1:
            self.mf_garray[15:21] = np.array(args[0], dtype=np.float64)
        elif len(args) == 6:
            self.mf_garray[15:21] = np.array(args, dtype=np.float64)

    @property
    def object_id(self) -> int:
        """Object id associated with this grasp.

        Returns:
            float: Id of the object.
        """
        return float(self.mf_garray[21])

    @object_id.setter
    def object_id(self, object_id: int):
        """Set object id.

        Args:
            object_id (int or float): Id of the object.
        """
        self.mf_garray[21] = object_id

    @property
    def width(self) -> float:
        """Grasp width. Width is in meters.

        Returns:
            float: Width of the grasp.
        """
        return float(self.mf_garray[22])

    @width.setter
    def width(self, width: float):
        """Set grasp width (clipped to valid range).
        Width is in meters.

        Args:
            width (float): Width of the grasp.
        """
        min_width, max_width = self.mf_gtax[str(self.grasp_type)]["width"]
        width = min(min(max_width, MAX_GRASP_WIDTH), max(max(min_width, MIN_GRASP_WIDTH), width))
        self.mf_garray[22] = width

    def get_array_grasp(self) -> np.ndarray:
        """Get grasp as numpy array.

        Returns:
            np.ndarray: Grasp represented as a numpy array.
        """
        return self.mf_garray

    def get_grasp_type_with_finger_name(self) -> str:
        """Get grasp type with finger name.

        Returns:
            str: Grasp type with finger name.
        """
        return self.mf_gtax[str(int(self.mf_garray[2]))]["name"]

    def _tfTR_2_mfTR(self, two_finger_grasp: Grasp):
        """Convert two-finger grasp transform to inspire grasp transform.

        Args:
            two_finger_grasp (Grasp): Two-finger grasp object.
        """
        trans_tf2world = two_finger_grasp.translation
        rot_tf2world = two_finger_grasp.rotation_matrix

        grasp_taxonomy_info_path = self.__class__.mf_model.GRASP_TAXONOMY_INFOS_PATH

        with open(grasp_taxonomy_info_path, "r", encoding="UTF-8") as f:
            grasp_taxonomy_info = json.load(f)

        # transformation matrix from two-finger grasp to world
        M_tf2world = np.vstack(
            (
                np.hstack(
                    (
                        np.array(rot_tf2world).reshape((3, 3)),
                        np.array(trans_tf2world).reshape((3, 1)),
                    )
                ),
                np.array((0, 0, 0, 1)),
            )
        )

        # translation and rotation from two-finger grasp to multi-finger hand base
        trans_tf2mfbase = grasp_taxonomy_info[self.get_grasp_type_with_finger_name()][
            str(np.round(self.width * 100, 1))
        ]["primary_closure_pose_translation"]
        rot_tf2mfbase = grasp_taxonomy_info[self.get_grasp_type_with_finger_name()][
            str(np.round(self.width * 100, 1))
        ]["primary_closure_pose_rotation"]

        # transformation matrix from two-finger grasp to multi-finger hand base
        M_tf2mfbase = np.vstack(
            (
                np.hstack(
                    (
                        np.array(rot_tf2mfbase).reshape((3, 3)),
                        np.array(trans_tf2mfbase).reshape((3, 1)),
                    )
                ),
                np.array((0, 0, 0, 1)),
            )
        )

        # transformation matrix from multi-finger hand base to world
        M_mfbase2world = np.dot(M_tf2world, np.linalg.inv(M_tf2mfbase))

        # set rotation matrix, translation and joint angles
        self.rotation_matrix = np.array(
            [M_mfbase2world[0, 0:3], M_mfbase2world[1, 0:3], M_mfbase2world[2, 0:3]]
        )
        self.translation = np.array(
            [M_mfbase2world[0, 3], M_mfbase2world[1, 3], M_mfbase2world[2, 3]]
        )
        self.angle = grasp_taxonomy_info[self.get_grasp_type_with_finger_name()][
            str(np.round(self.width * 100, 1))
        ]["hand_angles"]

    def from_grasp(self, two_finger_grasp: Grasp, grasp_type: int):
        """Grasp to InspireHandGrasp Transformation."""
        self.score = two_finger_grasp.score
        self.depth = two_finger_grasp.depth
        self.grasp_type = grasp_type
        self.width = two_finger_grasp.width
        self._tfTR_2_mfTR(two_finger_grasp)
        self.object_id = two_finger_grasp.object_id

    def to_open3d_geometry(self):
        """Convert the InspireHandGrasp to an Open3D geometry for visualization.

        Returns:
            open3d.geometry.Geometry: The Open3D geometry of the Inspire hand in the grasp pose.
        """
        self.__class__.mf_model.update_cfg(
            joints=self.angle,
            translation=self.translation,
            rotation=self.rotation_matrix.reshape(9, -1),
        )
        return self.__class__.mf_model.open3d_mesh


@MULTIFINGERGRASPGROUP.register()
class InspireHandGraspGroup(MultiFingerGraspGroupBase):
    def __init__(self, *args):
        """Initialize InspireHandGraspGroup object.
        Args:
            args:
                - No argument: initializes with default values.
                - One argument (np.ndarray): initializes with the given grasp group array.

        The format of numpy array is [score, depth, grasp_type, rot_matrix(9), translation(3), joint_angles(6), obj_id, width]
        """
        super().__init__(self, *args)
        if len(args) == 0:
            self.mf_ggarray = np.array(
                [0, 0, 1]  # score, depth, grasp_type
                + [1, 0, 0, 0, 1, 0, 0, 0, 1]  # rotation matrix
                + [0] * 3  # translation
                + [0] * 6  # joint angles
                + [-1, 0],  # obj_id, width
                dtype=np.float64,
            ).reshape(1, INSPIRE_ARRAY_LEN)
        elif len(args) == 1:
            if isinstance(args[0], np.ndarray):
                self.mf_ggarray = copy.deepcopy(args[0])
            elif isinstance(args[0], list):
                self.mf_ggarray = np.array(args[0], dtype=np.float64)
            else:
                raise ValueError("Argument must be a numpy array or a list.")
            # check the shape
            if self.mf_ggarray.ndim != 2 or self.mf_ggarray.shape[1] != INSPIRE_ARRAY_LEN:
                raise ValueError(
                    "Grasp group array must have shape of (-1, {}).".format(INSPIRE_ARRAY_LEN)
                )
        else:
            raise ValueError("Only zero or one argument is allowed.")

        self.mf_gtax = InspireGraspTaxonomy

    def __len__(self):
        """Get the number of grasps in the group.

        Returns:
            int: Number of grasps in the group.
        """
        return super().__len__()

    def __repr__(self):
        """Print the grasp group information."""
        repr = "----------\nInspire Grasp Group, Number={}:\n".format(self.__len__())
        if self.__len__() <= 6:
            for mf_garray in self.mf_ggarray:
                repr += InspireHandGrasp(mf_garray).__repr__() + "\n"
        else:
            for i in range(3):
                repr += InspireHandGrasp(self.mf_ggarray[i]).__repr__() + "\n"
            repr += "......\n"
            for i in range(3):
                repr += InspireHandGrasp(self.mf_ggarray[-(3 - i)]).__repr__() + "\n"
        return repr + "----------"

    def __getitem__(self, index: Union[int, slice, np.ndarray, list]):
        """Get InspireHandGrasp or InspireHandGraspGroup by index.

        Args:
            index(int, slice, np.ndarray, list): the index to get the InspireHandGrasp or InspireHandGraspGroup.
        Returns:
            InspireHandGrasp or InspireHandGraspGroup: the grasp or grasp group at the given index.
        """

        if type(index) == int:
            return InspireHandGrasp(self.mf_ggarray[index])
        elif type(index) == slice:
            mf_graspgroup = InspireHandGraspGroup()
            mf_graspgroup.mf_ggarray = copy.deepcopy(self.mf_ggarray[index])
            return mf_graspgroup
        elif type(index) == np.ndarray:
            return InspireHandGraspGroup(self.mf_ggarray[index])
        elif type(index) == list:
            return InspireHandGraspGroup(self.mf_ggarray[index])
        else:
            raise TypeError(
                'unknown type "{}" for calling __getitem__ for InspireHandGraspGroup'.format(
                    type(index)
                )
            )

    @property
    def scores(self) -> np.ndarray:
        """Get scores of the grasps in the group.

        Returns:
            numpy array of shape (-1, ) of the scores.
        """
        return self.mf_ggarray[:, 0]

    @scores.setter
    def scores(self, scores: Union[np.ndarray, list]):
        """Set scores of the grasps in the group.

        Args:
            scores(numpy array or list of shape (-1, )): the scores.
        """
        if isinstance(scores, list):
            scores = np.array(scores)
        from cprint import cprint

        assert scores.size == len(self)
        self.mf_ggarray[:, 0] = copy.deepcopy(scores)

    @property
    def depths(self) -> np.ndarray:
        """Get depths of the grasps in the group.

        Returns:
            numpy array of shape (-1, ) of the depths.
        """
        return self.mf_ggarray[:, 1]

    @depths.setter
    def depths(self, depths: Union[np.ndarray, list]):
        """Set depths of the grasps in the group.

        Args:
            depths(numpy array or list of shape (-1, )): the depths.
        """
        if isinstance(depths, list):
            depths = np.array(depths)
        assert depths.size == len(self)
        self.mf_ggarray[:, 1] = copy.deepcopy(depths)

    @property
    def grasp_types(self) -> np.ndarray:
        """Get grasp types of the grasps in the group.

        Returns:
            numpy array of shape (-1, ) of the grasp types.
        """
        return self.mf_ggarray[:, 2]

    @grasp_types.setter
    def grasp_types(self, grasp_types: Union[np.ndarray, list]):
        """Set grasp types of the grasps in the group.

        Args:
            grasp_types(numpy array or list of shape (-1, )): the grasp types.
        """
        if isinstance(grasp_types, list):
            grasp_types = np.array(grasp_types)
        assert grasp_types.size == len(self)
        self.mf_ggarray[:, 2] = copy.deepcopy(grasp_types)

    @property
    def rotation_matrices(self) -> np.ndarray:
        """Get rotation_matrices of the grasps in the group.

        Returns:
            np.array of shape (-1, 3, 3) of the rotation_matrices
        """
        return self.mf_ggarray[:, 3:12].reshape((-1, 3, 3))

    @rotation_matrices.setter
    def rotation_matrices(self, rotation_matrices: Union[np.ndarray, list]):
        """Set rotation_matrices of the grasps in the group.

        Args:
            rotation_matrices(numpy array or list of shape (-1, 3, 3)): the rotation_matrices.
        """
        if isinstance(rotation_matrices, list):
            rotation_matrices = np.array(rotation_matrices)
        assert rotation_matrices.shape == (len(self), 3, 3)
        self.mf_ggarray[:, 3:12] = copy.deepcopy(rotation_matrices.reshape((-1, 9)))

    @property
    def translations(self) -> np.ndarray:
        """Get translations of the grasps in the group.

        Returns:
            np.array of shape (-1, 3) of the translations.
        """
        return self.mf_ggarray[:, 12:15]

    @translations.setter
    def translations(self, translations: Union[np.ndarray, list]):
        """Set translations of the grasps in the group.

        Args:
            translations(numpy array or list of shape (-1, 3)): the translations.
        """
        if isinstance(translations, list):
            translations = np.array(translations)
        assert translations.shape == (len(self), 3)
        self.mf_ggarray[:, 12:15] = copy.deepcopy(translations)

    @property
    def angles(self) -> np.ndarray:
        """Get joint angles of the grasps in the group.

        Returns:
            numpy array of shape (-1, 6) of the angles.
        """
        return self.mf_ggarray[:, 15:21]

    @angles.setter
    def angles(self, angles: Union[np.ndarray, list]):
        """Set joint angles of the grasps in the group.

        Args:
            angles(numpy array or list of shape (-1, 6)): the angles.
        """
        if isinstance(angles, list):
            angles = np.array(angles)
        assert angles.shape == (len(self), 6)
        self.mf_ggarray[:, 15:21] = copy.deepcopy(angles)

    @property
    def object_ids(self) -> np.ndarray:
        """Get object_ids of the grasps in the group.

        Returns:
            numpy array of shape (-1, ) of the object_ids.
        """
        return self.mf_ggarray[:, 21]

    @object_ids.setter
    def object_ids(self, object_ids: Union[np.ndarray, list]):
        """Set object_ids of the grasps in the group.

        Args:
            object_ids(numpy array or list of shape (-1, )): the object_ids.
        """
        if isinstance(object_ids, list):
            object_ids = np.array(object_ids)
        assert object_ids.size == len(self)
        self.mf_ggarray[:, 21] = copy.deepcopy(object_ids)

    @property
    def widths(self) -> np.ndarray:
        """Get widths of the grasps in the group.

        Returns:
            numpy array of shape (-1, ) of the widths.
        """
        return self.mf_ggarray[:, 22]

    @widths.setter
    def widths(self, widths: Union[np.ndarray, list]):
        """Set widths of the grasps in the group.

        Args:
            widths(numpy array or list of shape (-1, )): the widths.
        """
        if isinstance(widths, list):
            widths = np.array(widths)
        assert widths.size == len(self)
        for idx, _ in enumerate(widths):
            min_width, max_width = self.mf_gtax[str(int(self.grasp_types[idx]))]["width"]
            widths[idx] = min(
                min(max_width, MAX_GRASP_WIDTH), max(max(min_width, MIN_GRASP_WIDTH), widths[idx])
            )
        self.mf_ggarray[:, 22] = copy.deepcopy(widths)

    def get_graspgroup_types_with_finger_names(self):
        """Get graspgroup types with finger names. Converts the grasp type ids to finger names.

        Returns:
            list of str: the graspgroup types with finger names.
        """
        graspgroup_types_with_finger_names = []
        for finger_type in self.mf_ggarray[:, 2]:
            graspgroup_types_with_finger_names.append(self.mf_gtax[str(int(finger_type))]["name"])
        return graspgroup_types_with_finger_names

    def _tfGroupTR_2_mfGroupTR(self, tf_graspgroup: GraspGroup):
        """Convert two-finger grasp group transform to multi-finger grasp group transform.

        Args:
            tf_graspgroup (graspnetAPI.GraspGroup): Two-finger grasp group object.
        """
        tf_gg = tf_graspgroup

        grasp_taxonomy_info_path = InspireHandGrasp.mf_model.GRASP_TAXONOMY_INFOS_PATH

        with open(grasp_taxonomy_info_path, "r", encoding="UTF-8") as f:
            grasp_taxonomy_info = json.load(f)

        inspire_trans, inspire_rots = [], []
        inspire_angles, inspire_widths = [], []

        for idx, width in enumerate(tf_gg.widths):
            min_width, max_width = self.mf_gtax[str(int(self.grasp_types[idx]))]["width"]
            width = min(
                min(max_width, MAX_GRASP_WIDTH), max(max(min_width, MIN_GRASP_WIDTH), width)
            )
            inspire_widths.append(width)

            # translation and rotation from two-finger grasp to world
            trans_tf2world = tf_gg.translations[idx]
            rot_tf2world = tf_gg.rotation_matrices[idx]

            # transformation matrix from two-finger grasp to world
            M_tf2world = np.vstack(
                (
                    np.hstack(
                        (
                            np.array(rot_tf2world).reshape((3, 3)),
                            np.array(trans_tf2world).reshape((3, 1)),
                        )
                    ),
                    np.array((0, 0, 0, 1)),
                )
            )

            # translation and rotation from two-finger grasp to multi-finger hand base
            trans_tf2mfbase = grasp_taxonomy_info[
                self.get_graspgroup_types_with_finger_names()[idx]
            ][str(np.round(width * 100, 1))]["primary_closure_pose_translation"]
            rot_tf2mfbase = grasp_taxonomy_info[self.get_graspgroup_types_with_finger_names()[idx]][
                str(np.round(width * 100, 1))
            ]["primary_closure_pose_rotation"]

            # transformation matrix from two-finger grasp to multi-finger hand base
            M_tf2mfbase = np.vstack(
                (
                    np.hstack(
                        (
                            np.array(rot_tf2mfbase).reshape((3, 3)),
                            np.array(trans_tf2mfbase).reshape((3, 1)),
                        )
                    ),
                    np.array((0, 0, 0, 1)),
                )
            )

            # transformation matrix from multi-finger hand base to world
            M_mfbase2world = np.dot(M_tf2world, np.linalg.inv(M_tf2mfbase))

            # set rotation matrix, translation and joint angles
            inspire_rots.append(
                np.array([M_mfbase2world[0, 0:3], M_mfbase2world[1, 0:3], M_mfbase2world[2, 0:3]])
            )
            inspire_trans.append(
                np.array([M_mfbase2world[0, 3], M_mfbase2world[1, 3], M_mfbase2world[2, 3]])
            )

            angle = grasp_taxonomy_info[self.get_graspgroup_types_with_finger_names()[idx]][
                str(np.round(width * 100, 1))
            ]["hand_angles"]
            inspire_angles.append(np.array(angle).reshape(6))

        self.translations = np.array(inspire_trans)
        self.rotation_matrices = np.array(inspire_rots)
        self.angles = np.array(inspire_angles)

    def from_graspgroup(
        self,
        graspgroup: GraspGroup,
        grasp_types: Union[np.ndarray, list],
    ) -> "InspireHandGraspGroup":
        """Convert GraspGroup to MultiFingerGraspGroup.

        Args:
            graspgroup (graspnetAPI.GraspGroup): Two-finger grasp group object.
            grasp_types (numpy array or list of shape (-1, )): the grasp types for each grasp in the group.

        Returns:
            InspireHandGraspGroup: The converted InspireHandGraspGroup object.
        """
        self.mf_ggarray = np.zeros((len(graspgroup), INSPIRE_ARRAY_LEN), dtype=np.float64)
        self.scores = graspgroup.scores
        self.depths = graspgroup.depths
        self.grasp_types = grasp_types
        self.widths = graspgroup.widths
        self._tfGroupTR_2_mfGroupTR(graspgroup)
        self.object_ids = graspgroup.object_ids
        return self

    def sort_by_score(self, reverse: bool = False):
        """Sort the grasp group by score.

        Args:
            reverse (bool): If True, sort in ascending order. Default is False (descending order).

        Returns:
            InspireHandGraspGroup: The sorted grasp group.
        """
        score = self.mf_ggarray[:, 0]
        index = np.argsort(score)
        if not reverse:
            index = index[::-1]
        self.update_by_id(index)
        return self

    def update_by_id(self, index: Union[list, np.ndarray, slice]):
        """Update the grasp group by index.

        Args:
            index (list, np.ndarray, slice): The indices to update the grasp group.
        """
        self.mf_ggarray = self.mf_ggarray[index]
