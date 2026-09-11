import os
import sys
import json
import xlrd2
import numpy as np
import open3d as o3d
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robots.hands.grasp_taxonomy import DH3GraspTaxonomy
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Union, Tuple
from robots.hands.hand_base import HandBase, HAND, create_hand_model
from graspnetAPI import Grasp
from utils.angles2stl import DH3Angles2STLs
from utils.path import ROOT_DIR
from cprint import cprint

DIR_PATH = os.path.dirname(os.path.abspath(__file__))


@HAND.register()
class DH3Hand(HandBase):
    DOF = 2
    GRASP_TYPES = DH3GraspTaxonomy
    URDF_PATH = os.path.join(ROOT_DIR, f"data/urdf/dh3_hand/dh3_hand_glb_sphere.urdf")
    POINTCLOUD_CACHE = os.path.join(ROOT_DIR, f"data/pointcloud/dh3_hand/")
    GRASP_TAXONOMY_CONFIG_PATH = os.path.join(DIR_PATH, "./dh3_width_to_angle.xls")
    GRASP_TAXONOMY_INFOS_PATH = os.path.join(DIR_PATH, "./dh3_taxonomy_all_info.json")

    def __init__(
        self,
        urdf_path: str = URDF_PATH,
        use_cache: bool = False,
    ):
        """Load the DH3Hand right-hand URDF and its cached or computed taxonomy."""
        super().__init__(urdf_path=urdf_path, use_cache=use_cache)
        self.hand_name = "dh3_hand"

    def _init_internal_(self, use_cache: bool = False) -> None:
        """Initialize internal attributes specific to the DH3 hand model."""
        if use_cache and self._init_from_cache_():
            return

        self.grasp_taxonomy = self._load_grasp_taxonomy()

        self.grasp_all_info = defaultdict(dict)
        for type_id, type_info in enumerate(self.grasp_taxonomy):
            for ids in range(len(type_info)):
                hand_joints = type_info[ids][:-1]
                grasp_width = np.round(type_info[ids][-1], 1)
                type_name = DH3Hand.GRASP_TYPES[str(type_id + 1)]["name"]
                translation_2d, rotation_2d = self._get_predefined_pose(type_id + 1, grasp_width)
                translation_2d, rotation_2d = translation_2d.tolist(), rotation_2d.tolist()

                self.grasp_all_info[type_name][str(grasp_width)] = {
                    "hand_angles": hand_joints,
                    "primary_closure_pose_translation": translation_2d,
                    "primary_closure_pose_rotation": rotation_2d,
                }

        if use_cache:
            json_str = json.dumps(self.grasp_all_info, indent=4)
            cach_file = DH3Hand.GRASP_TAXONOMY_INFOS_PATH
            with open(cach_file, "w") as json_file:
                json_file.write(json_str)

    def _load_grasp_taxonomy(self) -> List:
        """Load the grasp taxonomy from an Excel file into a nested list."""
        grasp_taxonomy_path = os.path.normpath(
            os.path.join(DIR_PATH, DH3Hand.GRASP_TAXONOMY_CONFIG_PATH)
        )
        grasp_taxonomy = [[] for _ in range(len(DH3Hand.GRASP_TYPES))]

        wb = xlrd2.open_workbook(grasp_taxonomy_path)
        for type_id in range(len(DH3Hand.GRASP_TYPES)):
            sheet = wb.sheet_by_index(type_id)

            widths = sheet.col_values(0)[1:]
            widths = np.array(widths) * 0.1
            widths = np.round(widths, 1)

            rotation = sheet.col_values(1)[1:]
            position = sheet.col_values(2)[1:]

            for ids, _ in enumerate(widths):
                if rotation[ids] == -1:
                    continue
                grasp_taxonomy[type_id].append(
                    [
                        rotation[ids],
                        position[ids],
                        widths[ids],
                    ]
                )

        return grasp_taxonomy

    def _init_from_cache_(self) -> bool:
        """Initialize the DH3 hand model from cached data if available.
        The cached data structure is expected to be:
        {
            'grasp_type': {
                'grasp_width': {
                    'hand_angles': [...],
                    'primary_closure_pose_translation': [...],
                    'primary_closure_pose_rotation': [...],
            },
            ...
        }
        """
        cache_file = DH3Hand.GRASP_TAXONOMY_INFOS_PATH
        if not os.path.exists(cache_file):
            return False

        cprint.info("Loading DH3 hand grasp taxonomy from cache...")
        with open(cache_file, "r", encoding="utf-8") as f:
            self.grasp_all_info = json.load(f)  # grasp taxonomy
        return True

    def _get_predefined_joints(
        self,
        grasp_type: Union[int, str],
        grasp_width: float,
    ) -> np.array:
        """Retrieve a predefined joint configuration for a specified DH3 hand grasp.

        Args:
            grasp_type (int or str):
                Grasp type identifier (1-10 or key).
            grasp_width (float):
                Desired grasp width in cm (rounded to 0.1 cm).

        Returns:
            np.array: Joint configuration array if found, otherwise None.
        """
        # self._validate_grasp_type_and_width(grasp_type, grasp_width)

        cprint.info(
            f"Requested grasp_type: {DH3Hand.GRASP_TYPES[str(grasp_type)]['name']}, grasp_width: {grasp_width} cm."
        )

        # retrieve the joint configuration from the taxonomy
        type_info = self.grasp_taxonomy[int(grasp_type) - 1]
        for ids in range(len(type_info)):
            hand_joints = type_info[ids][:-1]
            width = np.round(type_info[ids][-1], 1)
            if np.isclose(grasp_width, width, atol=1e-3):
                break

        return np.array(hand_joints)

    def _get_predefined_pose(
        self,
        grasp_type: Union[int, str],
        grasp_width: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve the predefined grasp pose (translation and rotation) for a specified DH3 hand grasp.
        Args:
            grasp_type (int or str):
                Grasp type identifier (1-10 or key).
            grasp_width (float):
                Desired grasp width in cm (rounded to 0.1 cm).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                A tuple containing the translation vector and rotation matrix of the grasp pose.
        """
        hand_joints = self._get_predefined_joints(grasp_type, grasp_width)
        self.update_cfg(joints=hand_joints)

        dh3_tool = DH3Angles2STLs(DH3Hand.GRASP_TYPES)

        translation_2d, rotation_2d = dh3_tool.get_pose_information(
            self.open3d_mesh,
            grasp_type - 1,
        )

        _, _, zero_joints = self.zero_cfg
        self.update_cfg(joints=zero_joints)

        return translation_2d, rotation_2d

    def _validate_grasp_type_and_width(
        self, grasp_type: Union[int, str], grasp_width: float
    ) -> None:
        """Validate whether the given grasp type and grasp width fall within
        the allowed range defined in `DH3Hand.GRASP_TYPES`.

        This function performs the following checks:
        1. Ensures the provided `grasp_type` (int or str) corresponds to a valid
        key in the grasp taxonomy.
        2. Ensures the provided `grasp_width`, after rounding to one decimal place,
        falls within the allowed width range for this grasp type.

        Args:
            grasp_type (int | str):
                The grasp taxonomy ID. Can be an integer (e.g., 1) or a string (e.g., "1").
                Internally converted to string for dictionary lookup.

            grasp_width (float):
                The target grasp width **in centimeters**. It is rounded to one decimal
                place and converted to meters (value * 0.01) before validation.

        Raises:
            ValueError:
                - If `grasp_type` does not exist in the taxonomy.
                - If the (rounded and converted) grasp width is outside the valid range.
        """

        # Normalize grasp_type to string
        grasp_type_key = str(grasp_type)

        # Check if the grasp type exists
        if grasp_type_key not in DH3Hand.GRASP_TYPES:
            raise ValueError(
                f"Invalid grasp_type '{grasp_type}'. Must be one of: {list(DH3Hand.GRASP_TYPES.keys())}"
            )

        # Extract allowed width range (min_width, max_width in meters)
        min_width, max_width = DH3Hand.GRASP_TYPES[grasp_type_key]["width"]

        # Round grasp_width (in centimeters) then convert to meters
        grasp_width = np.round(grasp_width, 1)
        grasp_width_m = grasp_width * 0.01

        # Validate width
        if grasp_width_m < min_width - 1e-6 or grasp_width_m > max_width + 1e-6:
            raise ValueError(
                f"grasp_width {grasp_width_m:.3f} m is out of range "
                f"[{min_width}, {max_width}] for grasp_type {grasp_type_key}."
            )

    def get_predefined_joints(
        self,
        grasp_type: Union[int, str],
        grasp_width: float,
    ) -> np.array:
        """Retrieve a predefined joint configuration for a specified DH3 hand grasp.

        Args:
            grasp_type (int or str):
                Grasp type identifier (1-10 or key).
            grasp_width (float):
                Desired grasp width in cm (rounded to 0.1 cm).

        Returns:
            np.array: Joint configuration array if found, otherwise None.
        """
        self._validate_grasp_type_and_width(grasp_type, grasp_width)

        hand_joints = self.grasp_all_info[DH3Hand.GRASP_TYPES[str(grasp_type)]["name"]][
            f"{grasp_width:.1f}"
        ]["hand_angles"]
        return np.array(hand_joints)

    def get_predefined_pose(
        self,
        grasp_type: Union[int, str],
        grasp_width: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the predefined 2-finger grasp pose (translation and rotation) for a specified DH3 hand grasp.
        Args:
            grasp_type (int or str):
                Grasp type identifier (1-10 or key).
            grasp_width (float):
                Desired grasp width in cm (rounded to 0.1 cm).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                A tuple containing the translation vector and rotation matrix of the grasp pose.
        """
        self._validate_grasp_type_and_width(grasp_type, grasp_width)

        grasp_info = self.grasp_all_info[DH3Hand.GRASP_TYPES[str(grasp_type)]["name"]][
            f"{grasp_width:.1f}"
        ]

        translation_2d = grasp_info["primary_closure_pose_translation"]
        rotation_2d = grasp_info["primary_closure_pose_rotation"]

        return (np.array(translation_2d), np.array(rotation_2d))

    def vis_predefined_grasp_type(
        self,
        grasp_type: Union[int, str],
        grasp_width: float,
        is_visualize: bool = True,
        is_export: bool = False,
        export_path: Optional[str] = None,
        export_format: str = "obj",
    ) -> None:
        """Visualize or export the predefined configuration and pose of the DH3 hand for a specified grasp.

        Args:
            grasp_type (int or str):
                Grasp type identifier (1-10 or key).
            grasp_width (float):
                Desired grasp width in cm (rounded to 0.1 cm).
            is_visualize (bool):
                If True, visualize the hand and grasp pose using Open3D.
            is_export (bool):
                If True, export the hand and grasp pose mesh to the specified path.
            export_path (str, optional):
                Directory path to save the exported mesh. Required if is_export is True.
            export_format (str):
                Format of the exported mesh file (e.g., "obj", "ply", "glb").
        """
        hand_joints = self.get_predefined_joints(grasp_type, grasp_width)
        self.update_cfg(joints=hand_joints)

        translation_2d, rotation_2d = self.get_predefined_pose(grasp_type, grasp_width)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
        grasp_matrix = np.vstack(
            (np.hstack((rotation_2d, translation_2d.reshape((3, 1)))), np.array((0, 0, 0, 1)))
        )
        frame_grasp = o3d.geometry.TriangleMesh.create_coordinate_frame(0.03).transform(
            grasp_matrix
        )
        g = Grasp().transform(grasp_matrix)
        g.width = grasp_width * 0.01

        if is_visualize:
            o3d.visualization.draw_geometries(
                [self.open3d_mesh, frame, frame_grasp, g.to_open3d_geometry()],
                mesh_show_back_face=True,
            )

        if is_export and export_path is not None:
            export_mesh = o3d.geometry.TriangleMesh()
            for mesh in [self.open3d_mesh, frame, frame_grasp, g.to_open3d_geometry()]:
                if not isinstance(mesh, o3d.geometry.TriangleMesh):
                    continue
                export_mesh += mesh

            export_mesh.compute_vertex_normals()
            Path(export_path).mkdir(parents=True, exist_ok=True)
            o3d.io.write_triangle_mesh(
                os.path.join(
                    export_path,
                    f"dh3_{DH3Hand.GRASP_TYPES[str(grasp_type)]['name']}_{grasp_width: .1f}.{export_format}",
                ),
                export_mesh,
                write_vertex_colors=True,
            )

        _, _, zero_joionts = self.zero_cfg
        self.update_cfg(joints=zero_joionts)


if __name__ == "__main__":
    hand = create_hand_model("DH3Hand", use_cache=True)

    # # exmaple 1: visualize the hand predefined configuration
    # joints = hand.get_predefined_joints(1, 8)
    # hand.update_cfg(joints=joints)
    # hand.vis_joints()

    # exmaple2: visualize the hand predefined configuration
    hand.vis_predefined_grasp_type(1, 4)
