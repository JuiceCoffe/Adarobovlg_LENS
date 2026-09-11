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

from robots.hands.grasp_taxonomy import AllegroGraspTaxonomy
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Union, Tuple
from robots.hands.hand_base import HandBase, HAND, create_hand_model
from graspnetAPI import Grasp
from utils.angles2stl import AllegroAngles2STLs
from utils.path import ROOT_DIR
from cprint import cprint

DIR_PATH = os.path.dirname(os.path.abspath(__file__))


@HAND.register()
class AllegroHand(HandBase):
    DOF = 16
    HAND_TYPE = "right"
    GRASP_TYPES = AllegroGraspTaxonomy
    URDF_PATH = os.path.join(
        ROOT_DIR, f"data/urdf/allegro_hand/allegro_hand_{HAND_TYPE}_glb_sphere.urdf"
    )
    POINTCLOUD_CACHE = os.path.join(ROOT_DIR, f"data/pointcloud/allegro_hand/")
    GRASP_TAXONOMY_CONFIG_PATH = os.path.join(DIR_PATH, "./allegro_width_to_angle.xls")
    GRASP_TAXONOMY_INFOS_PATH = os.path.join(DIR_PATH, "./allegro_taxonomy_all_info.json")

    def __init__(
        self,
        urdf_path: str = URDF_PATH,
        use_cache: bool = False,
    ):
        """Load the AllegroHand right-hand URDF and its cached or computed taxonomy."""
        super().__init__(urdf_path=urdf_path, use_cache=use_cache)
        self.hand_name = "allegro_hand"

    def _init_internal_(self, use_cache: bool = False) -> None:
        """Initialize internal attributes specific to the Allegro hand model."""
        if use_cache and self._init_from_cache_():
            return

        self.grasp_taxonomy = self._load_grasp_taxonomy()

        self.grasp_all_info = defaultdict(dict)
        for type_id, type_info in enumerate(self.grasp_taxonomy):
            for ids in range(len(type_info)):
                hand_joints = type_info[ids][:-1]
                grasp_width = np.round(type_info[ids][-1], 1)
                type_name = AllegroHand.GRASP_TYPES[str(type_id + 1)]["name"]
                translation_2d, rotation_2d = self._get_predefined_pose(type_id + 1, grasp_width)
                translation_2d, rotation_2d = translation_2d.tolist(), rotation_2d.tolist()

                self.grasp_all_info[type_name][str(grasp_width)] = {
                    "hand_angles": hand_joints,
                    "primary_closure_pose_translation": translation_2d,
                    "primary_closure_pose_rotation": rotation_2d,
                }

        if use_cache:
            json_str = json.dumps(self.grasp_all_info, indent=4)
            cach_file = AllegroHand.GRASP_TAXONOMY_INFOS_PATH
            with open(cach_file, "w") as json_file:
                json_file.write(json_str)

    def _load_grasp_taxonomy(self) -> List:
        """Load the grasp taxonomy from an Excel file into a nested list."""
        grasp_taxonomy_path = os.path.normpath(
            os.path.join(DIR_PATH, AllegroHand.GRASP_TAXONOMY_CONFIG_PATH)
        )
        grasp_taxonomy = [[] for _ in range(len(AllegroHand.GRASP_TYPES))]

        wb = xlrd2.open_workbook(grasp_taxonomy_path)
        for type_id in range(len(AllegroHand.GRASP_TYPES)):
            sheet = wb.sheet_by_index(type_id)

            widths = sheet.col_values(0)[1:]
            widths = np.array(widths) * 0.1
            widths = np.round(widths, 1)

            thumb0 = sheet.col_values(1)[1:]
            thumb1 = sheet.col_values(2)[1:]
            thumb2 = sheet.col_values(3)[1:]
            thumb3 = sheet.col_values(4)[1:]

            index0 = sheet.col_values(5)[1:]
            index1 = sheet.col_values(6)[1:]
            index2 = sheet.col_values(7)[1:]
            index3 = sheet.col_values(8)[1:]

            middle0 = sheet.col_values(9)[1:]
            middle1 = sheet.col_values(10)[1:]
            middle2 = sheet.col_values(11)[1:]
            middle3 = sheet.col_values(12)[1:]

            ring0 = sheet.col_values(13)[1:]
            ring1 = sheet.col_values(14)[1:]
            ring2 = sheet.col_values(15)[1:]
            ring3 = sheet.col_values(16)[1:]

            for ids, _ in enumerate(widths):
                if thumb0[ids] == -1:
                    continue
                grasp_taxonomy[type_id].append(
                    [
                        index0[ids],
                        index1[ids],
                        index2[ids],
                        index3[ids],
                        middle0[ids],
                        middle1[ids],
                        middle2[ids],
                        middle3[ids],
                        ring0[ids],
                        ring1[ids],
                        ring2[ids],
                        ring3[ids],
                        thumb0[ids],
                        thumb1[ids],
                        thumb2[ids],
                        thumb3[ids],
                        widths[ids],
                    ]
                )

        return grasp_taxonomy

    def _init_from_cache_(self) -> bool:
        """Initialize the Allegro hand model from cached data if available.
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
        cache_file = AllegroHand.GRASP_TAXONOMY_INFOS_PATH
        if not os.path.exists(cache_file):
            return False

        cprint.info("Loading Allegro hand grasp taxonomy from cache...")
        with open(cache_file, "r", encoding="utf-8") as f:
            self.grasp_all_info = json.load(f)  # grasp taxonomy
        return True

    def _get_predefined_joints(
        self,
        grasp_type: Union[int, str],
        grasp_width: float,
    ) -> np.array:
        """Retrieve a predefined joint configuration for a specified Allegro hand grasp.

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
            f"Requested grasp_type: {AllegroHand.GRASP_TYPES[str(grasp_type)]['name']}, grasp_width: {grasp_width} cm."
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
        """Retrieve the predefined grasp pose (translation and rotation) for a specified Allegro hand grasp.
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

        allegro_tool = AllegroAngles2STLs(AllegroHand.GRASP_TYPES)

        facenet_thumb = AllegroHand.GRASP_TYPES[str(grasp_type)]["facenet_thumb"]
        facenet_index = AllegroHand.GRASP_TYPES[str(grasp_type)]["facenet_index"]

        if len(facenet_thumb) != 0:
            translation_2d, rotation_2d = allegro_tool.common_grasp_information(
                self.open3d_mesh, facenet_thumb, facenet_index, grasp_width, grasp_type - 1
            )
        else:
            translation_2d, rotation_2d = allegro_tool.special_grasp_information(
                self.open3d_mesh, facenet_index, grasp_width, grasp_type - 1
            )

        _, _, zero_joints = self.zero_cfg
        self.update_cfg(joints=zero_joints)

        return translation_2d, rotation_2d

    def _validate_grasp_type_and_width(
        self, grasp_type: Union[int, str], grasp_width: float
    ) -> None:
        """Validate whether the given grasp type and grasp width fall within
        the allowed range defined in `AllegroHand.GRASP_TYPES`.

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
        if grasp_type_key not in AllegroHand.GRASP_TYPES:
            raise ValueError(
                f"Invalid grasp_type '{grasp_type}'. Must be one of: {list(AllegroHand.GRASP_TYPES.keys())}"
            )

        # Extract allowed width range (min_width, max_width in meters)
        min_width, max_width = AllegroHand.GRASP_TYPES[grasp_type_key]["width"]

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
        """Retrieve a predefined joint configuration for a specified Allegro hand grasp.

        Args:
            grasp_type (int or str):
                Grasp type identifier (1-10 or key).
            grasp_width (float):
                Desired grasp width in cm (rounded to 0.1 cm).

        Returns:
            np.array: Joint configuration array if found, otherwise None.
        """
        self._validate_grasp_type_and_width(grasp_type, grasp_width)

        hand_joints = self.grasp_all_info[AllegroHand.GRASP_TYPES[str(grasp_type)]["name"]][
            f"{grasp_width:.1f}"
        ]["hand_angles"]
        return np.array(hand_joints)

    def get_predefined_pose(
        self,
        grasp_type: Union[int, str],
        grasp_width: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the predefined 2-finger grasp pose (translation and rotation) for a specified Allegro hand grasp.
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

        grasp_info = self.grasp_all_info[AllegroHand.GRASP_TYPES[str(grasp_type)]["name"]][
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
        """Visualize or export the predefined configuration and pose of the Allegro hand for a specified grasp.

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
                    f"allegro_{AllegroHand.GRASP_TYPES[str(grasp_type)]['name']}_{grasp_width: .1f}.{export_format}",
                ),
                export_mesh,
                write_vertex_colors=True,
            )

        _, _, zero_joionts = self.zero_cfg
        self.update_cfg(joints=zero_joionts)


if __name__ == "__main__":
    hand = create_hand_model("AllegroHand", use_cache=True)

    # exmaple 1: visualize the hand predefined configuration
    joints = hand.get_predefined_joints(1, 8)
    hand.update_cfg(joints=joints)
    hand.vis_joints()

    # # exmaple 2: visualize the hand predefined configuration with given pose
    # joints = hand.get_predefined_joints(1, 8)
    # hand.update_cfg(
    #     translation=[0.1,0.2,0.3],
    #     rotation=[0, 0, 0],
    #     joints=joints,
    # )
    # o3d_mesh = hand.open3d_mesh
    # fram = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
    # o3d.visualization.draw_geometries(
    #     [o3d_mesh, fram],
    # )
    # o3d_colli_mesh = hand.open3d_collision_mesh
    # o3d.visualization.draw_geometries(
    #     [o3d_colli_mesh, fram],
    # )

    # # exmaple3: visualize the hand predefined configuration
    hand.vis_predefined_grasp_type(1, 6)

    # # exmaple 4: export the hand predefined configuration visualization
    # hand.vis_predefined_grasp_type(
    #     grasp_type = 10,
    #     grasp_width = 8.5,
    #     is_visualize = True,
    #     is_export=True,
    #     export_path="./grasp_taxonomy_visualization/",
    #     export_format="glb"
    # )

    # # exmaple5: visualize the hand collision model
    hand.vis_joints(collision_geometry=True)

    # example6: print synergy basis
    # cprint.info(f'{hand.mu}')
    # cprint.info(f'{hand.M}')
    # cprint.info(f'{hand.m}')

    # # example7: convert between joint configurations and synergy coordinates
    # cfg_1_8 = hand.get_predefined_joints(1, 8)
    # cfg_1_12 = hand.get_predefined_joints(1, 12)
    # c_1_8 = hand.joints_to_synergy(cfg_1_8)
    # c_1_12 = hand.joints_to_synergy(cfg_1_12)
    # cfg_1_8_bar = hand.synergy_to_joints(c_1_8)

    # cfg_2_8 = hand.get_predefined_joints(2, 8)
    # c_2_8 = hand.joints_to_synergy(cfg_2_8)

    # cprint.info(f'Grasp Type 1, Width 8 cm: Synergy Coords: {c_1_8}')
    # cprint.info(f'Grasp Type 1, Width 12 cm: Synergy Coords: {c_1_12}')
    # cprint.info(f'Grasp Type 2, Width 8 cm: Synergy Coords: {c_2_8}')

    # cprint.info(f'Grasp Type 1, Width 8 cm: Original Config: \n {cfg_1_8}')
    # cprint.info(f'Grasp Type 1, Width 8 cm: Reconstructed Config: \n {cfg_1_8_bar}')

    # # example8: visualize the reconstructed configuration from synergy coords
    # cfg_1_8 = hand.get_predefined_joints(1, 8)
    # c_1_8 = hand.joints_to_synergy(cfg_1_8)
    # hand.vis_predefined_grasp_type(1, 8)
    # cfg_1_8_bar = hand.synergy_to_joints(c_1_8)
    # hand.vis_joints(
    #     np.concatenate([np.zeros(6), np.asarray(cfg_1_8_bar)])
    # )
