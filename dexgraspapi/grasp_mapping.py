"""Map two-finger candidates to hand taxonomies and prepare hand output geometry."""

import copy
import numpy as np
import open3d as o3d

from typing import Tuple
from dexgraspapi.multi_finger_grasp_base import MultiFingerGraspGroupBase, create_hand_graspgroup
from graspnetAPI import GraspGroup
from graspnetAPI.grasp import Grasp


class MultiFingerGraspGenerator:
    def __init__(
        self,
        pc_points: np.ndarray,
        tf_gg: GraspGroup,
        hand_name: str,
    ):
        """Convert a two-finger grasp group into multifinger grasp groups
        with different grasp types and grasp depths, without performing
        any collision checking.

        Args:
            pc_points (np.ndarray): point cloud, (N, 3).
            tf_gg (GraspGroup): two-finger grasp group to convert.
            hand_name (str): robot hand name.
        """
        self.tf_len = 0.03  # two-finger gripper finger length
        self.pc_points = pc_points
        self.tf_gg = tf_gg
        self.mf_gg = create_hand_graspgroup(hand_name=hand_name + "GraspGroup")

    def _adjust_gripper_centers(
        self,
        grasp_group: GraspGroup,
        targets: np.ndarray,
        heights: np.ndarray,
        depths: np.ndarray,
        widths: np.ndarray,
        width_scale: float = 1.7,  # note: for anydexgrasp, use 1.7
    ) -> Tuple[GraspGroup, np.ndarray]:
        """Adjust gripper centers based on points within the gripper fingers.

        Args:
            grasp_group (GraspGroup): input grasp group.
            targets (np.ndarray): points in grasp coordinate system, (G, N, 3).
            heights (np.ndarray): grasp heights, (G, 1).
            depths (np.ndarray): grasp depths, (G, 1).
            widths (np.ndarray): grasp widths, (G, 1).
            width_scale (float): scale factor for adjusting grasp widths.

        Returns:
            Tuple[GraspGroup, np.ndarray]: adjusted grasp group and targets.
        """
        targets = np.array(targets, dtype=np.float32)
        ## get point masks
        # height mask
        mask1 = (targets[:, :, 2] > -heights / 2) & (targets[:, :, 2] < heights / 2)
        # left finger mask
        mask2 = (targets[:, :, 0] > depths - self.tf_len) & (targets[:, :, 0] < depths)
        mask4 = targets[:, :, 1] < -widths / 2
        # right finger mask
        mask6 = targets[:, :, 1] > widths / 2
        # get inner mask of each point
        inner_mask = mask1 & mask2 & (~mask4) & (~mask6)

        ## adjust targets and gripper centers
        # get point bounds
        targets_y = targets[:, :, 1].copy()
        targets_y[~inner_mask] = 0
        ymin = targets_y.min(axis=1)
        ymax = targets_y.max(axis=1)
        # get offsets
        offsets = np.zeros([targets.shape[0], 3], dtype=targets.dtype)
        offsets[:, 1] = (ymin + ymax) / 2
        # adjust targets
        targets[:, :, 1] -= offsets[:, np.newaxis, 1]
        # adjust gripper centers
        rots = grasp_group.rotation_matrices

        # update grasp widths and translations
        grasp_group.widths = np.maximum(0.025 * np.ones(ymax.shape), width_scale * (ymax - ymin))
        grasp_group.translations += np.matmul(rots, offsets[:, :, np.newaxis]).squeeze(2)
        return (grasp_group, targets)

    def gen_multifinger_grasps(
        self,
        max_depth_increment_cm: int,
        adjust_gripper_centers: bool = False,
    ) -> Tuple[MultiFingerGraspGroupBase, GraspGroup]:
        """Generate multifinger grasps from two-finger grasps by varying grasp depths and types.

        Args:
            max_depth_increment_cm (int): maximum depth increment in cm.
            adjust_gripper_centers (bool): whether to adjust gripper centers based on points.
        Returns:
            MultiFingerGraspGroupBase: generated multifinger grasp group.
            GraspGroup: corresponding two-finger grasp group.
        """

        # get two-finger grasp parameters
        tf_gg_trans = self.tf_gg.translations
        tf_gg_rots = self.tf_gg.rotation_matrices
        tf_gg_heights = self.tf_gg.heights[:, np.newaxis]
        tf_gg_depths = self.tf_gg.depths[:, np.newaxis]
        tf_gg_widths = self.tf_gg.widths[:, np.newaxis]
        # points in grasp coordinate system
        target_pc = self.pc_points[np.newaxis, :, :] - tf_gg_trans[:, np.newaxis, :]
        target_pc = np.matmul(target_pc, tf_gg_rots)

        # adjust gripper centers if needed
        if adjust_gripper_centers:  # update gripper centers and points
            self.tf_gg, target_pc = self._adjust_gripper_centers(
                self.tf_gg,
                target_pc,
                tf_gg_heights,
                tf_gg_depths,
                tf_gg_widths,
                width_scale=1.5,  # for multifinger grasp, use 1.0
            )

        new_tf_garray, new_types = [], []

        # depth increments in meters
        # depth_increments_m = np.arange(
        #     max_depth_increment_cm + 1, dtype=np.float32
        # ) * 0.01
        # stride 2 cm
        depth_increments_m = np.arange(0, max_depth_increment_cm + 1, 1, dtype=np.float32) * 0.01

        for id in range(len(self.tf_gg)):
            # get base two-finger grasp
            base_tf_grasp: Grasp = self.tf_gg[id]
            base_tf_depth = base_tf_grasp.depth

            # iterate over depth increments
            for depth_inc in depth_increments_m:
                new_tf_depth = base_tf_depth + depth_inc
                # create new multifinger grasp
                new_tf_grasp: Grasp = copy.deepcopy(base_tf_grasp)
                new_tf_grasp.depth = new_tf_depth
                # convert to multifinger grasp for each grasp type
                for gtype in self.mf_gg.mf_gtax.keys():
                    new_tf_garray.append(new_tf_grasp.grasp_array)
                    new_types.append(gtype)

        return (
            self.mf_gg.from_graspgroup(GraspGroup(np.array(new_tf_garray)), new_types),
            GraspGroup(np.array(new_tf_garray)),
        )


def prepare_hand_output(hand_gg, gg, hand_name, pre_width_offset, final_width_offset):
    """Build hand poses and joint arrays shared by static and dynamic services.

    Args:
        hand_gg: Sorted multi-finger candidates in the world frame.
        gg: Corresponding two-finger candidates in the same order.
        hand_name: Supported right-hand model name.
        pre_width_offset: Pre-grasp opening offset in meters.
        final_width_offset: Closing offset in meters, clamped at zero width.

    Returns:
        Tuple of a copied hand group with depth applied and a dictionary of
        pose/joint arrays. Rotations are (G, 3, 3), translations are (G, 3)
        meters, and joint arrays are (G, DoF) radians in hand-model order.
        Input groups are not modified.
    """
    grasp_types = np.asarray(hand_gg.grasp_types).copy()
    grasp_widths = gg.widths.copy()
    depth_bias = hand_gg.depths.reshape(-1, 1) * gg.rotation_matrices[:, :, 0]
    pre_gg = copy.deepcopy(gg)
    pre_gg.widths = grasp_widths + pre_width_offset
    pre_hand = create_hand_graspgroup(hand_name + "GraspGroup").from_graspgroup(pre_gg, grasp_types)
    final_gg = copy.deepcopy(gg)
    final_gg.widths = np.clip(grasp_widths + final_width_offset, 0.0, None)
    final_hand = create_hand_graspgroup(hand_name + "GraspGroup").from_graspgroup(
        final_gg, grasp_types
    )
    visual_hand = copy.deepcopy(hand_gg)
    visual_hand.translations += depth_bias
    return visual_hand, {
        "rotations": visual_hand.rotation_matrices.copy(),
        "translations": visual_hand.translations.copy(),
        "approach_dirs": gg.rotation_matrices[:, :, 0].copy(),
        "grasp_widths": grasp_widths,
        "final_gripper_width": float(grasp_widths[0]),
        "hand_joint_pre_angles": pre_hand.angles.copy(),
        "hand_joint_angles": final_hand.angles.copy(),
        "grasp_types": grasp_types,
    }
