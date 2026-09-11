"""Evaluate multi-finger grasp candidates with sampled hand/object geometry."""

import numpy as np
import torch
from tqdm import tqdm

from .config import config as cfg
from .infer import (
    data_preprocess,
    hand_decision_infer,
    hand_decision_init,
    to_data_dict,
)

from dexgraspapi import create_hand_graspgroup
from robots.hands import create_hand_sampler
from utils.rotation_torch import matrix_to_euler


class HandDecision:
    """Score multi-finger grasp candidates with the HandDecision network."""

    def __init__(self, checkpoint_path):
        """Load the hand-quality scoring network from the supplied checkpoint."""
        self.model = hand_decision_init(cfg, checkpoint_path)

    def evaluate(self, hand_gg, tf_gg, obj_pts, obj_normals=None, hand_name="AllegroHand"):
        """Return a score for each candidate using hand and object geometry."""
        batch_data = []
        hand_model = create_hand_sampler(
            hand_name + "Sampler", device="cpu", num_fixed_points=32768, use_cache=True
        )
        num_grasps = len(hand_gg)
        depth_mf_gg = create_hand_graspgroup(hand_name + "GraspGroup", hand_gg.mf_ggarray.copy())
        depth_bias = tf_gg.rotation_matrices[:, :, 0] * tf_gg.depths.reshape(-1, 1)
        depth_mf_gg.translations += depth_bias
        robot_pts, robot_normals = hand_model.get_sampled_pc(
            joints=torch.tensor(depth_mf_gg.angles).float(),
            translations=torch.tensor(depth_mf_gg.translations).float(),
            rotations=matrix_to_euler(torch.tensor(depth_mf_gg.rotation_matrices).float()),
        )
        robot_pts = robot_pts.numpy()
        robot_normals = robot_normals.numpy()
        for i in tqdm(range(num_grasps)):
            robot_pts_i = robot_pts[i]
            robot_normals_i = robot_normals[i]
            data = data_preprocess(
                robot_pts_i,
                robot_normals_i,
                obj_pts,
                obj_normals,
                max_obj_pts=512,
                nearest_pts=True,
            )
            batch_data.append(data)
        empty_idx = [i for i in range(len(batch_data)) if batch_data[i]["obj_pts"].shape[0] < 2]
        valid_idx = [i for i in range(len(batch_data)) if i not in empty_idx]
        scores = np.zeros((num_grasps,), dtype=np.float32)
        batch_data = [batch_data[i] for i in range(len(batch_data)) if i not in empty_idx]
        if batch_data:
            valid_scores = hand_decision_infer(self.model, to_data_dict(batch_data))
            scores[valid_idx] = np.asarray(valid_scores).reshape(-1)
        self.scores = scores
        return scores
