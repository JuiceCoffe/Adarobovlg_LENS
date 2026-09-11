"""Preprocessing and inference helpers for HandDecision."""

import os

import numpy as np
import open3d as o3d
import torch

from .network import HandDecisionModel
from .utils import find_close_pts, pts2dict


def data_preprocess(
    robot_points,
    robot_normals,
    obj_points,
    obj_normals=None,
    max_obj_pts=512,
    nearest_pts=True,
):
    """Prepare aligned hand/object point features for one grasp candidate."""
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
    if obj_normals is None:
        obj_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        obj_normals = np.asarray(obj_pcd.normals, dtype=np.float32)

    if obj_points.shape[0] > max_obj_pts:
        indices = np.random.choice(obj_points.shape[0], max_obj_pts, replace=False)
        obj_points = obj_points[indices]
        obj_normals = obj_normals[indices]

    if nearest_pts:
        close_pairs = find_close_pts(obj_points, robot_points, threshold=0.025)
        obj_indices = [pair[0] for pair in close_pairs]
        robot_indices = [pair[1] for pair in close_pairs]
        obj_points = obj_points[obj_indices]
        obj_normals = obj_normals[obj_indices]
        robot_points = robot_points[robot_indices]
        robot_normals = robot_normals[robot_indices]

    obj_center = np.mean(obj_points, axis=0, keepdims=True)
    obj_points = obj_points - obj_center
    robot_points = robot_points - obj_center

    return {
        "obj_pts": obj_points.astype(np.float32),
        "obj_normals": obj_normals.astype(np.float32),
        "hand_pts": robot_points.astype(np.float32),
        "hand_normals": robot_normals.astype(np.float32),
    }


def to_data_dict(batch_data):
    """Convert a list of preprocessed samples into a PTv3 input dictionary."""
    obj_pts = [torch.as_tensor(data["obj_pts"], dtype=torch.float32) for data in batch_data]
    obj_normals = [
        torch.as_tensor(data["obj_normals"], dtype=torch.float32) for data in batch_data
    ]
    robot_pts = [
        torch.as_tensor(data["hand_pts"], dtype=torch.float32) for data in batch_data
    ]
    robot_normals = [
        torch.as_tensor(data["hand_normals"], dtype=torch.float32) for data in batch_data
    ]
    features = [
        torch.cat((obj_pts[i], obj_normals[i], robot_pts[i], robot_normals[i]), dim=1)
        for i in range(len(obj_pts))
    ]
    valid_indices = [i for i, points in enumerate(obj_pts) if points.shape[0] > 0]
    return pts2dict(
        [obj_pts[i] for i in valid_indices],
        [features[i] for i in valid_indices],
    )


def hand_decision_init(runtime_config, checkpoint_path):
    """Create the scoring model and load an optional state-dict checkpoint."""
    runtime_config.ptv3["pretrained"] = None
    model = HandDecisionModel(runtime_config)
    model.to(runtime_config.device)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=runtime_config.device,
            weights_only=True,
        )
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint found; initialized HandDecision from scratch.")
    model.eval()
    return model


def hand_decision_infer(model, data_dict):
    """Run batched scoring and return sigmoid probabilities as a NumPy array."""
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.to(model.device)
    with torch.no_grad():
        scores = model(data_dict, sigmoid=True)
    return scores.cpu().numpy()
