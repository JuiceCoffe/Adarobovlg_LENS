"""Point-cloud conversion and nearest-neighbor helpers for HandDecision."""

import numpy as np
import open3d as o3d
import torch


def pts2dict(points, features, grid_size=0.005):
    """Convert variable-length point clouds into a PTv3 input dictionary."""
    all_coords = []
    all_features = []
    all_batches = []
    for batch_id, (coords, feats) in enumerate(zip(points, features)):
        all_coords.append(coords)
        all_features.append(feats)
        all_batches.append(
            torch.full((coords.shape[0],), batch_id, dtype=torch.long)
        )
    return {
        "coord": torch.cat(all_coords, dim=0),
        "feat": torch.cat(all_features, dim=0),
        "batch": torch.cat(all_batches, dim=0),
        "grid_size": grid_size,
    }


def dict2pts(data_dict):
    """Split a batched PTv3 dictionary into per-sample points and features."""
    coord = data_dict["coord"]
    feat = data_dict["feat"]
    batch = data_dict["batch"]
    points = []
    features = []
    for batch_id in torch.unique(batch):
        mask = batch == batch_id
        points.append(coord[mask])
        features.append(feat[mask])
    return points, features


def find_close_pts(points_a, points_b, threshold, method="open3d"):
    """Match each point in A to its closest point in B within a radius."""
    if method != "open3d":
        raise ValueError(f"Unknown nearest-neighbor method: {method}")

    points_a = (
        points_a.detach().cpu().numpy() if torch.is_tensor(points_a) else np.asarray(points_a)
    )
    points_b = (
        points_b.detach().cpu().numpy() if torch.is_tensor(points_b) else np.asarray(points_b)
    )
    point_cloud_b = o3d.geometry.PointCloud()
    point_cloud_b.points = o3d.utility.Vector3dVector(points_b)
    search_tree = o3d.geometry.KDTreeFlann(point_cloud_b)

    close_pairs = []
    for index_a, point_a in enumerate(points_a):
        count, indices, squared_distances = search_tree.search_radius_vector_3d(
            point_a, threshold
        )
        if count > 0:
            nearest = int(np.argmin(squared_distances))
            close_pairs.append(
                (
                    index_a,
                    indices[nearest],
                    float(np.sqrt(squared_distances[nearest])),
                )
            )
    return close_pairs
