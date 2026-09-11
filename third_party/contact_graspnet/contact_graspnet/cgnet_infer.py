import argparse
import os
import numpy as np
np.float = np.float64  # temp fix for following import
import torch
import open3d as o3d
import sys
from .utils.config import *
from .tools import builder
from .utils import utils as utils
from tqdm import tqdm

from graspnetAPI import GraspGroup
from collision_detector import ModelFreeCollisionDetector

def resolve_torch_device(device='cuda:0'):
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    elif str(device).startswith('cuda') and not torch.cuda.is_available():
        print(f"CUDA device {device} requested, but no CUDA GPU is available. Falling back to CPU.")
        device = 'cpu'
    return torch.device(device)

def contact_grasp_init(config_path, checkpoint_path, device='cuda:0'):
    device = resolve_torch_device(device)
    config = cfg_from_yaml_file(config_path)
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, checkpoint_path)
    base_model.to(device)
    base_model.eval()
    base_model.inference_device = device
    return base_model

def farthest_point_sample(xyz, npoint):
    """Select ``npoint`` indices from batched point clouds using FPS."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids


def point_set_belong(A, B, eps=1e-4):
    """Return whether each point in A lies within ``eps`` of point set B."""
    is_num = False
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A).float()
        is_num = True
    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B).float()
        is_num = True
    A_expanded = A.unsqueeze(1)  # (N, 1, 3)
    B_expanded = B.unsqueeze(0)  # (1, M, 3)
    distances = torch.norm(A_expanded - B_expanded, dim=2)  # (N, M)
    min_distances, _ = torch.min(distances, dim=1)  # (N,)
    if is_num:
        min_distances = min_distances.numpy()
    return min_distances < eps

def query_best_points(base_model, scene_points, top_k=5, debug=False, device=None):
    """Return the highest-scoring predicted grasp points."""
    device = resolve_torch_device(device if device is not None else getattr(base_model, 'inference_device', None))
    if isinstance(scene_points, np.ndarray):
        pcd_torch = torch.from_numpy(scene_points).float()
    elif isinstance(scene_points, torch.Tensor):
        pcd_torch = scene_points.float()
    else:
        raise ValueError("scene_points should be numpy array or torch tensor")
    pred = base_model(pcd_torch.unsqueeze(0).to(device))
    pred_grasps = pred['pred_grasps'].detach().cpu() # (B, N, 4, 4)
    pred_scores = pred['pred_scores'] # (B, N)
    pred_points = pred['pred_points'].detach().cpu() # (B, N, 3)
    pred_grasp_width_bin = pred['pred_width'] # (B, N, 1)

    pred_grasps = pred_grasps.view(-1, 4, 4)
    pred_scores = pred_scores.view(-1, 1)
    pred_points = pred_points.view(-1, 3)
    pred_grasp_width_bin = pred_grasp_width_bin.view(-1, 1)

    #sort by scores
    sorted_idx = torch.argsort(pred_scores.squeeze(), descending=True)[:top_k].cpu().numpy()

    # #sort by z value
    # sorted_idx = torch.argsort(pred_points[:, 2], descending=False)[:top_k].cpu().numpy()
    sorted_pred_grasps = pred_grasps[sorted_idx]
    sorted_pred_scores = pred_scores[sorted_idx]
    sorted_pred_points = pred_points[sorted_idx]
    sorted_pred_width_bin = pred_grasp_width_bin[sorted_idx]

    return sorted_pred_points


def adjust_gripper_centers(grasp_group, obj_points, width_scale=1.5):
    if obj_points is None or len(obj_points) == 0 or len(grasp_group) == 0:
        return grasp_group

    tf_len = 0.03
    grasp_trans = grasp_group.translations
    grasp_rots = grasp_group.rotation_matrices
    grasp_heights = grasp_group.heights[:, np.newaxis]
    grasp_depths = grasp_group.depths[:, np.newaxis]
    grasp_widths = grasp_group.widths[:, np.newaxis]

    targets = obj_points[np.newaxis, :, :] - grasp_trans[:, np.newaxis, :]
    targets = np.matmul(targets, grasp_rots)
    targets = np.array(targets, dtype=np.float32)

    mask1 = ((targets[:, :, 2] > -grasp_heights / 2) & (targets[:, :, 2] < grasp_heights / 2))
    mask2 = ((targets[:, :, 0] > grasp_depths - tf_len) & (targets[:, :, 0] < grasp_depths))
    mask4 = (targets[:, :, 1] < -grasp_widths / 2)
    mask6 = (targets[:, :, 1] > grasp_widths / 2)
    inner_mask = (mask1 & mask2 & (~mask4) & (~mask6))

    targets_y = targets[:, :, 1].copy()
    targets_y[~inner_mask] = 0
    ymin = targets_y.min(axis=1)
    ymax = targets_y.max(axis=1)

    offsets = np.zeros([targets.shape[0], 3], dtype=targets.dtype)
    offsets[:, 1] = (ymin + ymax) / 2

    grasp_group.widths = np.maximum(0.025 * np.ones(ymax.shape), width_scale * (ymax - ymin))
    grasp_group.translations += np.matmul(grasp_rots, offsets[:, :, np.newaxis]).squeeze(2)
    return grasp_group


def contact_grasp_infer(base_model, scene_points, top_k=-1, color=None, debug=False, device=None, collision_thresh=0.01, valid_pts=None):
    device = resolve_torch_device(device if device is not None else getattr(base_model, 'inference_device', None))
    if isinstance(scene_points, np.ndarray):
        pcd_torch = torch.from_numpy(scene_points).float()
    elif isinstance(scene_points, torch.Tensor):
        pcd_torch = scene_points.float()
    else:
        raise ValueError("scene_points should be numpy array or torch tensor")
    pred = base_model(pcd_torch.unsqueeze(0).to(device), valid_pts=torch.from_numpy(valid_pts).float().unsqueeze(0).to(device) if valid_pts is not None else None)
    pred_grasps = pred['pred_grasps'].detach().cpu() # (B, N, 4, 4)
    pred_scores = pred['pred_scores'].detach().cpu() # (B, N)
    pred_points = pred['pred_points'].detach().cpu() # (B, N, 3)
    pred_grasp_width_bin = pred['pred_width'].detach().cpu() # (B, N, 1)


    #select grasp in valid_pts
    if valid_pts is not None and len(valid_pts) > 0:
        valid_idx = point_set_belong(pred_points.view(-1,3), torch.from_numpy(valid_pts), eps=3e-3)
        valid_idx = valid_idx.view(1, -1) # (B, N)

        #filter by valid idx
        pred_grasps = pred_grasps[valid_idx].view(1, -1, 4, 4)
        pred_scores = pred_scores[valid_idx].view(1, -1, 1)
        pred_points = pred_points[valid_idx].view(1, -1, 3)
        pred_grasp_width_bin = pred_grasp_width_bin[valid_idx].view(1, -1, 1)
        print('after valid pts filter: ', pred_grasps.shape, pred_scores.shape, pred_points.shape, pred_grasp_width_bin.shape)
        

    control_points = utils.get_dc6d_control_point_tensor(1, use_torch=True, symmetric=False)
    control_points = control_points.unsqueeze(1)
    control_points = control_points.repeat(1, pred_points.shape[1], 1, 1) # (B, N, 5, 3)
    pred_contact_rot = pred_grasps[:, :, :3, :3] # (B, N, 3, 3)
    pred_contact_trans = pred_grasps[:, :, :3, 3] # (B, N, 3)
    pred_contact_rot = pred_contact_rot.type(torch.double)
    pred_contact_trans = pred_contact_trans.type(torch.double)
    
    # sorting the pred score highest to lowest
    sorted_pred_score, sorted_idx = torch.topk(pred_scores.squeeze(), k=pred_scores.shape[1], largest=True)
    sorted_idx = sorted_idx
    sorted_pred_contact_rot = pred_contact_rot[:, sorted_idx, :, :] # (1, N, 3, 3)
    sorted_pred_contact_trans = pred_contact_trans[:, sorted_idx, :] # (1, N, 3) = (x,y,z)
    sorted_pred_width_bin = pred_grasp_width_bin[:, sorted_idx, :] # (1, N, 1)
    sorted_pred_points = pred_points[:, sorted_idx, :] # (1, N, 3)
    control_points = control_points.type(torch.double)
    pred_control_points = torch.matmul(sorted_pred_contact_rot, control_points.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) + sorted_pred_contact_trans.unsqueeze(2) # (B, N, 5, 3)

    sorted_pred_score = sorted_pred_score.float().numpy()
    sorted_pred_points = sorted_pred_points.detach().cpu().numpy()[0]
    sorted_pred_width_bin = sorted_pred_width_bin.detach().cpu().numpy()[0, :, 0]
    sorted_pred_contact_rot = sorted_pred_contact_rot.detach().cpu().numpy()[0]
    sorted_pred_contact_trans = sorted_pred_contact_trans.detach().cpu().numpy()[0]


    pred_line_sets = []
    g_array = []
    gg_points = []
    for i in range(len(sorted_idx)):
        score = sorted_pred_score[i]
        width = sorted_pred_width_bin[i]
        rot = sorted_pred_contact_rot[i]
        rot = rot.reshape(-1)
        trans = sorted_pred_contact_trans[i].reshape(-1)
        g_array.append([score, width, 0.02, 0.02, *rot, *trans, -1])
        gg_points.append(sorted_pred_points[i])
        
    g_array = np.array(g_array)
    gg_points = np.array(gg_points) #Nx3
    gg = GraspGroup(g_array)

    if valid_pts is not None and len(valid_pts) > 0:
        gg = adjust_gripper_centers(gg, valid_pts, width_scale=1.5)

    if collision_thresh > 0:
        mfcdetector = ModelFreeCollisionDetector(scene_points, voxel_size=0.01)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        gg = gg[~collision_mask]
        gg_points = gg_points[~collision_mask]


    sort_idx = np.argsort(gg.scores)[::-1]
    gg = gg[sort_idx]
    gg_points = gg_points[sort_idx]#[:len(sort_idx) // 2]
    if top_k > 0:
        # gg = gg[:top_k]
        if top_k < len(gg):
            stride = len(gg) // top_k
            gg = gg[::stride][:top_k]
            # Apply farthest-point sampling.
            # index = farthest_point_sample(torch.from_numpy(gg_points).float().unsqueeze(0), top_k).numpy()[0]
            # gg = gg[index]
        else:
            gg = gg[:top_k]

    if debug:
        valid_o3d = o3d.geometry.PointCloud()
        valid_o3d.points = o3d.utility.Vector3dVector(valid_pts) if valid_pts is not None else o3d.utility.Vector3dVector()
        valid_o3d.paint_uniform_color([0, 1, 0])

        scene_o3d = o3d.geometry.PointCloud()
        scene_o3d.points = o3d.utility.Vector3dVector(scene_points)
        if color is not None:
            scene_o3d.colors = o3d.utility.Vector3dVector(color[:,::-1])
        
        gg_o3d = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([scene_o3d, valid_o3d] + gg_o3d)

    return gg

def main():

    import cv2
    img_path = './0080_rgb.png'
    depth_path = './0080.png'
    cam_k_path = './camK.npy'
    image = cv2.imread(img_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    cam_k = np.load(cam_k_path) # 3x3
    factor = 1000.0  # Convert depth values from millimeters to meters.
    #create point cloud
    points = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            z = depth[i, j] / factor
            x = (j - cam_k[0, 2]) * z / cam_k[0, 0]
            y = (i - cam_k[1, 2]) * z / cam_k[1, 1]
            points.append([x, y, z])
    points = np.array(points).astype(np.float32)
    color = image.reshape(-1, 3).astype(np.float32)[:, ::-1] / 255.0

    base_model = contact_grasp_init('cfgs/GC6D/Contact_GraspNet.yaml', 'data/models/gc6d.pth')
    gg = contact_grasp_infer(base_model, points, topk=5, color=color, debug=True, device='cuda:0', collision_thresh=0.01)
    print(gg)


if __name__ == '__main__':
    main()
