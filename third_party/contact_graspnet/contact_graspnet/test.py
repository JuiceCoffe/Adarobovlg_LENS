import argparse
import os
import numpy as np
np.float = np.float64  # temp fix for following import
import torch
import open3d as o3d
import sys

from utils.config import *
from tools import builder
import utils.utils as utils
from tqdm import tqdm

from graspclutter6dAPI import GraspClutter6D, GraspClutter6DEval, GraspGroup
from collision_detector import ModelFreeCollisionDetector

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_config',
        default='cfgs/GC6D/Contact_GraspNet.yaml',
        help = 'yaml config file',
    )
    parser.add_argument(
        '--model_checkpoint', 
        default='data/models/gc6d.pth',
        help = 'pretrained weight'
    )
    parser.add_argument(
        '--camera',
        type=str,
        default='realsense-d435'
    )
    parser.add_argument('--dataset', default='graspclutter6d', help='Dataset to use')
    parser.add_argument(
        '--grasp_root',
        type=str,
        default='/SSDg/shback/DiverseClutter6D/'
    )
    parser.add_argument(
        '--device',
        default='cuda:0'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8
    )
    parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
    parser.add_argument('--dump_dir', type=str, default='dump', help='Dump dir to save outputs')
    parser.add_argument('--vis', default=True, action='store_true', help='Visualize the results')
    args = parser.parse_args()
    return args

def inference_single(base_model, scene_points, args, config, scene_id, im_id, color=None):

    pcd_torch = torch.from_numpy(scene_points).float()
    pred = base_model(pcd_torch.unsqueeze(0).to(args.device))
    pred_grasps = pred['pred_grasps'].detach().cpu() # (B, N, 4, 4)
    pred_scores = pred['pred_scores'] # (B, N)
    pred_points = pred['pred_points'].detach().cpu() # (B, N, 3)
    pred_grasp_width_bin = pred['pred_width'] # (B, N, 1)

    control_points = utils.get_dc6d_control_point_tensor(1, use_torch=True, symmetric=False)
    control_points = control_points.unsqueeze(1)
    control_points = control_points.repeat(1, pred_points.shape[1], 1, 1) # (B, N, 5, 3)
    pred_contact_rot = pred_grasps[:, :, :3, :3] # (B, N, 3, 3)
    pred_contact_trans = pred_grasps[:, :, :3, 3] # (B, N, 3)
    pred_contact_rot = pred_contact_rot.type(torch.double)
    pred_contact_trans = pred_contact_trans.type(torch.double)
    
    # sorting the pred score highest to lowest
    sorted_pred_score, sorted_idx = torch.topk(pred_scores.squeeze(), k=2048, largest=True)
    sorted_idx = sorted_idx.detach().cpu()
    sorted_pred_contact_rot = pred_contact_rot[:, sorted_idx, :, :] # (1, N, 3, 3)
    sorted_pred_contact_trans = pred_contact_trans[:, sorted_idx, :] # (1, N, 3) = (x,y,z)
    sorted_pred_width_bin = pred_grasp_width_bin[:, sorted_idx, :] # (1, N, 1)
    control_points = control_points.type(torch.double)
    pred_control_points = torch.matmul(sorted_pred_contact_rot, control_points.permute(0, 1, 3, 2)).permute(0, 1, 3, 2) + sorted_pred_contact_trans.unsqueeze(2) # (B, N, 5, 3)

    sorted_pred_score = sorted_pred_score.detach().cpu().numpy()
    pred_control_points = pred_control_points.detach().cpu().numpy()
    sorted_pred_width_bin = sorted_pred_width_bin.detach().cpu().numpy()[0, :, 0]
    sorted_pred_contact_rot = sorted_pred_contact_rot.detach().cpu().numpy()[0]
    sorted_pred_contact_trans = sorted_pred_contact_trans.detach().cpu().numpy()[0]


    pred_line_sets = []
    g_array = []
    for i in range(len(sorted_idx)):
        score = sorted_pred_score[i]
        width = sorted_pred_width_bin[i]
        rot = sorted_pred_contact_rot[i]
        rot = rot.reshape(-1)
        trans = sorted_pred_contact_trans[i].reshape(-1)
        g_array.append([score, width, 0.02, 0.02, *rot, *trans, -1])
        
    g_array = np.array(g_array)
    gg = GraspGroup(g_array)
    if args.collision_thresh > 0:
        mfcdetector = ModelFreeCollisionDetector(scene_points, voxel_size=0.01)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=args.collision_thresh)
        gg = gg[~collision_mask]

    if args.vis:
        scene_o3d = o3d.geometry.PointCloud()
        scene_o3d.points = o3d.utility.Vector3dVector(scene_points)
        if color is not None:
            scene_o3d.colors = o3d.utility.Vector3dVector(color)
        gg_o3d = gg[:20].to_open3d_geometry_list()
        o3d.visualization.draw_geometries([scene_o3d] + gg_o3d)

    save_dir = os.path.join(args.dump_dir, scene_id, args.camera)
    save_path = os.path.join(save_dir, im_id+'.npy')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gg.save_npy(save_path)
    


def main():
    args = get_args()

    config = cfg_from_yaml_file(args.model_config)
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.model_checkpoint)
    base_model.to(args.device.lower())
    base_model.eval()

    # if args.dataset == 'graspclutter6d':
    #     g = GraspClutter6D(args.grasp_root, camera=args.camera, split=args.split)
    #     ge = GraspClutter6DEval(args.grasp_root, camera=args.camera, split=args.split)
    #     scene_ids = g.getSceneIds()
    #     print('Inferencing on {} scenes'.format(len(scene_ids)))
    #     for scene_id in tqdm(scene_ids): 
    #         for ann_id in range(13):
    #             im_id = g.annId2ImgId(ann_id, args.camera)
    #             scene_points, _ = g.loadScenePointCloud(scene_id, args.camera, ann_id, format='numpy',)
    #             inference_single(base_model, scene_points, args, config, str(scene_id).zfill(6), str(im_id).zfill(6))
    #     res, ap = ge.eval_all(args.dump_dir, proc=args.num_workers)
    #     save_dir = os.path.join(args.dump_dir, 'ap_{}.npy'.format(args.camera))
    #     np.save(save_dir, res)


    # pcd_path = './point_cloud_masked.ply'
    # pcd = o3d.io.read_point_cloud(pcd_path)
    # scene_points = np.asarray(pcd.points)
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
    scene_id = 1
    im_id = 1
    inference_single(base_model, points, args, config, str(scene_id).zfill(6), str(im_id).zfill(6), color)


if __name__ == '__main__':
    main()
    
    
