import argparse
import os
import numpy as np
np.float = np.float64  # temp fix for following import
import torch
import open3d as o3d
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from utils.config import *
from tools import builder
import utils.utils as utils
from tqdm import tqdm

from graspclutter6dAPI import GC6DGraspNet, GC6DGraspNetEval
from graspnetAPI import GraspNet, GraspNetEval, GraspGroup, Grasp
from collision_detector import ModelFreeCollisionDetector

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_config',
        help = 'yaml config file',
    )
    parser.add_argument(
        'model_checkpoint', 
        help = 'pretrained weight'
    )
    parser.add_argument(
        '--camera',
        type=str,
        default='zivid'
    )
    parser.add_argument('--dataset', default='graspclutter6d', help='Dataset to use [default: graspnet]')
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
    args = parser.parse_args()
    return args

def inference_single(base_model, scene_points, args, config, scene_id, im_id):
    
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



    # gg = gg.nms()
    # gg = gg.sort_by_score()
    # if gg.__len__() > 50:
    #     gg = gg[:50]
    # grippers = gg.to_open3d_geometry_list()
    # cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(scene_points.astype(np.float32))
    # o3d.visualization.draw_geometries([cloud, *grippers])

    


def main():
    args = get_args()

    config = cfg_from_yaml_file(args.model_config)
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.model_checkpoint)
    base_model.to(args.device.lower())
    base_model.eval()

    if args.dataset == 'graspclutter6d':
        g = GC6DGraspNet(args.grasp_root, camera=args.camera, split=args.split)
        ge = GC6DGraspNetEval(args.grasp_root, camera=args.camera, split=args.split)
        scene_ids = g.getSceneIds()
        print('Inferencing on {} scenes'.format(len(scene_ids)))
        for scene_id in tqdm(scene_ids): # if you want small subset, use scene_ids[:10]
            for ann_id in range(13):
                im_id = g.annId2ImgId(ann_id, args.camera)
                scene_points, _ = g.loadScenePointCloud(scene_id, args.camera, ann_id, format='numpy', use_workspace=True)
                inference_single(base_model, scene_points, args, config, str(scene_id).zfill(6), str(im_id).zfill(6))
                break
        res, ap = ge.eval_all(args.dump_dir, proc=args.num_workers)
        save_dir = os.path.join(args.dump_dir, 'ap_{}.npy'.format(args.camera))
        np.save(save_dir, res)
    elif args.dataset == 'graspnet':
        g = GraspNet(root=args.grasp_root, camera=args.camera, split=args.split)
        ge = GraspNetEval(root=args.grasp_root, camera=args.camera, split=args.split)
        scene_ids = g.getSceneIds()
        print('Inferencing on {} scenes'.format(len(scene_ids)))
        for scene_id in tqdm(scene_ids):
            for ann_id in range(256):
                if ann_id % 10 != 0:
                    continue
                print(ann_id)
                scene_points, _ = g.loadScenePointCloud(scene_id, args.camera, ann_id, format='numpy', use_workspace=True, outlier=0.1)
                inference_single(base_model, scene_points, args, config, 'scene_'+str(scene_id).zfill(4), str(ann_id).zfill(4))
        res, ap = ge.eval_similar(args.dump_dir, proc=args.num_workers)
        save_dir = os.path.join(args.dump_dir, 'ap_{}.npy'.format(args.camera))
        np.save(save_dir, res)
    else:
        raise ValueError('Dataset not supported')




if __name__ == '__main__':
    main()
    
    