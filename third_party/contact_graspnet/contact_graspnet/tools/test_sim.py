import argparse
import numpy as np
import json
from pathlib import Path

import argparse
import os
import numpy as np
np.float = np.float64  # temp fix for following import
import torch
import open3d as o3d
import sys




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
# sys.path.append('/SSD1/Workspace/grasping_baselines/contact-graspnet/tools')
import builder

from vgn.detection import VGN
from vgn.detection_implicit import VGNImplicit
from vgn.experiments import clutter_removal
from vgn.utils.misc import set_random_seed
from vgn.grasp import Grasp as G
from vgn.utils.transform import Transform

from utils.config import *
import utils.utils as custom_utils
from collision_detector import ModelFreeCollisionDetector
from graspnetAPI import GraspGroup, Grasp
from scipy.spatial.transform import Rotation as R

# sys.path.append('/SSD1/Workspace/grasping_baselines/anygrasp_sdk/grasp_detection')
sys.path.append(os.path.join(BASE_DIR, '../../', 'anygrasp_sdk', 'grasp_detection'))
print(os.path.join(BASE_DIR, '../', 'anygrasp_sdk'))
try:
    from gsnet import AnyGrasp
except Exception as e:
    print(e)
    print('Can not import AnyGrasp. Skip it')

class CGNet:
    def __init__(self, args):
        config = cfg_from_yaml_file(args.model_config)
        self.base_model = builder.model_builder(config.model)
        builder.load_model(self.base_model, args.model)
        self.base_model.to('cuda')
        self.base_model.eval()

    def __call__(self, state):

        is_acronym = 'ACRONYM' in args.model_config

        # shsh 
        extrinsic = np.array([[ 1.00000000e+00,  6.12323400e-17,  3.08148791e-33, -1.50000000e-01],
                    [ 3.06161700e-17, -5.00000000e-01, -8.66025404e-01,  1.61602540e-01],
                    [-5.30287619e-17,  8.66025404e-01, -5.00000000e-01,  5.20096189e-01],
                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        pcd_o3d = state.pc
        pcd_o3d = pcd_o3d.transform(extrinsic)
        pcd = np.asarray(pcd_o3d.points)
        
        # remove point under 0.01
        if is_acronym:
            pcd_obj = pcd.copy()
            # remove points in the plane using RANSAC
            pcd_obj_o3d = o3d.geometry.PointCloud()
            pcd_obj_o3d.points = o3d.utility.Vector3dVector(pcd_obj)
            plane_model, inliers = pcd_obj_o3d.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
            # select only outliers
            pcd_obj_o3d = pcd_obj_o3d.select_by_index(inliers, invert=True)
            pcd_obj = np.asarray(pcd_obj_o3d.points)
            if pcd_obj.shape[0]  == 0:
                return [], [], 0
            if pcd_obj.shape[0] < 20000:
                # add duplicated points to make it 20000
                idx = np.random.choice(pcd_obj.shape[0], 20000 - pcd_obj.shape[0])
                pcd_obj = np.concatenate([pcd_obj, pcd_obj[idx]], axis=0)

        if is_acronym:
            pcd_torch = torch.from_numpy(pcd_obj).float()
            pred = self.base_model(pcd_torch.unsqueeze(0).to('cuda'))
            pred_grasps = pred['pred_grasps_cam'].detach().cpu() # (B, N, 4, 4)
            pred_scores = pred['pred_scores'][:, :, 0] # (B, N)
            pred_points = pred['pred_points'].detach().cpu() # (B, N, 3)
            pred_grasp_width_bin = pred['offset_pred'] # (B, 1, N)
            pred_grasp_width_bin = pred_grasp_width_bin.transpose(1, 2) # (B, N, 1)
            pc_mean = pred['pc_mean'].detach().cpu()
            pred_grasps[:, :, :3, 3] += pc_mean
            pred_points[:, :, :3] += pc_mean
        else:
            pcd_torch = torch.from_numpy(pcd).float()
            pred = self.base_model(pcd_torch.unsqueeze(0).to('cuda'))
            pred_grasps = pred['pred_grasps'].detach().cpu() # (B, N, 4, 4)
            pred_scores = pred['pred_scores'] # (B, N)
            pred_points = pred['pred_points'].detach().cpu() # (B, N, 3)
            pred_grasp_width_bin = pred['pred_width'] # (B, N, 1)


        if is_acronym:
            control_points = custom_utils.get_acronym_control_point_tensor(1, use_torch=True, symmetric=False)
        else:
            control_points = custom_utils.get_dc6d_control_point_tensor(1, use_torch=True, symmetric=False)
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


        if isinstance(sorted_pred_score, torch.Tensor):
            sorted_pred_score = sorted_pred_score.detach().cpu().numpy()
            pred_control_points = pred_control_points.detach().cpu().numpy()
            sorted_pred_width_bin = sorted_pred_width_bin.detach().cpu().numpy()[0, :, 0]
            sorted_pred_contact_rot = sorted_pred_contact_rot.detach().cpu().numpy()[0]
            sorted_pred_contact_trans = sorted_pred_contact_trans.detach().cpu().numpy()[0]


        pred_line_sets = []
        g_array = []
        lower = [0.03, 0.03, 0.045]
        upper = [0.27, 0.27, 0.5]
        for i in range(len(sorted_idx)):
            width = sorted_pred_width_bin[i]
            if width > 0.08 or width < 0.001:
                continue
            score = sorted_pred_score[i]
            rot = sorted_pred_contact_rot[i]
            trans = sorted_pred_contact_trans[i].reshape(-1)


            # check if the grasp is within the workspace
            x, y, z = trans.copy()
            _trans = np.array([x, y, z, 1])
            _trans = np.matmul(np.linalg.inv(extrinsic), _trans)
            x, y, z, _ = _trans

            if not is_acronym:
                if x < lower[0] or x > upper[0] or y < lower[1] or y > upper[1] or z < lower[2] or z > upper[2]:
                    continue
                # compute angle b/w z axis and the normal of the gripper
                # leave only 30 
                # if np.arccos(np.abs(np.dot(rot[:, 2], [0, 0, 1]))) > np.pi/4:
                #     continue
            
            if is_acronym:
                rot = rot.reshape(3, 3)
                rot_ = R.from_euler('y', -90, degrees=True).as_matrix()
                rot__ = R.from_euler('x', 90, degrees=True).as_matrix()
                rot = rot @ rot_ 
                rot = rot @ rot__
                # trans += np.array([0, 0.0, 0.12]) @ rot_
                trans =  np.linalg.inv(rot) @ trans
                trans += np.array([0.0852, width/2, 0.0])
                trans = rot @ trans
                x, y, z = trans.copy()
                _trans = np.array([x, y, z, 1])
                _trans = np.matmul(np.linalg.inv(extrinsic), _trans)
                x, y, z, _ = _trans

                if x < lower[0] or x > upper[0] or y < lower[1] or y > upper[1] or z < lower[2] or z > upper[2]:
                    continue
            rot = rot.reshape(-1)
            if is_acronym:
                g_array.append([score, width*1.2, 0.02, 0.02, *rot, *trans, -1])
            else:
                g_array.append([score, width, 0.02, 0.02, *rot, *trans, -1])
            
        g_array = np.array(g_array)

        gg = GraspGroup(g_array)

        # grippers = gg.to_open3d_geometry_list()
        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # grippers.append(origin)
        # o3d.visualization.draw_geometries([pcd_o3d, *grippers])

        if gg.__len__() == 0:
            print('no valid grasps')
            return [], [], 0
        if args.collision_thresh > 0:
            # add plane (z=0.055) to the point cloud from -1 to 1
            mfcdetector = ModelFreeCollisionDetector(pcd, voxel_size=0.01)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=args.collision_thresh)
            gg = gg[~collision_mask]

        if gg.__len__() == 0:
            print('no valid grasps')
            return [], [], 0
        
        # gg = gg.nms()
        gg = gg.sort_by_score()
        if gg.__len__() > 10:
            gg = gg[:10]
        
        # grippers = gg.to_open3d_geometry_list()
        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # grippers.append(origin)
        # o3d.visualization.draw_geometries([pcd_o3d, *grippers])

        best_g = gg[0]
        best_g = best_g.transform(np.linalg.inv(extrinsic))

        width = best_g.width * 1.2
        width = max(0.08, width)
        score = best_g.score
        trans = best_g.translation
        rot = best_g.rotation_matrix
        # transform -90 degree around y axis
        # turn on for gc6d, g1b, turn off for acrnoym
        rot_ = R.from_euler('y', 90, degrees=True).as_matrix()
        rot = rot @ rot_

        # convert to scipy rotation
        pose = Transform(R.from_matrix(rot), trans)
        grasp = G(pose, width)

        planning_time = 0.01
        return [grasp], [score], planning_time



class AnyGraspNet:
    def __init__(self, args):
        cfgs = args
        cfgs.checkpoint_path = os.path.join(BASE_DIR, '../../anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar')
        cfgs.max_gripper_width = 0.08
        cfgs.gripper_height = 0.02
        cfgs.top_down_grasp = False
        cfgs.debug = False
        cfgs.max_gripper_width = max(0, min(0.08, cfgs.max_gripper_width))
        print('Loading AnyGrasp model...')
        self.anygrasp = AnyGrasp(cfgs)
        self.anygrasp.load_net()
        print('AnyGrasp model loaded!')

    def __call__(self, state):

        extrinsic = np.array([[ 1.00000000e+00,  6.12323400e-17,  3.08148791e-33, -1.50000000e-01],
                    [ 3.06161700e-17, -5.00000000e-01, -8.66025404e-01,  1.61602540e-01],
                    [-5.30287619e-17,  8.66025404e-01, -5.00000000e-01,  5.20096189e-01],
                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        pcd_o3d = state.pc
        pcd_o3d = pcd_o3d.transform(extrinsic)
        pcd = np.asarray(pcd_o3d.points, dtype=np.float32)
        rgb_img = state.rgb
        depth_img = state.depth
        w, h, fx, fy, cx, cy = 1920, 1080, 1415.0, 1424.0, 969.0, 548.0
        # import matplotlib.pyplot as plt
        # plt.imshow(rgb_img)
        # plt.show()
        
        xmap, ymap = np.arange(depth_img.shape[1]), np.arange(depth_img.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth_img 
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z
        
        mask = (points_z > 0)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = rgb_img[mask].astype(np.float32) / 255.0
        # print(points.min(axis=0), points.max(axis=0))
        # print(np.unique(points[:, 2]))
        # print(points.shape, colors.shape)
        # visualize it using open3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([pcd])
        
        gg, cloud = self.anygrasp.get_grasp(points, colors, apply_object_mask=True, dense_grasp=False, collision_detection=False)
        pred_line_sets = []
        gg = gg.sort_by_score()
        gg = gg[:200]
        
        g_array = []
        lower = [0.03, 0.03, 0.045]
        upper = [0.27, 0.27, 0.5]
        
        for i in range(len(gg)):
            width = gg[i].width
            if width < 0.001 or width > 0.08:
                continue
            score = gg[i].score
            rot = gg[i].rotation_matrix
            trans = gg[i].translation

            x, y, z = trans.copy()
            _trans = np.array([x, y, z, 1])
            _trans = np.matmul(np.linalg.inv(extrinsic), _trans)
            x, y, z, _ = _trans

            if x < lower[0] or x > upper[0] or y < lower[1] or y > upper[1] or z < lower[2] or z > upper[2]:
                continue
        
            rot = rot.reshape(-1)
            g_array.append([score, width, 0.02, 0.02, *rot, *trans, -1])

            
        g_array = np.array(g_array)

        gg = GraspGroup(g_array)
        if args.collision_thresh > 0:
            try:
                mfcdetector = ModelFreeCollisionDetector(pcd, voxel_size=0.01)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=args.collision_thresh)
            except:
                return [], [], 0
            gg = gg[~collision_mask]
        if gg.__len__() == 0:
            return [], [], 0
        
        # gg = gg.nms()
        gg = gg.sort_by_score()
        if gg.__len__() > 10:
            gg = gg[:10]
        
        # grippers = gg.to_open3d_geometry_list()
        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # grippers.append(origin)
        # o3d.visualization.draw_geometries([pcd_o3d, *grippers])

        best_g = gg[0]
        best_g = best_g.transform(np.linalg.inv(extrinsic))

        width = best_g.width * 1.2
        width = max(0.08, width)
        score = best_g.score
        trans = best_g.translation
        rot = best_g.rotation_matrix
        # turn on for gc6d, g1b, turn off for acrnoym
        rot_ = R.from_euler('y', 90, degrees=True).as_matrix()
        rot = rot @ rot_

        # convert to scipy rotation
        pose = Transform(R.from_matrix(rot), trans)
        grasp = G(pose, width)

        planning_time = 0.01
        return [grasp], [score], planning_time


def main(args):

    if args.type in ['giga', 'giga_aff']:
        grasp_planner = VGNImplicit(args.model,
                                    args.type,
                                    best=args.best,
                                    qual_th=args.qual_th,
                                    force_detection=args.force,
                                    out_th=0.1,
                                    select_top=False,
                                    visualize=args.vis)
    elif args.type == 'vgn':
        grasp_planner = VGN(args.model,
                            args.type,
                            best=args.best,
                            qual_th=args.qual_th,
                            force_detection=args.force,
                            out_th=0.1,
                            visualize=args.vis)
    elif args.type == 'contactgraspnet':
        grasp_planner = CGNet(args)
    elif args.type == 'anygrasp':
        grasp_planner = AnyGraspNet(args)
    
    else:
        raise NotImplementedError(f'model type {args.type} not implemented!')

    gsr = []
    dr = []
    for seed in args.seeds:
        set_random_seed(seed)
        success_rate, declutter_rate = clutter_removal.run(
            grasp_plan_fn=grasp_planner,
            logdir=args.logdir,
            description=args.description,
            scene=args.scene,
            object_set=args.object_set,
            num_objects=args.num_objects,
            n=args.num_view,
            num_rounds=args.num_rounds,
            seed=seed,
            sim_gui=args.sim_gui,
            result_path=None,
            add_noise=args.add_noise,
            sideview=args.sideview,
            silence=args.silence,
            visualize=args.vis,)
        gsr.append(success_rate)
        dr.append(declutter_rate)
    results = {
        'gsr': {
            'mean': np.mean(gsr),
            'std': np.std(gsr),
            'val': gsr
        },
        'dr': {
            'mean': np.mean(dr),
            'std': np.std(dr),
            'val': dr
        }
    }
    print('Average results:')
    print(f'Grasp sucess rate: {np.mean(gsr):.2f} ± {np.std(gsr):.2f} %')
    print(f'Declutter rate: {np.mean(dr):.2f} ± {np.std(dr):.2f} %')
    with open(args.result_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default='')
    parser.add_argument("--type", type=str, default='giga')
    parser.add_argument("--logdir", type=Path, default="data/experiments")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--scene",
                        type=str,
                        choices=["pile", "packed"],
                        default="pile")
    parser.add_argument("--object-set", type=str, default="blocks")
    parser.add_argument("--num-objects", type=int, default=5)
    parser.add_argument("--num-view", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs='+', default=[0, 1, 2, 3, 4])
    # parser.add_argument("--seeds", type=int, nargs='+', default=[0])
    parser.add_argument("--sim-gui", action="store_true")
    # parser.add_argument("--grad-refine", action="store_true")
    parser.add_argument("--qual-th", type=float, default=0.9)
    parser.add_argument("--eval-geo",
                        action="store_true",
                        help='whether evaluate geometry prediction')
    parser.add_argument(
        "--best",
        action="store_true",
        help="Whether to use best valid grasp (or random valid grasp)")
    parser.add_argument("--result-path", type=str)
    parser.add_argument(
        "--force",
        action="store_true",
        help=
        "When all grasps are under threshold, force the detector to select the best grasp"
    )
    parser.add_argument(
        "--add-noise",
        type=str,
        default='',
        help="Whether add noise to depth observation, trans | dex | norm | ''")
    parser.add_argument("--sideview",
                        action="store_true",
                        help="Whether to look from one side")
    parser.add_argument("--silence",
                        action="store_true",
                        help="Whether to disable tqdm bar")
    parser.add_argument("--vis",
                        action="store_true",
                        help="visualize and save affordance")
    
    parser.add_argument(
        '--model_config',
        type=str,
        default='./cfgs/DCGRASP1B/Contact_GraspNet.yaml'
    )
    parser.add_argument(
        '--collision_thresh',
        type=float,
        default=0.01
    )


    args = parser.parse_args()
    main(args)
