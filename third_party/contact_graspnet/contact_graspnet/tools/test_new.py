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

import numpy as np
from graspnetAPI.graspnet_eval import *
from graspnetAPI.utils.eval_utils import *

np.random.seed(0)
import torch
torch.manual_seed(0)
import random
random.seed(0)

# ignore_obj_lists = [42, 47, 50, 53, 54, 67, 60, 62, 63, 64, 65, 45, 70, 74, 33, 37, 59] + list(range(75, 88)) \
                    # + [0, 2, 5, 7, 8, 9, 11, 14, 17, 18, 20, 21, 22, 26, 27, 29, 30, 38, 41, 48, 51, 52, 58, 60, 61, 62, 63, 66]
ignore_obj_lists = []

def eval_grasp(grasp_group, models, dexnet_models, poses, config, table=None, voxel_size=0.008, TOP_K=50, model_list=[]):
    '''
    **Input:**

    - grasp_group: GraspGroup instance for evaluation.

    - models: in model coordinate

    - dexnet_models: models in dexnet format

    - poses: from model to camera coordinate

    - config: dexnet config.

    - table: in camera coordinate

    - voxel_size: float of the voxel size.

    - TOP_K: int of the number of top grasps to evaluate.
    '''
    num_models = len(models)
    
    
    model_list
    
    ## grasp nms
    grasp_group = grasp_group.nms(0.03, 30.0 / 180 * np.pi)

    ## assign grasps to object
    # merge and sample scene
    model_trans_list = list()
    seg_mask = list()
    for i, model in enumerate(models):
        model_trans = transform_points(model, poses[i])
        seg = i * np.ones(model_trans.shape[0], dtype=np.int32)
        model_trans_list.append(model_trans)
        seg_mask.append(seg)
    seg_mask = np.concatenate(seg_mask, axis=0)
    scene = np.concatenate(model_trans_list, axis=0)

    # assign grasps
    # indices = compute_closest_points(grasp_group.translations, scene)
    dists = compute_point_distance(grasp_group.translations, scene)
    min_dists = np.min(dists, axis=-1)
    mask = min_dists < 0.02 # filter out grasps that are too far away from the object
    indices = np.argmin(dists, axis=-1)
    model_to_grasp = seg_mask[indices] # assign each grasp to a model
    pre_grasp_list = list()
    for i in range(num_models):
        grasp_i = grasp_group[(model_to_grasp == i) & mask] #
        grasp_i.sort_by_score()
        pre_grasp_list.append(grasp_i[:5].grasp_group_array)
    all_grasp_list = np.vstack(pre_grasp_list)
    remain_mask = np.argsort(all_grasp_list[:, 0])[::-1]
    if len(remain_mask) ==0:
        grasp_list = []
        score_list = []
        collision_mask_list = []
        for i in range(num_models):
            grasp_list.append([])
            score_list.append([])
            collision_mask_list.append([])
        return grasp_list, score_list, collision_mask_list

    min_score = all_grasp_list[remain_mask[min(49, len(remain_mask) - 1)], 0]

    grasp_list = []
    for i in range(num_models):
        remain_mask_i = pre_grasp_list[i][:, 0] >= min_score
        grasp_list.append(pre_grasp_list[i][remain_mask_i])
    # grasp_list = pre_grasp_list

    ## collision detection
    if table is not None:
        scene = np.concatenate([scene, table])

    collision_mask_list, empty_list, dexgrasp_list = collision_detection(
        grasp_list, model_trans_list, dexnet_models, poses, scene, outlier=0.05, return_dexgrasps=True)

    ## evaluate grasps
    # score configurations
    force_closure_quality_config = dict()
    fc_list = np.array([1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    for value_fc in fc_list:
        value_fc = round(value_fc, 2)
        config['metrics']['force_closure']['friction_coef'] = value_fc
        force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(
            config['metrics']['force_closure'])
    # get grasp scores
    score_list = list()

    for i in range(num_models):
        dexnet_model = dexnet_models[i]
        collision_mask = collision_mask_list[i]
        dexgrasps = dexgrasp_list[i]
        scores = list()
        num_grasps = len(dexgrasps)
        for grasp_id in range(num_grasps):
            if collision_mask[grasp_id]:
                scores.append(-1.)
                continue
            if dexgrasps[grasp_id] is None:
                scores.append(-1.)
                continue
            grasp = dexgrasps[grasp_id]
            score = get_grasp_score(grasp, dexnet_model, fc_list, force_closure_quality_config)
            scores.append(score)
        score_list.append(np.array(scores))

    return grasp_list, score_list, collision_mask_list



class NewGraspNetEval(GraspNetEval):

    def eval_scene(self, scene_id, dump_folder, TOP_K=50, return_list=False, vis=False, max_width=0.16):
        '''
        **Input:**

        - scene_id: int of the scene index.

        - dump_folder: string of the folder that saves the dumped npy files.

        - TOP_K: int of the top number of grasp to evaluate

        - return_list: bool of whether to return the result list.

        - vis: bool of whether to show the result

        - max_width: float of the maximum gripper width in evaluation

        **Output:**

        - scene_accuracy: np.array of shape (256, 50, 6) of the accuracy tensor.
        '''
        config = get_config()
        table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)

        list_coe_of_friction = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

        model_list, dexmodel_list, obj_list = self.get_scene_models(scene_id, ann_id=0)
        
        # filter out model_list and dexmodel_list
        model_list = [model for i, model in enumerate(model_list) if obj_list[i] not in ignore_obj_lists]
        dexmodel_list = [model for i, model in enumerate(dexmodel_list) if obj_list[i] not in ignore_obj_lists]
        if len(model_list) == 0:
            return None

        model_sampled_list = list()
        for model in model_list:
            model_sampled = voxel_sample_points(model, 0.008)
            model_sampled_list.append(model_sampled)

        scene_accuracy = []
        grasp_list_list = []
        score_list_list = []
        collision_list_list = []

        for ann_id in range(256):
            grasp_group = GraspGroup().from_npy(
                os.path.join(dump_folder, get_scene_name(scene_id), self.camera, '%04d.npy' % (ann_id,)))
            _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id)
            table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

            # clip width to [0,max_width]
            gg_array = grasp_group.grasp_group_array
            min_width_mask = (gg_array[:, 1] < 0)
            max_width_mask = (gg_array[:, 1] > max_width)
            gg_array[min_width_mask, 1] = 0
            gg_array[max_width_mask, 1] = max_width
            grasp_group.grasp_group_array = gg_array

            grasp_list, score_list, collision_mask_list = eval_grasp(grasp_group, model_sampled_list, dexmodel_list,
                                                                     pose_list, config, table=table_trans,
                                                                     voxel_size=0.008, TOP_K=TOP_K, model_list=model_list)

            # remove empty
            grasp_list = [x for x in grasp_list if len(x) != 0]
            score_list = [x for x in score_list if len(x) != 0]
            collision_mask_list = [x for x in collision_mask_list if len(x) != 0]

            if len(grasp_list) == 0:
                grasp_accuracy = np.zeros((TOP_K, len(list_coe_of_friction)))
                scene_accuracy.append(grasp_accuracy)
                grasp_list_list.append([])
                score_list_list.append([])
                collision_list_list.append([])
                print('\rMean Accuracy for scene:{} ann:{}='.format(scene_id, ann_id), np.mean(grasp_accuracy[:, :]),
                      end='')
                continue

            # concat into scene level
            grasp_list, score_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(
                score_list), np.concatenate(collision_mask_list)

            # sort in scene level
            grasp_confidence = grasp_list[:, 0]
            indices = np.argsort(-grasp_confidence)
            grasp_list, score_list, collision_mask_list = grasp_list[indices], score_list[indices], collision_mask_list[
                indices]

            grasp_list_list.append(grasp_list)
            score_list_list.append(score_list)
            collision_list_list.append(collision_mask_list)

            # calculate AP
            grasp_accuracy = np.zeros((TOP_K, len(list_coe_of_friction)))
            for fric_idx, fric in enumerate(list_coe_of_friction):
                for k in range(0, TOP_K):
                    if k + 1 > len(score_list):
                        grasp_accuracy[k, fric_idx] = np.sum(((score_list <= fric) & (score_list > 0)).astype(int)) / (
                                    k + 1)
                    else:
                        grasp_accuracy[k, fric_idx] = np.sum(
                            ((score_list[0:k + 1] <= fric) & (score_list[0:k + 1] > 0)).astype(int)) / (k + 1)

            print('\rMean Accuracy for scene:%04d ann:%04d = %.3f' % (
            scene_id, ann_id, 100.0 * np.mean(grasp_accuracy[:, :])), end='', flush=True)
            scene_accuracy.append(grasp_accuracy)
            break
        if not return_list:
            return scene_accuracy
        else:
            return scene_accuracy, grasp_list_list, score_list_list, collision_list_list
        
        
    def parallel_eval_scenes(self, scene_ids, dump_folder, proc = 2):
        '''
        **Input:**

        - scene_ids: list of int of scene index.

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - scene_acc_list: list of the scene accuracy.
        '''
        from multiprocessing import Pool
        p = Pool(processes = proc)
        res_list = []
        for scene_id in scene_ids:
            res_list.append(p.apply_async(self.eval_scene, (scene_id, dump_folder)))
        p.close()
        p.join()
        scene_acc_list = []
        for res in res_list:
            if res.get() is not None:
                scene_acc_list.append(res.get())
        return scene_acc_list

    def eval_custom(self, dump_folder, proc = 2):
        '''
        **Input:**

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - res: numpy array of the detailed accuracy.

        - ap: float of the AP for all split.
        '''
        res = np.array(self.parallel_eval_scenes(scene_ids = list(range(130, 140)), dump_folder = dump_folder, proc = proc))
        _res = res.copy()
        _res = _res.transpose(3,0,1,2).reshape(6,-1)
        _res = np.mean(_res, axis=1)
        ap = [np.mean(res), _res[0], _res[1], _res[2], _res[3], _res[4], _res[5]]
        print('\nEvaluation Result (Custom):\n----------\n{}\nAP={:.2f}\nAP0.2={:.2f}\nAP0.4={:.2f}\nAP0.6={:.2f}\nAP0.8={:.2f}\nAP1.0={:.2f}\nAP1.2={:.2f}'.format(
                        self.camera, 100*ap[0], 100*ap[1], 100*ap[2], 100*ap[3], 100*ap[4], 100*ap[5], 100*ap[6]))
        return res, ap



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

    save_dir = os.path.join(args.dump_dir, scene_id, args.camera)
    save_path = os.path.join(save_dir, im_id+'.npy')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(save_path):
        # import os
        os.remove(save_path)
    gg.save_npy(save_path)
    


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
        # for scene_id in tqdm(scene_ids): # if you want small subset, use scene_ids[:10]
        #     for ann_id in range(13):
        #         im_id = g.annId2ImgId(ann_id, args.camera)
        #         scene_points, _ = g.loadScenePointCloud(scene_id, args.camera, ann_id, format='numpy', use_workspace=True)
        #         inference_single(base_model, scene_points, args, config, str(scene_id).zfill(6), str(im_id).zfill(6))
        res, ap = ge.eval_all(args.dump_dir, proc=args.num_workers)
        save_dir = os.path.join(args.dump_dir, 'ap_{}.npy'.format(args.camera))
        np.save(save_dir, res)
    elif args.dataset == 'graspnet':
        g = GraspNet(root=args.grasp_root, camera=args.camera, split=args.split)
        ge = NewGraspNetEval(root=args.grasp_root, camera=args.camera, split=args.split)
        scene_ids = g.getSceneIds()
        scene_ids = list(range(130, 140))
        print('Inferencing on {} scenes'.format(len(scene_ids)))
        for scene_id in tqdm(scene_ids):
            for ann_id in range(256):
                scene_points, _ = g.loadScenePointCloud(scene_id, args.camera, ann_id, format='numpy', use_workspace=True, outlier=0.1)
                inference_single(base_model, scene_points, args, config, 'scene_'+str(scene_id).zfill(4), str(ann_id).zfill(4))
                break
        res, ap = ge.eval_custom(args.dump_dir, proc=args.num_workers)
        save_dir = os.path.join(args.dump_dir, 'ap_{}.npy'.format(args.camera))
        np.save(save_dir, res)
    else:
        raise ValueError('Dataset not supported')




if __name__ == '__main__':
    main()
    
    