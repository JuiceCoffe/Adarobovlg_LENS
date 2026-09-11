import torch
import numpy as np
import open3d as o3d
import math
import torch.nn as nn
import torch.nn.functional as F
import time
from scipy.optimize import linear_sum_assignment
import trimesh
from ..utils.utils import create_robotiq_marker
import time
from tqdm import tqdm
from ..utils import utils as utils
from ..options.config_utils import load_config, load_config_origin
from ..utils import mesh_utils

class ContactGraspNetLoss(nn.Module):
    def __init__(self, config):
        super(ContactGraspNetLoss, self).__init__()
        self.config = config['MODEL']
        bin_weights = self.config['bin_weights']
        self.bin_weights = torch.tensor(bin_weights)
        self.bin_vals = self._get_bin_vals()
        
        gripper_control_points = utils.get_dc6d_control_point_tensor(1, use_torch=True, symmetric=False)
        gripper_control_points_sym = utils.get_dc6d_control_point_tensor(1, use_torch=True, symmetric=True)
        self.gripper_control_points = gripper_control_points
        self.gripper_control_points_sym = gripper_control_points_sym
        
        
    def forward(self, pred, data, epoch, is_train=True):
        """_summary_
        """
        # prediction values
        pred_grasps = pred['pred_grasps'] # (B, N, 4, 4)
        pred_scores = pred['pred_scores'] # (B, N, 1)
        pred_points = pred['pred_points'] # (B, N, 3)
        grasp_width_head = pred['grasp_width_head'].permute(0,2,1) # (B, N, len(bin_vals))
        
        # gt values
        pos_contact_points = data['pos_contact_points'] # (B, M, 3)
        pos_contact_width = data['pos_contact_width'] # (B, M)
        pos_contact_rot = data['pos_contact_rot'] # (B, M, 3, 3)
        pos_contact_trans = data['pos_contact_trans'] # (B, M, 3)
        
        # #* check is the gt values are not wrong
        # for i in range(pred_grasps.shape[0]):
        #     for j in range(128):
        #         gripper_control_points = utils.get_dc6d_control_point_tensor(1, use_torch=False, symmetric=False).squeeze()
        #         gt_rot = pos_contact_rot[i, j].detach().cpu().numpy() # (3, 3)
        #         gt_trans = pos_contact_trans[i, j].detach().cpu().numpy() # (3)
        #         gt_cp = np.matmul(gt_rot, gripper_control_points.T).T + gt_trans
        #         tmp_gt_cp = []
        #         mid_gt_grasp_point = (gt_cp[1] + gt_cp[2]) / 2
        #         tmp_gt_cp.append(gt_cp[0])
        #         tmp_gt_cp.append(mid_gt_grasp_point)
        #         tmp_gt_cp.append(gt_cp[1])
        #         tmp_gt_cp.append(gt_cp[2])
        #         tmp_gt_cp.append(gt_cp[3])
        #         tmp_gt_cp.append(gt_cp[4])
        #         gt_cp = np.asarray(tmp_gt_cp)
        #         gt_line_set = o3d.geometry.LineSet()
        #         gt_line_set.points = o3d.utility.Vector3dVector(gt_cp)
        #         gt_line_set.lines = o3d.utility.Vector2iVector([[3,5], [2,4], [2,3], [0,1]])
        #         gt_line_set.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0] for i in range(4)]))
                
        #         pc = data['pc'][i].detach().cpu().numpy()
        #         pc_o3d = o3d.geometry.PointCloud()
        #         pc_o3d.points = o3d.utility.Vector3dVector(pc)
        #         o3d.visualization.draw_geometries([gt_line_set, pc_o3d])

        #     exit()
        
        
        width_label, grasp_success_label, contact_rot_label, contact_trans_label = self._compute_labels(pred_points=pred_points,
                                                                                                        pos_contact_points=pos_contact_points,
                                                                                                        pos_contact_width=pos_contact_width,
                                                                                                        pos_contact_rot=pos_contact_rot,
                                                                                                        pos_contact_trans=pos_contact_trans)
                
        #* check how many success label is in train data after compute label
        # print(grasp_success_label.shape)
        # tmp_grasp_success_label = grasp_success_label.squeeze(2)
        # pos_idx = torch.nonzero(tmp_grasp_success_label, as_tuple=False)
        # print(pos_idx.shape)
        
        min_geom_loss_divisor = float(1)
        pos_grasps_in_view = torch.clamp(grasp_success_label.sum(dim=1), min=min_geom_loss_divisor) # (B, )
        #* Grasp Confidence Loss
        
        bin_ce_loss = F.binary_cross_entropy(pred_scores, grasp_success_label, reduction='none') # (B, N, 1)
        
        # a number of positive grasp success label is too small, we do not compute the grasp confidence loss for all points
        # sample the negative grasp success label to balance the number of positive and negative grasp success label
        # then select bin_ce_loss using sampled neg grasp idx
        bin_ce_loss_tmp = 0
        negative_grasp_success_label = 1 - grasp_success_label
        for i in range(grasp_success_label.shape[0]):
            neg_grasp_idx = torch.nonzero(negative_grasp_success_label[i].squeeze(1), as_tuple=False)
            # sampled neg grasp idx same as pos_grrasp_in_view
            sampled_neg_grasp_idx = torch.randperm(neg_grasp_idx.shape[0])[:int(pos_grasps_in_view[i])]
            sampled_neg_grasp_idx = neg_grasp_idx[sampled_neg_grasp_idx]
            pos_grasp_idx = torch.nonzero(grasp_success_label[i].squeeze(1), as_tuple=False)
            if pos_grasp_idx.shape[0] == 0:
                pos_grasp_idx = torch.randperm(neg_grasp_idx.shape[0])[:int(pos_grasps_in_view[i])]
                pos_grasp_idx = neg_grasp_idx[pos_grasp_idx]

            sampled_grasp_idx = torch.cat([pos_grasp_idx, sampled_neg_grasp_idx], dim=0).squeeze(1)

            sampled_bin_ce_loss = bin_ce_loss[i, sampled_grasp_idx]
            sampled_bin_ce_loss = torch.mean(sampled_bin_ce_loss)
            bin_ce_loss_tmp += sampled_bin_ce_loss
            
        bin_ce_loss = bin_ce_loss_tmp / grasp_success_label.shape[0]
        
        # print('pred_score', pred_scores[0])
        # print(torch.nonzero(pred_scores[0] >= 0.5, as_tuple=False).shape)
        # if pred_scores.device == torch.device('cuda:0'):
        #     print('pred grasp True: ', torch.nonzero(pred_scores[0].squeeze(1) >= 0.5, as_tuple=False).shape[0], '  pred grasp False: ', torch.nonzero(pred_scores[0].squeeze(1) < 0.5, as_tuple=False).shape[0])
        #     print('gt grasp True: ', torch.nonzero(grasp_success_label[0].squeeze(1) == 1, as_tuple=False).shape[0], '   gt grasp False: ', torch.nonzero(grasp_success_label[0].squeeze(1) == 0, as_tuple=False).shape[0])
        #     print('-------------------------------------------')
                
                
        # print('pred grasp False', torch.nonzero(pred_scores[0].squeeze(1) < 0.5, as_tuple=False).shape[0])
        
        # print('pred grasp True: ', torch.nonzero(pred_scores[0] >= 0.5, as_tuple=False).shape)
        # top k confidence
        # if is_train:
        #     bin_ce_loss, ce_topk_idx = torch.topk(bin_ce_loss.squeeze(), k=256)
        #     bin_ce_loss = torch.mean(bin_ce_loss)
        #     if pred_scores.device == torch.device('cuda:0'):
        #         print('topk pred grasp True: ', torch.nonzero(pred_scores[0].squeeze(1)[ce_topk_idx[0]] >= 0.5, as_tuple=False).shape[0], '  topk pred grasp False: ', torch.nonzero(pred_scores[0].squeeze(1)[ce_topk_idx[0]] < 0.5, as_tuple=False).shape[0])
        #         print('topk gt grasp True: ', torch.nonzero(grasp_success_label[0].squeeze(1)[ce_topk_idx[0]] == 1, as_tuple=False).shape[0], '   topk gt grasp False: ', torch.nonzero(grasp_success_label[0].squeeze(1)[ce_topk_idx[0]] == 0, as_tuple=False).shape[0])
        #         print('-------------------------------------------')
        # else:
        #     bin_ce_loss = torch.mean(bin_ce_loss)
        
        # no top_k confidence
        # bin_ce_loss = torch.mean(torch.mean(bin_ce_loss.squeeze(2), dim=-1), dim=-1)
        
        #* Grasp Width Loss
        # convert to multihot
        bin_vals = self.config['offset_bins']
        grasp_width_labels_multihot = self._bin_labels_to_multihot(width_label, bin_vals) # (B, N, 10)
        width_loss = F.binary_cross_entropy_with_logits(grasp_width_head,
                                                        grasp_width_labels_multihot,
                                                        reduction='none') # (B, N, 1)
        device = pred_scores.device
        bin_weights = self.bin_weights[None, None, :].to(device)
        width_loss = (bin_weights * width_loss).mean(axis=2)
        masked_width_loss = width_loss * grasp_success_label.squeeze()
        width_loss = torch.mean(torch.sum(masked_width_loss, axis=1, keepdim=True) / pos_grasps_in_view)
        
        
        #* Grasp 6D Pose Loss
        # select positive grasp
        success_mask_rot = grasp_success_label.bool()[:, :, :, None] # (B, N, 1, 1)
        success_mask_rot = torch.broadcast_to(success_mask_rot, contact_rot_label.shape) # (B, N, 3, 3)
        pos_contact_rot_label = torch.where(success_mask_rot, contact_rot_label, torch.ones_like(contact_rot_label)*100000) # (B, N, 3, 3)
        success_mask_trans = grasp_success_label.bool() # (B, N, 1)
        success_mask_trans = torch.broadcast_to(success_mask_trans, contact_trans_label.shape) # (B, N, 3)
        pos_contact_trans_label = torch.where(success_mask_trans, contact_trans_label, torch.ones_like(contact_trans_label)*100000) # (B, N, 3)
        
        # expand gripper control points to match the number of the points
        self.gripper_control_points = self.gripper_control_points.to(device)
        self.gripper_control_points_sym = self.gripper_control_points_sym.to(device)
        
        control_points = self.gripper_control_points.unsqueeze(1) # (1, 1, 5, 3)
        control_points = control_points.repeat(pred_points.shape[0], pred_points.shape[1], 1, 1) # (B, N, 5, 3)
        sym_control_points = self.gripper_control_points_sym.unsqueeze(1) # (1, 1, 5, 3)
        sym_control_points = sym_control_points.repeat(pred_points.shape[0], pred_points.shape[1], 1, 1) # (B, N, 5, 3)
        
        # transform pred and gt control points
        pred_contact_rot = pred_grasps[:, :, :3, :3] # (B, N, 3, 3)
        pred_contact_trans = pred_grasps[:, :, :3, 3] # (B, N, 3)
        pred_control_points = torch.matmul(pred_contact_rot, control_points.permute(0,1,3,2)).permute(0,1,3,2) + pred_contact_trans.unsqueeze(2)
        gt_control_points = torch.matmul(pos_contact_rot_label, control_points.permute(0,1,3,2)).permute(0,1,3,2) + pos_contact_trans_label.unsqueeze(2)
        sym_gt_control_points = torch.matmul(pos_contact_rot_label, sym_control_points.permute(0,1,3,2)).permute(0,1,3,2) + pos_contact_trans_label.unsqueeze(2)
        
        # #* visualize pred control points and gt control points with pred points
        # for i in range(pred_control_points.shape[0]):
        #     pred_line_sets = []
        #     gt_line_sets = []
        #     for j in range(128):
        #         if grasp_success_label[i, j] == 0:
        #             continue
        #         else:
        #             pred_cp = pred_control_points[i, j].detach().cpu().numpy()
        #             gt_cp = gt_control_points[i, j].detach().cpu().numpy() # (5, 3)
        #             tmp_pred_cp = []
        #             mid_pred_grasp_point = (pred_cp[1] + pred_cp[2]) / 2
        #             tmp_pred_cp.append(pred_cp[0])
        #             tmp_pred_cp.append(mid_pred_grasp_point)
        #             tmp_pred_cp.append(pred_cp[1])
        #             tmp_pred_cp.append(pred_cp[2])
        #             tmp_pred_cp.append(pred_cp[3])
        #             tmp_pred_cp.append(pred_cp[4])
        #             pred_cp = np.asarray(tmp_pred_cp)
        #             tmp_gt_cp = []
        #             mid_gt_grasp_point = (gt_cp[1] + gt_cp[2]) / 2
        #             tmp_gt_cp.append(gt_cp[0])
        #             tmp_gt_cp.append(mid_gt_grasp_point)
        #             tmp_gt_cp.append(gt_cp[1])
        #             tmp_gt_cp.append(gt_cp[2])
        #             tmp_gt_cp.append(gt_cp[3])
        #             tmp_gt_cp.append(gt_cp[4])
        #             gt_cp = np.asarray(tmp_gt_cp)
                    
        #             pred_line_set = o3d.geometry.LineSet()
        #             pred_line_set.points = o3d.utility.Vector3dVector(pred_cp)
        #             pred_line_set.lines = o3d.utility.Vector2iVector([[3,5], [2,4], [2,3], [0,1]])
        #             pred_line_set.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0] for i in range(4)]))
        #             pred_line_sets.append(pred_line_set)
        #             gt_line_set = o3d.geometry.LineSet()
        #             gt_line_set.points = o3d.utility.Vector3dVector(gt_cp)
        #             gt_line_set.lines = o3d.utility.Vector2iVector([[3,5], [2,4], [2,3], [0,1]])
        #             gt_line_set.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0] for i in range(4)]))
        #             gt_line_sets.append(gt_line_set)
                    
        #             pc = data['pc'][i].detach().cpu().numpy()
        #             pc_o3d = o3d.geometry.PointCloud()
        #             pc_o3d.points = o3d.utility.Vector3dVector(pc)
                    
        #             # print pred width and gt width
        #             # print('pred width: ', grasp_width_head[i, j])
        #             print('pred width idx: ', torch.argmax(grasp_width_head[i, j]))
                    
        #             print('gt width idx: ', torch.argmax(grasp_width_labels_multihot[i, j]))
                    
                    
        #             o3d.visualization.draw_geometries([gt_line_set, pc_o3d, pred_line_set])
        #     exit()
        
        
        
        # compute dist btw pred and gt control points
        expanded_pred_control_points = pred_control_points.unsqueeze(2) # (B, N, 1, 5, 3)
        expanded_gt_control_points = gt_control_points.unsqueeze(1) # (B, 1, M, 5, 3)
        expanded_sym_gt_control_points = sym_gt_control_points.unsqueeze(1) # (B, 1, M, 5, 3)
        
        # sum of squared dist btw all points
        squared_add = torch.sum((expanded_pred_control_points - expanded_gt_control_points)**2, dim=(3,4)) # (B, N, M)
        sym_squared_add = torch.sum((expanded_pred_control_points - expanded_sym_gt_control_points)**2, dim=(3,4)) # (B, N, M

        # combine dist btw gt and sym gt grasp and take min dist to gt grasp for each pred grasp
        squared_adds = torch.concat([squared_add, sym_squared_add], dim=-1) # (B, N, 2*M)
        squared_adds_k = torch.topk(squared_adds, k=1, dim=2, largest=False, sorted=False)[0] # (B, N, 1)

        # mask negative grasp
        sum_grasp_success_labels = torch.sum(grasp_success_label, dim=2, keepdim=True) # (B, N, 1)
        binary_grasp_success_labels = torch.clamp(sum_grasp_success_labels, 0, 1)
        min_adds = binary_grasp_success_labels * torch.sqrt(squared_adds_k) # (B, N, 1)
        adds_loss = torch.sum(pred_scores * min_adds, dim=(1), keepdim=True) # (B, 1, 1)
        adds_loss = adds_loss.squeeze() / pos_grasps_in_view.squeeze() # B
        adds_loss = torch.mean(adds_loss)
        total_loss = 1*bin_ce_loss + 1*width_loss + 1*adds_loss
        # total_loss = bin_ce_loss
        # loss_info = {
        #     'bin_ce_loss': bin_ce_loss,
        #     'width_loss': width_loss,
        #     'adds_loss': adds_loss
        # }
        return total_loss, bin_ce_loss, width_loss, adds_loss
        
    
    def _get_bin_vals(self):
        """
        Creates bin values for grasping widths according to bounds defined in config
        """
        bins_bounds = np.array(self.config['offset_bins'])
        bin_vals = (bins_bounds[1:] + bins_bounds[:-1]) / 2
        bin_vals[-1] = bins_bounds[-1]
        bin_vals = np.minimum(bin_vals, self.config['gripper_width']-0.005)
        bin_vals = torch.tensor(bin_vals, dtype=torch.float32)
        
        return bin_vals

    def _compute_labels(self, pred_points, pos_contact_points, pos_contact_width, pos_contact_rot, pos_contact_trans):
        """_summary_
        Project gt grasp labels on the predicted points
        All points w/o nearby successful grasp contact are considered negative contact points
        
        Args:
            pred_points (_type_): _description_
            pos_contact_points (_type_): _description_
            pos_contact_width (_type_): _description_
            pos_contact_rot (_type_): _description_
            pos_contact_trans (_type_): _description_

        Returns:
            _type_: _description_
        """
        nsample = 1
        radius = 0.005
        filter_z = True
        z_val = -0.1
        
        _, N, _ = pred_points.shape
        B, M, _ = pos_contact_points.shape
        
        # make grasp width B, M, 1
        pos_contact_width = pos_contact_width[:, :, None]
        
        # if filter_z:
        #     # filter out direction that are too far
        
        # compute distance
        pred_points = pred_points.unsqueeze(2) # (B, N, 1, 3)
        pos_contact_points = pos_contact_points.unsqueeze(1) # (B, 1, M, 3)
        squared_dist = torch.sum((pred_points - pos_contact_points)**2, dim=-1) # (B, N, M)
        squared_dist_k, close_contact_pt_idcs = torch.topk(squared_dist, k=nsample, dim=2, largest=False, sorted=False) # (B, N, nsample)
        
        # group labels
        grouped_contact_width = utils.index_points(pos_contact_width, close_contact_pt_idcs) # (B, N, nsample, 1)
        grouped_contact_rot = utils.index_points(pos_contact_rot, close_contact_pt_idcs) # (B, N, nsample, 3, 3)
        grouped_contact_trans = utils.index_points(pos_contact_trans, close_contact_pt_idcs) # (B, N, nsample, 3)
        # compute label
        width_label = grouped_contact_width.mean(dim=2) # (B, N, 3)
        grasp_success_label = torch.mean(squared_dist_k, dim=2, keepdim=True) < radius**2 # (B, N, 1)
        grasp_success_label = grasp_success_label.type(torch.float32)
        grasp_contact_rot_label = grouped_contact_rot.mean(dim=2) # (B, N, 3, 3)
        grasp_contact_trans_label = grouped_contact_trans.mean(dim=2) # (B, N, 3)
        
        return width_label, grasp_success_label, grasp_contact_rot_label, grasp_contact_trans_label
        
    def _bin_labels_to_multihot(self, cont_labels, bin_boundaries):
        """_summary_
        Computes binned grasp width labels from continous labels and bin boundaries
        
        Args:
            garsp_width_labels (_type_): _description_
            bin_vals (_type_): _description_

        Returns:
            _type_: _description_
        """
        bins = []
        for b in range(len(bin_boundaries)-1):
            bins.append(torch.logical_and(torch.greater_equal(cont_labels, bin_boundaries[b]), torch.less(cont_labels, bin_boundaries[b+1])))
        multi_hot_labels = torch.cat(bins, dim=2)
        multi_hot_labels = multi_hot_labels.to(dtype=torch.float32, device=cont_labels.device)
        
        return multi_hot_labels


class ContactGraspNetLossOrigin(nn.Module):
    def __init__(self, config):
        super(ContactGraspNetLossOrigin, self).__init__()
        self.global_config = config
        config_losses = [
            'pred_contact_base',
            'pred_contact_success', # True
            'pred_contact_offset',  # True
            'pred_contact_approach',
            'pred_grasps_adds',  # True
            'pred_grasps_adds_gt2pred'
        ]

        config_weights = [
            'dir_cosine_loss_weight',
            'score_ce_loss_weight',  # True
            'offset_loss_weight',  # True
            'approach_cosine_loss_weight',
            'adds_loss_weight',  # True
            'adds_gt2pred_loss_weight'
        ]
        
        bin_weights = self.global_config['DATA']['labels']['bin_weights']
        self.bin_weights = torch.tensor(bin_weights)
        self.bin_vals = self._get_bin_vals()
        
        for config_loss, config_weight in zip(config_losses, config_weights):
            if self.global_config['MODEL'][config_loss]:
                setattr(self, config_weight, self.global_config['OPTIMIZER'][config_weight])
            else:
                setattr(self, config_weight, 0.0)
        self.gripper = mesh_utils.create_gripper('panda')
        
        n_copies = 1  # We will repeat this according to the batch size
        gripper_control_points = self.gripper.get_control_point_tensor(n_copies) # b x 5 x 3
        sym_gripper_control_points = self.gripper.get_control_point_tensor(n_copies, symmetric=True)
        
        self.gripper_control_points_homog = torch.cat([gripper_control_points,
            torch.ones((n_copies, gripper_control_points.shape[1], 1))], dim=2)  # b x 5 x 4
        self.sym_gripper_control_points_homog = torch.cat([sym_gripper_control_points,
            torch.ones((n_copies, gripper_control_points.shape[1], 1))], dim=2)  # b x 5 x 4

        # self.gripper_control_points_homog = self.gripper_control_points_homog.cuda()
        # self.sym_gripper_control_points_homog = self.sym_gripper_control_points_homog.cuda()
    
    def forward(self, pred, target):
        """
        Computes loss terms from pointclouds, network predictions and labels

        Arguments:
            pointclouds_pl {tf.placeholder} -- bxNx3 input point clouds
            end_points {dict[str:tf.variable]} -- endpoints of the network containing predictions
            dir_labels_pc_cam {tf.variable} -- base direction labels in camera coordinates (bxNx3)
            offset_labels_pc {tf.variable} -- grasp width labels (bxNx1)
            grasp_success_labels_pc {tf.variable} -- contact success labels (bxNx1)
            approach_labels_pc_cam {tf.variable} -- approach direction labels in camera coordinates (bxNx3)
            global_config {dict} -- config dict

        Returns:
            [dir_cosine_loss, bin_ce_loss, offset_loss, approach_cosine_loss, adds_loss,
            adds_loss_gt2pred, gt_control_points, pred_control_points, pos_grasps_in_view] -- All losses (not all are used for training)
        """
        pred_grasps_cam = pred['pred_grasps_cam']           # B x N x 4 x 4
        pred_scores = pred['pred_scores']                   # B x N x 1
        device = pred_scores.device
        self.bin_weights = self.bin_weights.to(device)
        self.bin_vals = self.bin_vals.to(device)
        self.gripper_control_points_homog = self.gripper_control_points_homog.to(device)
        self.sym_gripper_control_points_homog = self.sym_gripper_control_points_homog.to(device)
        pred_points = pred['pred_points']                   # B x N x 3
        # offset_pred = pred['offset_pred']                   # B x N  # We use the grasp_offset_head instead of this
        grasp_offset_head = pred['grasp_offset_head'].permute(0, 2, 1)       # B x N x 10


        # # Generated in acronym_dataloader.py
        # grasp_success_labels_pc = target['grasp_success_label']  # B x N
        # grasp_offset_labels_pc = target['grasp_diff_label']    # B x N x 3

        # approach_labels_pc_cam = target['grasp_approach_label']    # B x N x 3
        # dir_labels_pc_cam = target['grasp_dir_label']              # B x N x 3
        # pointclouds_pl = target['pc_cam']                    # B x N x 3

        # -- Interpolate Labels -- #
        pos_contact_points = target['pos_contact_points']    # B x M x 3
        pos_contact_dirs = target['pos_contact_dirs']        # B x M x 3
        pos_finger_diffs = target['pos_finger_diffs']        # B x M
        pos_approach_dirs = target['pos_approach_dirs']      # B x M x 3
        camera_pose = target['camera_pose']                  # B x 4 x 4
        
        dir_labels_pc_cam, \
        grasp_offset_labels_pc, \
        grasp_success_labels_pc, \
        approach_labels_pc_cam, \
        debug = self._compute_labels(pred_points, 
                                     camera_pose,
                                     pos_contact_points,
                                     pos_contact_dirs,
                                     pos_finger_diffs,
                                     pos_approach_dirs)
        
        # I think this is the number of positive grasps that are in view
        min_geom_loss_divisor = float(self.global_config['LOSS']['min_geom_loss_divisor'])  # This is 1.0
        pos_grasps_in_view = torch.clamp(grasp_success_labels_pc.sum(dim=1), min=min_geom_loss_divisor)  # B
        # pos_grasps_in_view = torch.maximum(grasp_success_labels_pc.sum(dim=1), min_geom_loss_divisor)  # B

        total_loss = 0.0

        if self.dir_cosine_loss_weight > 0:
            raise NotImplementedError

        # -- Grasp Confidence Loss -- #
        if self.score_ce_loss_weight > 0:  # TODO (bin_ce_loss)
            bin_ce_loss = F.binary_cross_entropy(pred_scores, grasp_success_labels_pc, reduction='none')  # B x N x 1
            if 'topk_confidence' in self.global_config['LOSS'] \
                and self.global_config['LOSS']['topk_confidence']:
                bin_ce_loss, _ = torch.topk(bin_ce_loss.squeeze(), k=self.global_config['LOSS']['topk_confidence'])
            bin_ce_loss = torch.mean(bin_ce_loss)

            total_loss += self.score_ce_loss_weight * bin_ce_loss

        if self.offset_loss_weight > 0:  # TODO  (offset_loss)
            if self.global_config['MODEL']['bin_offsets']:
                # Convert labels to multihot
                bin_vals = self.global_config['DATA']['labels']['offset_bins']
                grasp_offset_labels_multihot = self._bin_label_to_multihot(grasp_offset_labels_pc, 
                                                                           bin_vals)

                if self.global_config['LOSS']['offset_loss_type'] == 'softmax_cross_entropy':
                    raise NotImplementedError

                else:
                    offset_loss = F.binary_cross_entropy_with_logits(grasp_offset_head,
                                                                     grasp_offset_labels_multihot, reduction='none')  # B x N x 1
                    if 'too_small_offset_pred_bin_factor' in self.global_config['LOSS'] \
                        and self.global_config['LOSS']['too_small_offset_pred_bin_factor']:
                        raise NotImplementedError

                    # Weight loss for each bin
                    shaped_bin_weights = self.bin_weights[None, None, :]
                    offset_loss = (shaped_bin_weights * offset_loss).mean(axis=2)
            else:
                raise NotImplementedError
            masked_offset_loss = offset_loss * grasp_success_labels_pc.squeeze()
            # Divide each batch by the number of successful grasps in the batch
            offset_loss = torch.mean(torch.sum(masked_offset_loss, axis=1, keepdim=True) / pos_grasps_in_view)

            total_loss += self.offset_loss_weight * offset_loss
        
        if self.approach_cosine_loss_weight > 0:
            raise NotImplementedError
        
        # -- 6 Dof Pose Loss -- #
        if self.adds_loss_weight > 0:  # TODO  (adds_loss)
            # Build groudn truth grasps and compare distances to predicted grasps

            ### ADS Gripper PC Loss
            # Get 6 DoF pose of predicted grasp
            if self.global_config['MODEL']['bin_offsets']:
                thickness_gt = self.bin_vals[torch.argmax(grasp_offset_labels_pc, dim=2)]
            else:
                thickness_gt = grasp_offset_labels_pc[:, :, 0]
            
            # TODO: Move this to dataloader? 
            pred_grasps = pred_grasps_cam  # B x N x 4 x 4
            gt_grasps_proj = utils.build_6d_grasp(approach_labels_pc_cam, 
                                                            dir_labels_pc_cam, 
                                                            pred_points, 
                                                            thickness_gt, 
                                                            use_torch=True,
                                                            device=pred_grasps_cam.device) # b x N x 4 x 4
            # Select positive grasps I think?
            success_mask = grasp_success_labels_pc.bool()[:, :, :, None] # B x N x 1 x 1
            success_mask = torch.broadcast_to(success_mask, gt_grasps_proj.shape) # B x N x 4 x 4
            pos_gt_grasps_proj = torch.where(success_mask, gt_grasps_proj, torch.ones_like(gt_grasps_proj) * 100000) # B x N x 4 x 4

            # Expand gripper control points to match number of points
            # only use per point pred grasps but not per point gt grasps
            control_points = self.gripper_control_points_homog.unsqueeze(1)  # 1 x 1 x 5 x 4
            control_points = control_points.repeat(pred_points.shape[0], pred_points.shape[1], 1, 1)  # b x N x 5 x 4

            sym_control_points = self.sym_gripper_control_points_homog.unsqueeze(1)  # 1 x 1 x 5 x 4
            sym_control_points = sym_control_points.repeat(pred_points.shape[0], pred_points.shape[1], 1, 1)  # b x N x 5 x 4

            pred_control_points = torch.matmul(control_points, pred_grasps.permute(0, 1, 3, 2))[:, :, :, :3]  # b x N x 5 x 3

            # Transform control points to ground truth locations
            gt_control_points = torch.matmul(control_points, pos_gt_grasps_proj.permute(0, 1, 3, 2))[:, :, :, :3]  # b x N x 5 x 3
            sym_gt_control_points = torch.matmul(sym_control_points, pos_gt_grasps_proj.permute(0, 1, 3, 2))[:, :, :, :3]  # b x N x 5 x 3

            # Compute distances between predicted and ground truth control points
            expanded_pred_control_points = pred_control_points.unsqueeze(2)         # B x N x 1 x 5 x 3
            expanded_gt_control_points = gt_control_points.unsqueeze(1)             # B x 1 x N' x 5 x 3  I think N' == N
            expanded_sym_gt_control_points = sym_gt_control_points.unsqueeze(1)     # B x 1 x N' x 5 x 3  I think N' == N

            # Sum of squared distances between all points
            squared_add = torch.sum((expanded_pred_control_points - expanded_gt_control_points)**2, dim=(3, 4))  # B x N x N'
            sym_squared_add = torch.sum((expanded_pred_control_points - expanded_sym_gt_control_points)**2, dim=(3, 4))  # B x N x N'

            # Combine distances between gt and symmetric gt grasps
            squared_adds = torch.concat([squared_add, sym_squared_add], dim=2)  # B x N x 2N'

            # Take min distance to gt grasp for each predicted grasp
            squared_adds_k = torch.topk(squared_adds, k=1, dim=2, largest=False)[0]  # B x N

            # Mask negative grasps
            # TODO: If there are bugs, its prob here.  The original code sums on axis=1
            # Which just determines if there is a successful grasp in the batch.  
            # I think we just want to select the positive grasps so the sum is redundant.
            sum_grasp_success_labels = torch.sum(grasp_success_labels_pc, dim=2, keepdim=True)
            binary_grasp_success_labels = torch.clamp(sum_grasp_success_labels, 0, 1)
            min_adds = binary_grasp_success_labels * torch.sqrt(squared_adds_k)  # B x N x 1
            adds_loss = torch.sum(pred_scores * min_adds, dim=(1), keepdim=True)  # B x 1
            adds_loss = adds_loss.squeeze() / pos_grasps_in_view.squeeze()  # B x 1
            adds_loss = torch.mean(adds_loss)
            total_loss += self.adds_loss_weight * adds_loss

        if self.adds_gt2pred_loss_weight > 0:
            raise NotImplementedError
        
        return total_loss, bin_ce_loss, offset_loss, adds_loss

    def _get_bin_vals(self):
        """
        Creates bin values for grasping widths according to bounds defined in config

        Arguments:
            global_config {dict} -- config

        Returns:
            tf.constant -- bin value tensor
        """
        bins_bounds = np.array(self.global_config['DATA']['labels']['offset_bins'])
        if self.global_config['TEST']['bin_vals'] == 'max':
            bin_vals = (bins_bounds[1:] + bins_bounds[:-1])/2
            bin_vals[-1] = bins_bounds[-1]
        elif self.global_config['TEST']['bin_vals'] == 'mean':
            bin_vals = bins_bounds[1:]
        else:
            raise NotImplementedError

        if not self.global_config['TEST']['allow_zero_margin']:
            bin_vals = np.minimum(bin_vals, self.global_config['DATA']['gripper_width']-self.global_config['TEST']['extra_opening'])

        bin_vals = torch.tensor(bin_vals, dtype=torch.float32)
        return bin_vals
    
    
    def _bin_label_to_multihot(self, cont_labels, bin_boundaries):
        """
        Computes binned grasp width labels from continuous labels and bin boundaries

        Arguments:
            cont_labels {torch.Tensor} -- continuous labels
            bin_boundaries {list} -- bin boundary values

        Returns:
            torch.Tensor -- one/multi hot bin labels
        """
        bins = []
        for b in range(len(bin_boundaries)-1):
            bins.append(torch.logical_and(torch.greater_equal(cont_labels, bin_boundaries[b]), torch.less(cont_labels, bin_boundaries[b+1])))
        multi_hot_labels = torch.cat(bins, dim=2)
        multi_hot_labels = multi_hot_labels.to(torch.float32)
        return multi_hot_labels
    
    
    def _compute_labels(self, 
                        processed_pc_cams: torch.Tensor, 
                        camera_poses: torch.Tensor, 
                        pos_contact_points: torch.Tensor,
                        pos_contact_dirs: torch.Tensor,
                        pos_finger_diffs: torch.Tensor, 
                        pos_approach_dirs: torch.Tensor):
        """
        Project grasp labels defined on meshes onto rendered point cloud 
        from a camera pose via nearest neighbor contacts within a maximum radius. 
        All points without nearby successful grasp contacts are considered 
        negative contact points.

        Here N is the number of points returned by the PointNet Encoder (2048) while
        M is the number of points in the ground truth data.  B is the batch size.
        We are trying to assign a label to each of the PointNet points by 
        sampling the nearest ground truth points.

        Arguments:
            pc_cam_pl (torch.Tensor): (B, N, 3) point cloud in camera frame
            camera_pose_pl (torch.Tensor): (B, 4, 4) homogenous camera pose
            pos_contact_points (torch.Tensor): (B, M, 3) contact points in world frame (3 DoF points)
            pos_contact_dirs (torch.Tensor): (B, M, 3) contact directions (origin centered vectors?)
            pos_finger_diffs (torch.Tensor): (B, M, ) finger diffs in world frame  (scalar distances)
            pos_approach_dirs (torch.Tensor): (B, M, 3) approach directions in world frame (origin centered vectors?)
        """
        label_config = self.global_config['DATA']['labels']

        nsample = label_config['k']  # Currently set to 1
        radius = label_config['max_radius']
        filter_z = label_config['filter_z']
        z_val = label_config['z_val']

        _, N, _ = processed_pc_cams.shape
        B, M, _ = pos_contact_points.shape

        # -- Make sure pcd is B x N x 3 -- #
        if processed_pc_cams.shape[2] != 3:
            xyz_cam = processed_pc_cams[:,:,:3]  # N x 3
        else:
            xyz_cam = processed_pc_cams

        # -- Transform Ground Truth to Camera Frame -- #
        # Transform contact points to camera frame  (This is a homogenous transform)
        # We use matmul to accommodate batch
        # pos_contact_points_cam = pos_contact_points @ (camera_poses[:3,:3].T) + camera_poses[:3,3][None,:]
        pos_contact_points_cam = torch.matmul(pos_contact_points, camera_poses[:, :3, :3].transpose(1, 2)) \
            + camera_poses[:,:3,3][:, None,:]

        # Transform contact directions to camera frame (Don't translate because its a direction vector)
        # pos_contact_dirs_cam = pos_contact_dirs @ camera_poses[:3,:3].T
        pos_contact_dirs_cam = torch.matmul(pos_contact_dirs, camera_poses[:, :3,:3].transpose(1, 2))
        
        # Make finger diffs B x M x 1
        pos_finger_diffs = pos_finger_diffs[:, :, None]

        # Transform approach directions to camera frame (Don't translate because its a direction vector)
        # pos_approach_dirs_cam = pos_approach_dirs @ camera_poses[:3,:3].T
        pos_approach_dirs_cam = torch.matmul(pos_approach_dirs, camera_poses[:, :3,:3].transpose(1, 2))

        # -- Filter Direction -- #
        # TODO: Figure out what is going on here
        if filter_z:
            # Filter out directions that are too far
            dir_filter_passed = (pos_contact_dirs_cam[:, :, 2:3] > z_val).repeat(1, 1, 3)
            pos_contact_points_cam = torch.where(dir_filter_passed, 
                                                 pos_contact_points_cam, 
                                                 torch.ones_like(pos_contact_points_cam) * 10000)
        
        # -- Compute Distances -- #
        # We want to compute the distance between each point in the point cloud and each contact point
        # We can do this by expanding the dimensions of the tensors and then summing the squared differences
        xyz_cam_expanded = torch.unsqueeze(xyz_cam, 2)  # B x N x 1 x 3
        pos_contact_points_cam_expanded = torch.unsqueeze(pos_contact_points_cam, 1)  # B x 1 x M x 3
        squared_dists_all = torch.sum((xyz_cam_expanded - pos_contact_points_cam_expanded)**2, dim=3)  # B x N x M

        # B x N x k, B x N x k
        squared_dists_k, close_contact_pt_idcs = torch.topk(squared_dists_all, 
            k=nsample, dim=2, largest=False, sorted=False)

        # -- Group labels -- #
        grouped_contact_dirs_cam = utils.index_points(pos_contact_dirs_cam, close_contact_pt_idcs)  # B x N x k x 3
        grouped_finger_diffs = utils.index_points(pos_finger_diffs, close_contact_pt_idcs)  # B x N x k x 1
        grouped_approach_dirs_cam = utils.index_points(pos_approach_dirs_cam, close_contact_pt_idcs)  # B x N x k x 3

        # grouped_contact_dirs_cam = pos_contact_dirs_cam[close_contact_pt_idcs, :]  # B x N x k x 3
        # grouped_finger_diffs = pos_finger_diffs[close_contact_pt_idcs]  # B x N x k x 1
        # grouped_approach_dirs_cam = pos_approach_dirs_cam[close_contact_pt_idcs, :]  # B x N x k x 3

        # -- Compute Labels -- #
        # Take mean over k nearest neighbors and normalize
        dir_label = grouped_contact_dirs_cam.mean(dim=2)  # B x N x 3
        dir_label = F.normalize(dir_label, p=2, dim=2)  # B x N x 3

        diff_label = grouped_finger_diffs.mean(dim=2)# B x N x 1

        approach_label = grouped_approach_dirs_cam.mean(dim=2)  # B x N x 3
        approach_label = F.normalize(approach_label, p=2, dim=2)  # B x N x 3

        grasp_success_label = torch.mean(squared_dists_k, dim=2, keepdim=True) < radius**2  # B x N x 1 
        grasp_success_label = grasp_success_label.type(torch.float32)  

        # debug = dict(
        #     xyz_cam = xyz_cam,
        #     pos_contact_points_cam = pos_contact_points_cam,
        # )
        debug = {}


        return dir_label, diff_label, grasp_success_label, approach_label, debug
