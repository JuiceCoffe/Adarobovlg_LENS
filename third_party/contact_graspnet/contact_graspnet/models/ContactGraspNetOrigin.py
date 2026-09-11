import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..options.config_utils import load_config, load_config_origin
# import pointnet2_ops.pointnet2_modules as pointnet2
from timm.models.layers import DropPath, trunc_normal_
from .build import MODELS
from .losses import ContactGraspNetLoss, ContactGraspNetLossOrigin
                        
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'Pointnet_Pointnet2_pytorch'))
from Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation


@MODELS.register_module()
class ContactGraspNetOrigin(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.global_config = load_config_origin('base_model_param_cgnet_origin.yaml')
        self.model_config = self.global_config['MODEL']
        self.data_config = self.global_config['DATA']
        
        radius_list_0 = self.model_config['pointnet_sa_modules_msg'][0]['radius_list']
        radius_list_1 = self.model_config['pointnet_sa_modules_msg'][1]['radius_list']
        radius_list_2 = self.model_config['pointnet_sa_modules_msg'][2]['radius_list']

        nsample_list_0 = self.model_config['pointnet_sa_modules_msg'][0]['nsample_list']
        nsample_list_1 = self.model_config['pointnet_sa_modules_msg'][1]['nsample_list']
        nsample_list_2 = self.model_config['pointnet_sa_modules_msg'][2]['nsample_list']

        mlp_list_0 = self.model_config['pointnet_sa_modules_msg'][0]['mlp_list'] # list of lists
        mlp_list_1 = self.model_config['pointnet_sa_modules_msg'][1]['mlp_list']
        mlp_list_2 = self.model_config['pointnet_sa_modules_msg'][2]['mlp_list']

        npoint_0 = self.model_config['pointnet_sa_modules_msg'][0]['npoint']
        npoint_1 = self.model_config['pointnet_sa_modules_msg'][1]['npoint']
        npoint_2 = self.model_config['pointnet_sa_modules_msg'][2]['npoint']

        fp_mlp_0 = self.model_config['pointnet_fp_modules'][0]['mlp']
        fp_mlp_1 = self.model_config['pointnet_fp_modules'][1]['mlp']
        fp_mlp_2 = self.model_config['pointnet_fp_modules'][2]['mlp']

        sa_mlp = self.model_config['pointnet_sa_module']['mlp']
        sa_group_all = self.model_config['pointnet_sa_module']['group_all']

        self.input_normals = self.data_config['input_normals']
        self.offset_bins = self.data_config['labels']['offset_bins']
        self.joint_heads = self.model_config['joint_heads']
        
        additional_channel = 3 if self.input_normals else 0
        
        if 'asymmetric_model' in self.model_config and self.model_config['asymmetric_model']:
            # It looks like there are xyz_points and "points".  The points are normals I think?
            self.sa1 = PointNetSetAbstractionMsg(npoint=npoint_0,
                                                radius_list=radius_list_0,
                                                nsample_list=nsample_list_0,
                                                in_channel = 3 + additional_channel,
                                                mlp_list=mlp_list_0)

            # Sum the size of last layer of each mlp in mlp_list
            sa1_out_channels = sum([mlp_list_0[i][-1] for i in range(len(mlp_list_0))])
            self.sa2 = PointNetSetAbstractionMsg(npoint=npoint_1,
                                                radius_list=radius_list_1,
                                                nsample_list=nsample_list_1,
                                                in_channel=sa1_out_channels,
                                                mlp_list=mlp_list_1)

            sa2_out_channels = sum([mlp_list_1[i][-1] for i in range(len(mlp_list_1))])
            self.sa3 = PointNetSetAbstractionMsg(npoint=npoint_2,
                                                radius_list=radius_list_2,
                                                nsample_list=nsample_list_2,
                                                in_channel=sa2_out_channels,
                                                mlp_list=mlp_list_2)

            sa3_out_channels = sum([mlp_list_2[i][-1] for i in range(len(mlp_list_2))])
            self.sa4 = PointNetSetAbstraction(npoint=None,
                                                radius=None,
                                                nsample=None,
                                                in_channel=3 + sa3_out_channels,
                                                mlp=sa_mlp,
                                                group_all=sa_group_all)

            self.fp3 = PointNetFeaturePropagation(in_channel=sa_mlp[-1] + sa3_out_channels,
                                                mlp=fp_mlp_2)

            self.fp2 = PointNetFeaturePropagation(in_channel=fp_mlp_2[-1] + sa2_out_channels,
                                                mlp=fp_mlp_1)

            self.fp1 = PointNetFeaturePropagation(in_channel=fp_mlp_1[-1] + sa1_out_channels,
                                                mlp=fp_mlp_0)
        else:
            raise NotImplementedError

        if self.joint_heads:
            raise NotImplementedError
        else:
            # Theres prob some bugs here

            # Head for grasp direction
            self.grasp_dir_head = nn.Sequential(
                nn.Conv1d(fp_mlp_0[-1], 128, 1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3), # p = 1 - keep_prob  (tf is inverse of torch)
                nn.Conv1d(128, 3, 1, padding=0)
            )

            # Remember to normalize the output of this head

            # Head for grasp approach
            self.grasp_approach_head = nn.Sequential(
                nn.Conv1d(fp_mlp_0[-1], 128, 1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(128, 3, 1, padding=0)
            )

            # Head for grasp width
            if self.model_config['dir_vec_length_offset']:
                raise NotImplementedError
            elif self.model_config['bin_offsets']:
                self.grasp_offset_head = nn.Sequential(
                    nn.Conv1d(fp_mlp_0[-1], 128, 1, padding=0),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Conv1d(128, len(self.offset_bins) - 1, 1, padding=0)
                )
            else:
                raise NotImplementedError

            # Head for contact points
            self.binary_seg_head = nn.Sequential(
                nn.Conv1d(fp_mlp_0[-1], 128, 1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.5),  # 0.5 in original code
                nn.Conv1d(128, 1, 1, padding=0)
            )
        self.build_loss_func()
            
    def build_loss_func(self):
        self.loss_func = ContactGraspNetLossOrigin(self.global_config)
    
    def get_loss(self, pred, data, epoch, is_train=True):
        loss = self.loss_func(pred, data)
        return loss

    def get_bin_vals(self):
        """
        Creates bin values for grasping widths according to bounds defined in config

        Arguments:
            global_config {dict} -- config

        Returns:
            torch.tensor -- bin value tensor
        """
        bins_bounds = np.array(self.data_config['labels']['offset_bins'])
        if self.global_config['TEST']['bin_vals'] == 'max':
            bin_vals = (bins_bounds[1:] + bins_bounds[:-1])/2
            bin_vals[-1] = bins_bounds[-1]
        elif self.global_config['TEST']['bin_vals'] == 'mean':
            bin_vals = bins_bounds[1:]
        else:
            raise NotImplementedError

        if not self.global_config['TEST']['allow_zero_margin']:
            bin_vals = np.minimum(bin_vals, self.global_config['DATA']['gripper_width'] \
                                  -self.global_config['TEST']['extra_opening'])

        bin_vals = torch.tensor(bin_vals, dtype=torch.float32).cuda()
        return bin_vals
    
    def build_6d_grasp(self, approach_dirs, base_dirs, contact_pts, thickness, use_torch=False, gripper_depth = 0.1034):
        """
        Build 6-DoF grasps + width from point-wise network predictions

        Arguments:
            approach_dirs {np.ndarray/tf.tensor} -- Nx3 approach direction vectors
            base_dirs {np.ndarray/tf.tensor} -- Nx3 base direction vectors
            contact_pts {np.ndarray/tf.tensor} -- Nx3 contact points
            thickness {np.ndarray/tf.tensor} -- Nx1 grasp width

        Keyword Arguments:
            use_tf {bool} -- whether inputs and outputs are tf tensors (default: {False})
            gripper_depth {float} -- distance from gripper coordinate frame to gripper baseline in m (default: {0.1034})

        Returns:
            np.ndarray -- Nx4x4 grasp poses in camera coordinates
        """
        # We are trying to build a stack of 4x4 homogeneous transform matricies of size B x N x 4 x 4.
        # To do so, we calculate the rotation and translation portions according to the paper.
        # This gives us positions as shown:
        # [ R R R T ]
        # [ R R R T ]
        # [ R R R T ]
        # [ 0 0 0 1 ]                    Note that the ^ dim is 2 and the --> dim is 3
        # We need to pad with zeros and ones to get the final shape so we generate
        # ones and zeros and stack them.
        if use_torch:
            grasp_R = torch.stack([base_dirs, torch.cross(approach_dirs,base_dirs),approach_dirs], dim=3)  # B x N x 3 x 3
            grasp_t = contact_pts + (thickness / 2) * base_dirs - gripper_depth * approach_dirs  # B x N x 3
            grasp_t = grasp_t.unsqueeze(3)  # B x N x 3 x 1
            ones = torch.ones((contact_pts.shape[0], contact_pts.shape[1], 1, 1), dtype=torch.float32).cuda()  # B x N x 1 x 1
            zeros = torch.zeros((contact_pts.shape[0], contact_pts.shape[1], 1, 3), dtype=torch.float32).cuda()  # B x N x 1 x 3
            homog_vec = torch.cat([zeros, ones], dim=3)  # B x N x 1 x 4
            grasps = torch.cat([torch.cat([grasp_R, grasp_t], dim=3), homog_vec], dim=2)  # B x N x 4 x 4

        else:
            grasps = []
            for i in range(len(contact_pts)):
                grasp = np.eye(4)

                grasp[:3,0] = base_dirs[i] / np.linalg.norm(base_dirs[i])
                grasp[:3,2] = approach_dirs[i] / np.linalg.norm(approach_dirs[i])
                grasp_y = np.cross( grasp[:3,2],grasp[:3,0])
                grasp[:3,1] = grasp_y / np.linalg.norm(grasp_y)
                # base_gripper xyz = contact + thickness / 2 * baseline_dir - gripper_d * approach_dir
                grasp[:3,3] = contact_pts[i] + thickness[i] / 2 * grasp[:3,0] - gripper_depth * grasp[:3,2]
                # grasp[0,3] = finger_width
                grasps.append(grasp)
            grasps = np.array(grasps)

        return grasps
    
    def forward(self, point_cloud):
        if 'raw_num_points' in self.data_config and self.data_config['raw_num_points'] != self.data_config['ndataset_points']:
            raise NotImplementedError
        
        inds = np.random.choice(point_cloud.shape[1], 20000, replace=False)
        point_cloud = point_cloud[:, inds, :]
        # compute the mean
        mean = torch.mean(point_cloud, dim=1, keepdim=True)
        point_cloud -= mean

        point_cloud = torch.transpose(point_cloud, 1, 2) # Now we have batch x channels (3 or 6) x num_points
        l0_xyz = point_cloud[:, :3, :]
        l0_points = point_cloud[:, 3:6, :] if self.input_normals else l0_xyz.clone()
        
        # -- PointNet Backbone -- #
        # Set Abstraction Layers
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Feature Propagation Layers
        l3_points = self.fp3(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp1(l1_xyz, l2_xyz, l1_points, l2_points)

        l0_points = l1_points
        pred_points = l1_xyz
        
        # -- Heads -- #
        # Grasp Direction Head
        grasp_dir_head = self.grasp_dir_head(l0_points)
        grasp_dir_head_normed = F.normalize(grasp_dir_head, p=2, dim=1)  # normalize along channels

        # Grasp Approach Head
        approach_dir_head = self.grasp_approach_head(l0_points)
        
        # computer gram schmidt orthonormalization
        dot_product = torch.sum(grasp_dir_head_normed * approach_dir_head, dim=1, keepdim=True)
        projection = dot_product * grasp_dir_head_normed
        approach_dir_head_orthog = F.normalize(approach_dir_head - projection, p=2, dim=1)

        # Grasp Width Head
        if self.model_config['dir_vec_length_offset']:
            raise NotImplementedError
            grasp_offset_head = F.normalize(grasp_dir_head, p=2, dim=1)
        elif self.model_config['bin_offsets']:
            grasp_offset_head = self.grasp_offset_head(l0_points)

        # Binary Segmentation Head
        binary_seg_head = self.binary_seg_head(l0_points)
        
        # -- Construct Output -- #
        # Get 6 DoF grasp pose
        torch_bin_vals = self.get_bin_vals()
        # PyTorch equivalent of tf.gather_nd with conditional
        # I think the output should be B x N
        if self.model_config['bin_offsets']:
            argmax_indices = torch.argmax(grasp_offset_head, dim=1, keepdim=True)
            offset_bin_pred_vals = torch_bin_vals[argmax_indices]  # kinda sketch but works?
            # expand_dims_indices = argmax_indices.unsqueeze(1)
            # offset_bin_pred_vals = torch.gather(torch_bin_vals, 1, argmax_indices)
        else:
            offset_bin_pred_vals = grasp_offset_head[:, 0, :]
        
        # TODO Start here
        pred_grasps_cam = self.build_6d_grasp(approach_dir_head_orthog.permute(0, 2, 1),
                                              grasp_dir_head.permute(0, 2, 1),
                                              pred_points.permute(0, 2, 1),
                                              offset_bin_pred_vals.permute(0, 2, 1),
                                              use_torch=True)  # B x N x 4 x 4
        
        # Get pred scores
        pred_scores = torch.sigmoid(binary_seg_head).permute(0, 2, 1)

        # Get pred points
        pred_points = pred_points.permute(0, 2, 1)

        # Get pred offsets
        offset_pred = offset_bin_pred_vals
        
        pred = dict(
            pred_grasps_cam=pred_grasps_cam,
            pred_scores=pred_scores,
            pred_points=pred_points,
            offset_pred=offset_pred,
            grasp_offset_head=grasp_offset_head, # For loss,
            pc_mean=mean
        )
        
        return pred
    