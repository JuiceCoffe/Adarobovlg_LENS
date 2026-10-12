"""Language-conditioned grasp generation and optional object tracking."""

import time
from typing import Tuple

import cv2
import numpy as np
import open3d as o3d
import torch
from cprint import cprint
from d3fields.fusion import interpolate_feats, project_points_coords
from graspnetAPI import GraspGroup

from dexgraspapi.collision_detector import MultiFingerCollisionDetector
from dexgraspapi.grasp_mapping import MultiFingerGraspGenerator

from .feat_projector import FeatProjector
from .grasp_generator import GraspGenerator
from .mask_generator import MaskGenerator


class AdaRoboVLG:
    def __init__(self, mask_conf=0.1, track=False, num_cam=4, gen_box=False, mask_backend=None):
        """Initialize the segmentation, projection, grasp-generation, and tracking modules.

        Args:
            mask_conf: confidence threshold for mask generation
            track: whether to enable grasp tracking
            num_cam: number of camera views
            gen_box: whether to generate bounding box before mask generation (SAM with box prompt vs text prompt)
        """
        self.maskgen = MaskGenerator(
            confidence_thres=mask_conf, gen_box=gen_box, backend=mask_backend
        )
        self.featproj = FeatProjector(num_cam=num_cam, track=track)
        self.track_featproj = None
        self.graspgen = GraspGenerator()
        self.handdecision = None
        self.track = track
        self.masks = None
        self.prompt = None
        self.images = None
        self.tf_gg = None
        self.mf_gg = None
        self.last_tracking_T = None
        self.last_tracked_obj_pts = None
        self.last_tracked_obj_colors = None

    def generate_mask(self, images, prompt, debug=False):
        """Segment RGB images (V, H, W, 3) and cache the target masks (V, H, W)."""
        self.prompt = prompt
        self.images = images
        assert len(images.shape) == 4 and images.shape[3] == 3, "Image must be NxHxWx3"
        self.masks, boxes = self.maskgen.generate_mask(self.images, self.prompt)  # NxHxW
        if debug:
            for i in range(len(self.images)):
                img = self.images[i].copy()
                mask = self.masks[i]
                img[mask > 0.5] = [0, 255, 0]  # Highlight the mask in green.
                box = boxes[i]
                if box is not None and len(box) == 4:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(
                        img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2
                    )  # Draw the generated box in blue.
                cv2.imshow(f"Mask {i}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return self.masks

    def generate_mask_with_3d_point(self, images, points, obs, distance_thres=5e-3, debug=False):
        """Generate 2D masks by projecting 3D points as SAM point prompts.

        Args:
            images: Multi-view input images shaped (N, H, W, 3).
            points: One or more high-scoring 3D grasp points shaped (3,) or (M, 3).
            obs: Observations containing camera intrinsics, depths, and poses.

        Returns:
            Object masks shaped (N, H, W), one for each input image.
        """
        # Support both single-view and multi-view camera intrinsics.
        self.images = images
        points = torch.as_tensor(points, dtype=torch.float32).detach().cpu()
        if points.ndim == 1:
            points = points[None, :]  # (M, 3)
        num_images, H, W = images.shape[0], images.shape[1], images.shape[2]
        assert (
            num_images == obs["K"].shape[0]
        ), "Number of images and camera intrinsics batch size should match"
        pts_2d, valid_mask, pts_depth = project_points_coords(
            points, torch.tensor(obs["pose"]).float(), torch.tensor(obs["K"]).float()
        )
        pts_depth = pts_depth[..., 0]  # [rfn,pn]

        # get interpolated depth and features
        inter_depth = interpolate_feats(
            torch.tensor(obs["depth"]).unsqueeze(1),
            pts_2d,
            h=H,
            w=W,
            padding_mode="zeros",
            align_corners=True,
            inter_mode="nearest",
        )[
            ..., 0
        ]  # [rfn,pn,1]

        # compute the distance to the closest point on the surface
        dist = inter_depth - pts_depth  # [rfn,pn]
        valid = (valid_mask == True) & (dist.abs() < distance_thres)
        valid_pts_2d = []  # NxKx2
        for i in range(num_images):
            if valid[i].sum() == 0:
                valid_pts_2d.append([[]])
            else:
                valid_pts_2d.append(pts_2d[i][valid[i]].cpu().long().tolist())

        self.masks = self.maskgen.generate_mask_with_2d_points(self.images, valid_pts_2d)  # NxHxW
        if debug:
            for i in range(len(self.images)):
                img = self.images[i].copy()
                mask = self.masks[i]
                img[mask > 0.5] = [0, 255, 0]  # Highlight the mask in green.
                valid_pts = valid_pts_2d[i]
                for pt in valid_pts:
                    if len(pt) == 0:
                        continue
                    cv2.circle(img, (pt[0], pt[1]), 5, (255, 0, 0), -1)  # Mark valid 2D points in blue.
                cv2.imshow(f"Mask with 3D Point {i}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return self.masks

    def mask_project(self, pts, obs, color=None, debug=False):
        """Project cached masks onto world points (N, 3), remove outliers, and cache object points."""
        if self.masks is None:
            raise ValueError("Mask not generated yet. Please call generate_mask() first.")
        mask = (self.masks > 0.5).astype(np.uint8)
        self.featproj.update_mask(obs, mask[..., None])
        feats = self.featproj.get_3d_features(torch.from_numpy(pts).float().cuda())
        mask_3d = feats["sam_mask"].detach().cpu().numpy()
        # Remove statistical outliers from the projected mask.
        mask_idxs = np.where(mask_3d[:, 0] > 0.19)[0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.asarray(pts[mask_idxs, :], dtype=np.float64, order="C")
        )
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.3)
        self.mask_3d = np.zeros_like(mask_3d)
        self.mask_3d[mask_idxs[ind], 0] = 1.0

        obj_idx = (self.mask_3d > 0.19)[:, 0]
        self.obj_pts = pts[obj_idx]
        self.obj_colors = color[obj_idx] if color is not None else None

        if debug:
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(pts)
            if color is None:
                color = np.zeros_like(pts)
                color[:, 1] = 1.0
            color[np.where(self.mask_3d[:, 0] < 0.1)[0], :] *= 0.3
            scene_pcd.colors = o3d.utility.Vector3dVector(color)
            o3d.visualization.draw_geometries([scene_pcd])

        return self.mask_3d

    def prepare_grasp_tracking(self, obs):
        """Initialize tracking from cached first-frame masks and world points; no-op unless track is enabled."""
        if not self.track:
            return
        if self.obj_pts is None or len(self.obj_pts) == 0:
            raise ValueError("Object points are not available for tracking initialization")
        if self.masks is None:
            raise ValueError("Object masks are not available for tracking initialization")

        num_cam = (
            int(obs["color"].shape[0]) if obs is not None and obs.get("color") is not None else 1
        )
        masks = self.masks
        if masks.shape[0] < num_cam:
            raise ValueError(
                f"Tracking masks have {masks.shape[0]} views, but obs requires {num_cam}"
            )
        if masks.shape[0] != num_cam:
            masks = masks[-num_cam:]

        self.track_featproj = FeatProjector(num_cam=num_cam, track=True)
        self.track_featproj.update_mask(obs, masks[..., None])
        self.track_featproj.init_track(obs, torch.from_numpy(self.obj_pts).float().cuda())

    def generate_grasps(
        self,
        pts,
        color,
        valid_pts,
        top_k=5,
        collision_thresh=0.01,
        debug=False,
        flip_z=False,
        aug_z=False,
    ):
        """Generate world-frame GraspGroup candidates from metric points and RGB colors.

        valid_pts optionally restricts contacts; flip_z is undone on output.
        collision_thresh is passed to the network inference adapter.
        Optional aug_z combines a second orientation; caches and returns self.tf_gg."""
        if aug_z:
            # Augment candidates with a 180-degree rotation around the z-axis.
            rot_z = np.diag([-1, -1, 1])
            pts_aug = pts @ rot_z
            valid_pts_aug = valid_pts @ rot_z if valid_pts is not None else None
        self.tf_gg = self.graspgen.generate_grasps(
            pts,
            color=color,
            valid_pts=valid_pts,
            top_k=top_k,
            collision_thresh=collision_thresh,
            debug=debug,
            flip_z=flip_z,
        )
        if aug_z:
            tf_gg_aug = self.graspgen.generate_grasps(
                pts_aug,
                color=color,
                valid_pts=valid_pts_aug,
                top_k=top_k,
                collision_thresh=collision_thresh,
                debug=debug,
                flip_z=flip_z,
            )
            rot_back = np.diag([-1, -1, 1])
            tf_gg_aug.translations = (rot_back @ tf_gg_aug.translations[:, :, None]).squeeze(-1)
            tf_gg_aug.rotation_matrices = rot_back @ tf_gg_aug.rotation_matrices
            self.tf_gg.add(tf_gg_aug)
        return self.tf_gg



    def hand_mapping(
        self,
        pcd,
        obj_pcd=None,
        hand_name="AllegroHand",
        DEBUG=False,
        collision_thresh=100,
        max_depth_increment_cm=4,
        max_scene_points=2048,
        allowed_grasp_types=None,
    ):
        """Map cached two-finger grasps to right-hand poses and filter collisions.

        Args:
            pcd: Scene points (N, 3) in meters or an Open3D point cloud.
            obj_pcd: Optional object points used to adjust grasp centers.
            hand_name: AllegroHand, InspireHand, or DH3Hand.
            DEBUG: Whether to display intermediate geometry (legacy keyword).
            collision_thresh: Threshold passed to the hand collision detector.
            max_depth_increment_cm: Maximum candidate depth increment in centimeters.
            max_scene_points: Scene sampling budget for collision detection.
            allowed_grasp_types: Optional collection of accepted taxonomy IDs.

        Returns:
            Matched hand and two-finger groups, also cached on this instance.
            Hand-root translations do not yet include the grasp-depth offset."""
        if isinstance(pcd, o3d.geometry.PointCloud):
            pass
        elif isinstance(pcd, np.ndarray):
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
            pcd = pcd_o3d
        # Initialize Multi-Finger Grasp Generator
        if obj_pcd is not None and len(obj_pcd) > 0:
            mf_graspgen = MultiFingerGraspGenerator(
                pc_points=np.asarray(obj_pcd),
                tf_gg=self.tf_gg,
                hand_name=hand_name,
            )
        else:
            mf_graspgen = MultiFingerGraspGenerator(
                pc_points=np.asarray(pcd.points),
                tf_gg=self.tf_gg,
                hand_name=hand_name,
            )

        # Generate multi-finger grasps
        # NOTE: The multi-finger grasp translations are not adjusted by the grasp depth yet;
        #       this offset must be taken into account during actual execution.
        mf_gg, new_tf_gg = mf_graspgen.gen_multifinger_grasps(
            max_depth_increment_cm=max_depth_increment_cm,
            adjust_gripper_centers=True,  # adjust gripper center positions
        )

        valid_idx = np.ones(len(mf_gg), dtype=bool)
        if hand_name == "AllegroHand":
            valid_idx &= mf_gg.grasp_types != 6
        if allowed_grasp_types is not None:
            valid_idx &= np.isin(mf_gg.grasp_types, allowed_grasp_types)
        mf_gg = mf_gg[valid_idx]
        new_tf_gg = new_tf_gg[valid_idx]
        if len(mf_gg) == 0:
            self.mf_gg_filtered, self.tf_gg_filtered = mf_gg, new_tf_gg
            return mf_gg, new_tf_gg

        # Visualize the first generated multi-finger grasp for debugging
        if DEBUG:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            o3d.visualization.draw_geometries(
                [
                    pcd,
                    frame,
                    self.tf_gg[0].to_open3d_geometry(),
                    mf_gg[0].to_open3d_geometry(),
                ]
            )

        # Initalize Multi-Finger Collision Detector
        mf_cd = MultiFingerCollisionDetector(
            hand_name=hand_name,
            device="cpu",
            max_points=max_scene_points,
        )
        mf_cd.set_scene(
            pc_points=np.asarray(pcd.points),
            pc_colors=np.asarray(pcd.colors) if len(pcd.colors) > 0 else None,
            tf_gg=new_tf_gg,
            mf_gg=mf_gg,
        )

        # Compute collision detection time
        tm1 = time.time()
        res = mf_cd.detect(approach_dist=0.08, collision_thresh=collision_thresh)
        tm2 = time.time()
        cprint.info(f"Collision detection time: {tm2 - tm1:.4f} seconds")
        is_colliding_path = res["is_colliding_path"].cpu().numpy()  # B
        mf_gg_filtered = mf_gg[is_colliding_path == False]
        tf_gg_filtered = new_tf_gg[is_colliding_path == False]
        cprint.info(
            f"Number of grasps before filtering: {len(mf_gg)}, after filtering: {len(mf_gg_filtered)}"
        )

        if DEBUG:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            o3d.visualization.draw_geometries(
                [
                    pcd,
                    frame,
                    tf_gg_filtered[0].to_open3d_geometry(),
                    mf_gg_filtered[0].to_open3d_geometry(),
                ]
            )

        self.mf_gg_filtered = mf_gg_filtered
        self.tf_gg_filtered = tf_gg_filtered

        return self.mf_gg_filtered, self.tf_gg_filtered

    def hand_decision(self, pts, camera_pos, hand_name="AllegroHand"):
        """Score and reorder cached hand/gripper groups using object normals oriented toward camera_pos."""
        if self.handdecision is None:
            from .hand_decision import HandDecision

            from utils.checkpoints import checkpoint_path

            checkpoint = checkpoint_path(
                "hand_decision/hand_decision_model_8145.pth", "ADAROBOVLG_HAND_CHECKPOINT"
            )
            self.handdecision = HandDecision(checkpoint_path=str(checkpoint))
        # estimate normals
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        if isinstance(camera_pos, torch.Tensor):
            camera_pos = camera_pos.cpu().numpy()
        elif isinstance(camera_pos, list):
            camera_pos = np.array(camera_pos)
        else:
            pass
        pcd.orient_normals_towards_camera_location(camera_pos)
        normals = np.asarray(pcd.normals).astype(np.float32)
        self.mf_scores = self.handdecision.evaluate(
            hand_gg=self.mf_gg_filtered,
            tf_gg=self.tf_gg_filtered,
            obj_pts=pts,
            obj_normals=normals,
            hand_name=hand_name,
        )
        self.mf_gg_filtered.scores = self.mf_scores
        self.tf_gg_filtered.scores = self.mf_scores
        sort_idx = np.argsort(self.mf_gg_filtered.scores)[::-1]
        self.mf_gg_filtered = self.mf_gg_filtered[sort_idx]
        self.tf_gg_filtered = self.tf_gg_filtered[sort_idx]
        return self.mf_scores

    def _segment_target(self, images, prompt, pts, obs, color, flip_z, debug):
        """Project language masks, falling back to the best grasp point if empty.

        Images are RGB (V, H, W, 3); obs poses are world-to-camera.
        Updates the same mask and object-point state for both end effectors.
        """
        if prompt is None or prompt.strip() == "":
            self.mask_3d = np.zeros((pts.shape[0], 1), dtype=np.uint8)
        else:
            self.generate_mask(images, prompt, debug=debug)
            self.mask_project(pts, obs, color=color, debug=debug)

        if np.sum(self.mask_3d) == 0 and getattr(self.maskgen, "backend", "sam3") != "lens":
            best_points = self.graspgen.get_best_points(pts, top_k=1, flip_z=flip_z, debug=debug)
            self.generate_mask_with_3d_point(images, best_points, obs, debug=debug)
            self.mask_project(pts, obs, color=color, debug=debug)

    def forward(
        self,
        prompt,
        pts,
        color,
        obs,
        flip_z=False,
        aug_z=False,
        top_k=5,
        collision_thresh=10,
        max_depth_increment_cm=6,
        max_scene_points=2048,
        hand_name="AllegroHand",
        debug=False,
        allowed_grasp_types=None,
    ):
        """Generate scored right-hand grasps from language and multi-view observations.

        Args:
            prompt: Target description; an empty prompt enables grasp-point fallback.
            pts: World-frame scene points (N, 3), in meters.
            color: Point RGB colors (N, 3), in [0, 1].
            obs: BGR images, depth in meters, intrinsics, and world-to-camera poses.
            flip_z: Rotate points about X for the grasp network, then transform back.
            aug_z: Also generate candidates after a 180-degree Z rotation.
            top_k: Requested two-finger candidate count before mapping.
            collision_thresh: Hand collision threshold.
            max_depth_increment_cm: Additional hand-mapping depth in centimeters.
            max_scene_points: Scene sampling budget for collision detection.
            hand_name: Supported right-hand model name.
            debug: Display intermediate masks and geometry.
            allowed_grasp_types: Optional accepted taxonomy IDs.

        Returns:
            Aligned (hand_group, gripper_group), or (None, None) when mapping is empty.
            Updates cached masks, object points, candidates, and scores."""
        with torch.no_grad():
            images = obs["color"][:, :, :, ::-1]
            cam_pos = np.linalg.inv(np.vstack((obs["pose"][0][:3, :4], [0, 0, 0, 1])))[:3, 3]
            self._segment_target(images, prompt, pts, obs, color, flip_z, debug)

            start_time = time.time()
            # Restrict collision detection to points near the target object.
            obj_pts = pts[self.mask_3d[:, 0] > 0.19]
            no_obj = obj_pts.shape[0] == 0
            if len(obj_pts) == 0:
                cprint.warn("No valid object points after mask projection! Skipping hand decision.")
                obj_pts = pts.copy()
            x_min, x_max, y_min, y_max = (
                obj_pts[:, 0].min(),
                obj_pts[:, 0].max(),
                obj_pts[:, 1].min(),
                obj_pts[:, 1].max(),
            )
            thres = 0.05
            near_pts = pts[
                (pts[:, 0] > x_min - thres)
                & (pts[:, 0] < x_max + thres)
                & (pts[:, 1] > y_min - thres)
                & (pts[:, 1] < y_max + thres)
            ]
            self.generate_grasps(
                pts,
                color=color,
                valid_pts=None if no_obj else obj_pts,
                top_k=top_k,
                collision_thresh=0.0,
                debug=debug,
                flip_z=flip_z,
                aug_z=aug_z,
            )

            if no_obj:
                collision_thresh = 1  # Disable collision rejection without valid object points.
            mf_gg_filtered, tf_gg_filtered = self.hand_mapping(
                near_pts,
                obj_pts if not no_obj else None,
                hand_name,
                debug,
                collision_thresh=collision_thresh,
                max_depth_increment_cm=max_depth_increment_cm,
                max_scene_points=max_scene_points,
                allowed_grasp_types=allowed_grasp_types,
            )
            if len(mf_gg_filtered) == 0:
                cprint.warn("No valid grasps after hand mapping! Skipping hand decision.")
                self.mf_gg_filtered = self.tf_gg_filtered = None
                return None, None
            end_time = time.time()
            cprint.info(f"Total time for hand mapping: {end_time - start_time:.4f} seconds")
            self.hand_decision(obj_pts, cam_pos, hand_name)
            mf_gg, tf_gg = self.mf_gg_filtered, self.tf_gg_filtered
            return mf_gg, tf_gg


    def _adjust_gripper_centers(
        self,
        grasp_group: GraspGroup,
        targets: np.ndarray,
        heights: np.ndarray,
        depths: np.ndarray,
        widths: np.ndarray,
        width_scale: float = 1.7,  # note: for anydexgrasp, use 1.7
    ) -> Tuple[GraspGroup, np.ndarray]:
        """Adjust gripper centers based on points within the gripper fingers.

        Args:
            grasp_group (GraspGroup): input grasp group.
            targets (np.ndarray): points in grasp coordinate system, (G, N, 3).
            heights (np.ndarray): grasp heights, (G, 1).
            depths (np.ndarray): grasp depths, (G, 1).
            widths (np.ndarray): grasp widths, (G, 1).
            width_scale (float): scale factor for adjusting grasp widths.

        Returns:
            Tuple[GraspGroup, np.ndarray]: adjusted grasp group and targets.
        """
        tf_len = 0.03
        targets = np.array(targets, dtype=np.float32)
        ## get point masks
        # height mask
        mask1 = (targets[:, :, 2] > -heights / 2) & (targets[:, :, 2] < heights / 2)
        # left finger mask
        mask2 = (targets[:, :, 0] > depths - tf_len) & (targets[:, :, 0] < depths)
        mask4 = targets[:, :, 1] < -widths / 2
        # right finger mask
        mask6 = targets[:, :, 1] > widths / 2
        # get inner mask of each point
        inner_mask = mask1 & mask2 & (~mask4) & (~mask6)

        ## adjust targets and gripper centers
        # get point bounds
        targets_y = targets[:, :, 1].copy()
        targets_y[~inner_mask] = 0
        ymin = targets_y.min(axis=1)
        ymax = targets_y.max(axis=1)
        # get offsets
        offsets = np.zeros([targets.shape[0], 3], dtype=targets.dtype)
        offsets[:, 1] = (ymin + ymax) / 2
        # adjust targets
        targets[:, :, 1] -= offsets[:, np.newaxis, 1]
        # adjust gripper centers
        rots = grasp_group.rotation_matrices

        # update grasp widths and translations
        grasp_group.widths = np.maximum(0.025 * np.ones(ymax.shape), width_scale * (ymax - ymin))
        grasp_group.translations += np.matmul(rots, offsets[:, :, np.newaxis]).squeeze(2)
        return (grasp_group, targets)

    def adjust_gripper_center(self, obj_pcd, width_scale=1.5):
        """Adjust cached gripper centers and widths in place from object points (N, 3), in meters."""
        tf_gg_trans = self.tf_gg.translations
        tf_gg_rots = self.tf_gg.rotation_matrices
        tf_gg_heights = self.tf_gg.heights[:, np.newaxis]
        tf_gg_depths = self.tf_gg.depths[:, np.newaxis]
        tf_gg_widths = self.tf_gg.widths[:, np.newaxis]
        # points in grasp coordinate system
        target_pc = obj_pcd[np.newaxis, :, :] - tf_gg_trans[:, np.newaxis, :]
        target_pc = np.matmul(target_pc, tf_gg_rots)

        self.tf_gg, target_pc = self._adjust_gripper_centers(
            self.tf_gg,
            target_pc,
            tf_gg_heights,
            tf_gg_depths,
            tf_gg_widths,
            width_scale=width_scale,  # for multifinger grasp, use 1.0
        )
        return self.tf_gg

    def forward_gripper(
        self, prompt, pts, color, obs, flip_z=False, top_k=5, collision_thresh=0.05, debug=False
    ):
        """Generate two-finger grasps and adjust centers within the segmented object.

        Args:
            prompt: Target description, with grasp-point fallback when empty.
            pts: World-frame points (N, 3), in meters.
            color: Point RGB colors (N, 3), in [0, 1].
            obs: BGR images, metric depth, intrinsics, and world-to-camera poses.
            flip_z: Rotate network input about X and transform predictions back.
            top_k: Requested network candidate count.
            collision_thresh: Compatibility parameter; generation currently uses 0.0.
                This entry does not enable two-finger collision filtering.
            debug: Display intermediate masks and geometry.

        Returns:
            A world-frame GraspGroup before the service depth offset and workspace
            filter. Updates cached masks, object points, and two-finger candidates."""
        with torch.no_grad():
            images = obs["color"][:, :, :, ::-1]
            self._segment_target(images, prompt, pts, obs, color, flip_z, debug)

            no_obj = self.obj_pts.shape[0] == 0
            if len(self.obj_pts) == 0:
                cprint.warn("No valid object points after mask projection! Skipping hand decision.")
            tf_gg = self.generate_grasps(
                pts,
                color=color,
                valid_pts=None if no_obj else self.obj_pts,
                top_k=top_k,
                collision_thresh=0.0,
                debug=debug,
                flip_z=flip_z,
            )
            if not no_obj and tf_gg is not None and len(tf_gg):
                tf_gg = self.adjust_gripper_center(self.obj_pts, width_scale=1.5)

            return tf_gg
