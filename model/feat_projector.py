"""Multi-view mask projection and initial-frame feature tracking."""

from d3fields.fusion import Fusion
import numpy as np
import torch


def fps_torch(points, num_samples):
    """
    points: torch.Tensor of shape [num_pts, 3]
    num_samples: int, number of points to sample
    return: index of sampled points, shape [num_samples]
    """
    device = points.device
    num_pts = points.shape[0]
    if num_pts <= num_samples:
        return torch.arange(num_pts, device=device)
    else:
        sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=device)
        sampled_indices[0] = torch.randint(0, num_pts, (1,), device=device)
        distances = torch.full((num_pts,), float("inf"), device=device)
        for i in range(1, num_samples):
            last_sampled_point = points[sampled_indices[i - 1]]
            dist_to_last_sampled = torch.norm(points - last_sampled_point, dim=1)
            distances = torch.minimum(distances, dist_to_last_sampled)
            sampled_indices[i] = torch.argmax(distances)
        return sampled_indices


class FeatProjector:
    def __init__(self, num_cam, track=False, track_points_num=100):
        """Configure multi-view mask fusion and the optional feature-tracking point budget."""
        from utils.checkpoints import ROOT, checkpoint_path

        options = {}
        if track:
            options = {
                "dinov2_repo": str(ROOT / "third_party" / "dinov2"),
                "dinov2_checkpoint": str(checkpoint_path("dinov2/dinov2_vitl14_pretrain.pth")),
                "xmem_checkpoint": str(checkpoint_path("xmem/XMem.pth")),
                "resnet18_checkpoint": str(checkpoint_path("xmem/resnet18-5c106cde.pth")),
                "resnet50_checkpoint": str(checkpoint_path("xmem/resnet50-19c8e357.pth")),
            }
        self.fusion = Fusion(
            num_cam, feat_backbone="dinov2" if track else None, just_mask=not track, **options
        )
        self.device = self.fusion.device
        self.num_cam = num_cam
        self.track = track
        self.track_points_num = track_points_num

    def update_mask(self, obs, mask):
        """Update fused masks (V, H, W, 1) using metric depth and world-to-camera observations."""
        self.fusion.update_mask(obs, mask)

    def get_3d_features(self, pts):
        """Return fused sam_mask features for a world-space point tensor (N, 3)."""
        return self.fusion.batch_eval(pts, return_names=["sam_mask"])

    def init_track(self, obs, obj_pts):
        """
        obs: dict with keys 'color', 'depth', 'K', 'pose'
        obj_pts: torch.Tensor of shape [num_pts, 3]
        """
        self.fusion.update(obs)
        self.fusion.curr_obs_torch["sam_mask"] = self.fusion.xmem_process(
            self.fusion.curr_obs_torch["color"], self.fusion.curr_obs_torch["sam_mask"][:, :, :, 0]
        ).to(dtype=self.fusion.dtype)[:, :, :, -1:]
        fps_idx = fps_torch(obj_pts, self.track_points_num)
        obj_pts_sampled = obj_pts[fps_idx].float()
        self.track_pts = obj_pts_sampled
        self.track_feats = self.fusion.batch_eval(obj_pts_sampled, return_names=["dino_feats"])[
            "dino_feats"
        ]
        self.T = np.eye(4, dtype=np.float32)

    def track_3d(self, new_obs):
        """
        new_obs: dict with keys 'color', 'depth', 'K', 'pose'
        """
        self.fusion.update(new_obs)
        self.fusion.curr_obs_torch["sam_mask"] = self.fusion.xmem_process(
            self.fusion.curr_obs_torch["color"], None
        ).to(dtype=self.fusion.dtype)[:, :, :, -1:]
        result = self.fusion.feature_tracking(
            self.track_feats.unsqueeze(0), self.track_pts.unsqueeze(0), track_mask=True
        )

        self.T = result["T"].T.detach().cpu().numpy()[:, :, 0]
        return self.T
