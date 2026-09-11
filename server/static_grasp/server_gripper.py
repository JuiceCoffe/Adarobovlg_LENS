"""Serve language-guided static target grasps for a parallel-jaw gripper."""

import sys
from pathlib import Path

import numpy as np

# Spawned processes inherit modified search paths; keep the project root first.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from server.common import SingleTaskGraspServerZMQ, create_server_parser, run_server


class TargetGripperServerZMQ(SingleTaskGraspServerZMQ):
    # Service and coordinate conventions.
    task_type = "target_grasp"
    default_port = 5555
    default_invert_obs_pose = True
    visualization_selection = "all"

    # Model proposal and final candidate limits are separate settings.
    default_mask_conf = 0.1
    default_top_k = 10
    max_return_grasps = 10

    # Collision and geometry offsets; positions and widths are in meters.
    collision_thresh = 0.05

    def run_task(self, request, debug=False):
        """Generate candidates, convert device poses, and filter by workspace."""
        gg = self.adarobovlg.forward_gripper(
            request.prompt,
            request.points,
            color=request.colors,
            obs=request.obs,
            flip_z=True,
            top_k=self.default_top_k,
            debug=debug,
            collision_thresh=self.collision_thresh,
        )

        gg = self._postprocess_grasps(gg)
        reply_data = self._build_reply_data(gg, request)
        gg, reply_data = self._filter_grasp_workspace(gg, reply_data)
        return gg, reply_data, self.visualization_selection

    def _postprocess_grasps(self, gg):
        """Apply device-specific ranking and one depth offset to generated gripper poses in place."""
        if gg is None or len(gg) == 0:
            raise ValueError(f"No {self.task_type} grasps generated.")

        if self.tracking_service:
            gg.scores += abs(gg.rotation_matrices[:, 0, 2]) * 0.7
            gg.sort_by_score()

        # Apply the depth offset once along the local grasp x-axis.
        depth_bias = gg.depths.reshape(-1, 1) * gg.rotation_matrices[:, :, 0]
        gg.translations = gg.translations + depth_bias
        return gg


def main():
    """Run the server gripper command-line entrypoint."""
    parser = create_server_parser(TargetGripperServerZMQ, "AdaRoboVLG static_grasp gripper")
    run_server(TargetGripperServerZMQ, parser)


if __name__ == "__main__":
    main()
