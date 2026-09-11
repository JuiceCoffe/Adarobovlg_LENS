"""Serve language-guided dynamic target grasps for a parallel-jaw gripper."""

import sys
from pathlib import Path

import numpy as np

# Spawned processes inherit modified search paths; keep the project root first.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from server.common import SingleTaskGraspServerZMQ, create_server_parser, run_server
from server.dynamic_grasp.tracking import _run_task_with_tracking


class TargetGripperServerZMQ(SingleTaskGraspServerZMQ):
    # Service and coordinate conventions.
    task_type = "target_grasp"
    default_port = 5557
    default_invert_obs_pose = True
    tracking_service = True
    visualization_selection = "all"

    # Model proposal and final candidate limits are separate settings.
    default_mask_conf = 0.9
    default_top_k = 10
    max_return_grasps = 10

    # Collision and geometry offsets; positions and widths are in meters.
    collision_thresh = 0.05

    # Dynamic orientation constraint.
    top_down = False  # Follow object rotation unless --top-down is enabled.

    def run_request(self, request, debug=False):
        """Confirm one initial candidate and update it on subsequent frames."""
        return _run_task_with_tracking(self, request, debug=debug)

    def run_task(self, request, debug=False):
        """Generate candidates, convert device poses, and filter by workspace."""
        # Generate only during init_track; shared tracking updates later frames.
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

    def _constrain_tracking_reply(self, reply, depths):
        """Apply the optional top-down orientation constraint.

        When ``--top-down`` is enabled, align the approach axis with world -Z and
        recompute one depth offset from the original center to prevent accumulation.
        """
        for index, matrix in enumerate(reply["rotations"]):
            current_x = matrix[:, 0]
            target_x = np.array([0.0, 0.0, -1.0])
            center = reply["translations"][index] - depths[index] * current_x
            cross = np.cross(current_x, target_x)
            cosine = np.clip(np.dot(current_x, target_x), -1.0, 1.0)
            if cosine < -1.0 + 1e-8:
                axis = matrix[:, 1]
                align = 2 * np.outer(axis, axis) - np.eye(3)
            else:
                skew = np.array(
                    [[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]]
                )
                align = np.eye(3) + skew + skew @ skew / (1.0 + cosine)
            constrained = align @ matrix
            u, _, vt = np.linalg.svd(constrained)
            correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
            constrained = u @ correction @ vt
            if constrained[0, 2] < 0:
                constrained = constrained @ np.diag([1.0, -1.0, -1.0])
            reply["rotations"][index] = constrained
            reply["translations"][index] = center + depths[index] * constrained[:, 0]
        reply["grasp_dirs"] = reply["rotations"][:, :, 0].copy()
        reply["orientation_policy"] = "top_down"


def main():
    """Run the server gripper command-line entrypoint."""
    parser = create_server_parser(TargetGripperServerZMQ, "AdaRoboVLG dynamic_grasp gripper")
    orientation = parser.add_mutually_exclusive_group()
    orientation.add_argument(
        "--top-down",
        action="store_true",
        help="Opt in to the legacy world -Z approach constraint; default follows object rotation",
    )
    orientation.add_argument(
        "--free-orientation",
        action="store_true",
        help="Compatibility alias for the default orientation",
    )
    run_server(TargetGripperServerZMQ, parser)


if __name__ == "__main__":
    main()
