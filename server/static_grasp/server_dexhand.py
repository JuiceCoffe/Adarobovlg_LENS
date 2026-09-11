"""Serve language-guided static target grasps for right-handed dexterous hands."""

import sys
from pathlib import Path

import numpy as np

# Spawned processes inherit modified search paths; keep the project root first.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from dexgraspapi.grasp_mapping import prepare_hand_output
from server.common import SingleTaskGraspServerZMQ, create_server_parser, run_server


class TargetDexHandServerZMQ(SingleTaskGraspServerZMQ):
    """Target grasp inference for supported right hands."""

    # Service and coordinate conventions.
    task_type = "target_hand_grasp"
    default_port = 5556
    default_invert_obs_pose = True
    visualization_selection = "all"

    # Model proposal and final candidate limits are separate settings.
    default_mask_conf = 0.9
    model_num_cam = 4
    default_top_k = 10
    max_return_grasps = 20

    # Supported right-hand models.
    supported_hands = ("AllegroHand", "InspireHand", "DH3Hand")
    default_hand_name = "AllegroHand"

    # Collision and geometry offsets; positions and widths are in meters.
    hand_collision_thresh = 10.0
    max_depth_increment_cm = 4  # The model interface expresses this value in centimeters.
    pre_width_offset = 0.04
    final_width_offset = -0.07

    def __init__(self, *args, hand_name=default_hand_name, **kwargs):
        """Validate the default right-hand model before initializing shared service resources."""
        if hand_name not in self.supported_hands:
            raise ValueError(f"Unsupported hand_name={hand_name!r}")
        self.hand_name = hand_name
        super().__init__(*args, **kwargs)

    def run_task(self, request, debug=False):
        """Generate candidates, convert device poses, and filter by workspace."""
        hand_name, top_k = self._resolve_hand_options(request)
        hand_gg, gg = self.adarobovlg.forward(
            request.prompt,
            request.points,
            color=request.colors,
            obs=request.obs,
            flip_z=True,
            aug_z=False,
            top_k=top_k,
            debug=debug,
            collision_thresh=self.hand_collision_thresh,
            hand_name=hand_name,
            max_depth_increment_cm=self.max_depth_increment_cm,
            allowed_grasp_types=(1, 2, 3, 4, 5),
        )
        if hand_gg is None or gg is None or len(hand_gg) == 0 or len(gg) == 0:
            raise ValueError(f"No {self.task_type} grasps generated.")
        if len(hand_gg) != len(gg):
            raise ValueError("Hand and gripper grasp counts must match.")
        sort_idx = np.argsort(-hand_gg.scores)
        hand_gg, gg = hand_gg[sort_idx], gg[sort_idx]
        visual_hand_gg, reply_data = prepare_hand_output(
            hand_gg,
            gg,
            hand_name,
            self.pre_width_offset,
            self.final_width_offset,
        )
        reply_data.update(
            task_type=self.task_type,
            hand_name=hand_name,
            left_hand_mode=False,
            model_entrypoint="forward",
            invert_obs_pose=request.invert_obs_pose,
        )
        visual_hand_gg, reply_data = self._filter_grasp_workspace(visual_hand_gg, reply_data)
        return visual_hand_gg, reply_data, self.visualization_selection

    def _resolve_hand_options(self, request):
        """Validate right-hand selection and positive top_k, inheriting hand type during tracking."""
        messages = request.messages
        state = self._tracking_state if request.mode == "track" else None
        default = state["reply"]["hand_name"] if state else self.hand_name
        hand_name = messages.get("hand_name", default)
        if hand_name not in self.supported_hands:
            raise ValueError(
                f"Unsupported hand_name={hand_name!r}; choose from {self.supported_hands}."
            )
        for key in ("left_hand_mode", "left_hand"):
            value = messages.get(key, False)
            enabled = (
                value.strip().lower() in {"1", "true", "yes", "y", "on"}
                if isinstance(value, str)
                else bool(value)
            )
            if enabled:
                raise ValueError("Only right-hand mode is supported.")
        top_k = messages.get("top_k", self.default_top_k)
        if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k <= 0:
            raise ValueError("top_k must be positive.")
        return hand_name, top_k


def main():
    """Run the server dexhand command-line entrypoint."""
    parser = create_server_parser(TargetDexHandServerZMQ, "AdaRoboVLG static_grasp dexhand")
    parser.add_argument(
        "--hand",
        dest="hand_name",
        choices=TargetDexHandServerZMQ.supported_hands,
        default=TargetDexHandServerZMQ.default_hand_name,
        help="Right-hand model",
    )
    parser.add_argument(
        "--right-hand",
        action="store_true",
        help="Compatibility alias; only right hands are supported",
    )
    run_server(TargetDexHandServerZMQ, parser)


if __name__ == "__main__":
    main()
