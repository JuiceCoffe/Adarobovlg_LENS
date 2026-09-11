"""Check deterministic dynamic grasp sessions and pose updates.

Coverage includes initial candidate locking, transforms relative to the first frame,
non-accumulating offsets, preserved widths and joints, session validation, cleanup,
and reinitialization. Mock inference does not validate visual tracking accuracy.

Run from the repository root:
    python test/test_target_tracking.py

Project dependencies and hand configurations are required, but recorded data, weights,
GPU inference, VLM, networking, display, and manual input are not."""

import copy
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graspnetAPI import GraspGroup
from dexgraspapi import create_hand_graspgroup
from server.dynamic_grasp.server_gripper import TargetGripperServerZMQ
from server.dynamic_grasp.server_dexhand import TargetDexHandServerZMQ


def check_tracking(server_cls):
    """Exercise fixed-candidate tracking, session validation, and failure reset with fake inference."""
    server = server_cls(init_socket=False, load_model=False, start_visualizer=False)
    server._select_candidate = lambda request, group, reply: 1
    gg = GraspGroup(np.zeros((2, 17)))
    gg.scores = np.array([0.9, 0.7])
    gg.widths = np.array([0.05, 0.06])
    gg.heights = np.full(2, 0.02)
    gg.depths = np.full(2, 0.01)
    gg.rotation_matrices = np.repeat(np.eye(3)[None], 2, axis=0)
    gg.translations = np.array([[0.5, 0, -0.2], [0.55, 0, -0.2]])
    transform = np.eye(4)
    angle = 0.1
    transform[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]
    transform[0, 3] = 0.01
    calls = []

    def predict_gripper(*args, **kwargs):
        """Return deterministic gripper candidates and record an inference call."""
        calls.append("predict")
        return copy.deepcopy(gg)

    def predict_hand(*args, **kwargs):
        """Map deterministic gripper candidates to the hand fixture."""
        base = predict_gripper()
        hand = create_hand_graspgroup("AllegroHandGraspGroup").from_graspgroup(
            copy.deepcopy(base), np.ones(2, dtype=int)
        )
        return hand, base

    model = SimpleNamespace(
        forward_gripper=predict_gripper,
        forward=predict_hand,
        obj_pts=gg.translations.copy(),
        obj_colors=np.ones((2, 3)),
        mask_3d=None,
    )

    def prepare(obs):
        """Install the deterministic first-frame tracker on the model fixture."""
        model.track_featproj = SimpleNamespace(track_3d=lambda obs: transform.copy())

    model.prepare_grasp_tracking = prepare
    server.adarobovlg = model
    packet = {
        "points": gg.translations.copy(),
        "colors": np.ones((2, 3)),
        "prompt": "white box",
        "obs": {
            "pose": np.eye(4)[None],
            "color": np.zeros((1, 2, 2, 3), dtype=np.uint8),
            "depth": np.ones((1, 2, 2)),
            "K": np.eye(3)[None],
        },
    }

    def send(**options):
        """Prepare a fixture packet with explicit overrides and return its service response."""
        request = server._prepare_request(pickle.dumps(dict(packet, **options)))
        return server.run_request(request)[1]

    def rejected(**options):
        """Assert that the supplied request options fail validation."""
        try:
            send(**options)
        except ValueError:
            return
        raise AssertionError(f"Accepted invalid request: {options}")

    try:
        rejected(mode="track", tracking_id="missing")
        initial = send(mode="init_track", grasp_index=1)
        assert len(initial["translations"]) == 1 and initial["frame_index"] == 0
        np.testing.assert_allclose(initial["grasp_widths"], [0.06])
        session_id = initial["tracking_id"]
        rejected(mode="track", tracking_id="wrong")
        rejected(mode="track", tracking_id=session_id, prompt="different target")
        if server_cls is TargetDexHandServerZMQ:
            rejected(mode="track", tracking_id=session_id, left_hand=True)
            rejected(mode="track", tracking_id=session_id, hand_name="DH3Hand")
        for frame in (1, 2):
            reply = send(mode="track", tracking_id=session_id)
            assert reply["frame_index"] == frame
            assert calls == ["predict"], "Tracking must not rerun grasp generation"
            np.testing.assert_allclose(
                reply["translations"],
                initial["translations"] @ transform[:3, :3].T + transform[:3, 3],
            )
            np.testing.assert_allclose(reply["rotations"], transform[:3, :3] @ initial["rotations"])
            np.testing.assert_array_equal(reply["grasp_widths"], initial["grasp_widths"])
            if "hand_joint_angles" in reply:
                np.testing.assert_array_equal(
                    reply["hand_joint_angles"], initial["hand_joint_angles"]
                )
        transform[0, 0] = np.nan
        rejected(mode="track", tracking_id=session_id)
        assert server._tracking_state is None
        transform[:] = np.eye(4)
        initial = send(mode="init_track")
        rejected(mode="predict")
        replacement = send(mode="init_track")
        assert replacement["tracking_id"] != initial["tracking_id"]
        rejected(mode="track", tracking_id=initial["tracking_id"])
        server.tracking_service = False
        rejected(mode="init_track")
        rejected(mode="track", tracking_id=initial["tracking_id"])
        print(
            f"PASS: {server_cls.__name__}: selected candidate, rigid transform, no cumulative offsets, session validation and reset"
        )
    finally:
        server.stop()


if __name__ == "__main__":
    for cls in (TargetGripperServerZMQ, TargetDexHandServerZMQ):
        check_tracking(cls)
