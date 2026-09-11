"""Check parallel-jaw depth offsets and top-down orientation constraints.

Coverage includes one-time static offsets, preserved rotations and candidate order,
dynamic top-down rotations, reversed axes, and prevention of repeated offset accumulation.
Here, alignment refers to grasp-pose conventions rather than camera calibration.

Run from the repository root:
    python test/test_gripper_alignment.py

The check uses fixed arrays and requires project dependencies, but no recorded data,
weights, GPU inference, VLM, network service, or display. It prints PASS on success."""

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graspnetAPI import GraspGroup
from server.static_grasp.server_gripper import TargetGripperServerZMQ
from server.dynamic_grasp.server_gripper import TargetGripperServerZMQ as DynamicGripperServerZMQ


def check_alignment():
    """Check depth offsets, orientation constraints, and non-accumulating rigid updates."""
    server = TargetGripperServerZMQ(init_socket=False, load_model=False, start_visualizer=False)
    dynamic = DynamicGripperServerZMQ(init_socket=False, load_model=False, start_visualizer=False)
    try:
        group = GraspGroup(np.zeros((3, 17)))
        group.widths = np.full(3, 0.05)
        group.depths = np.array([0.01, 0.02, 0.03])
        group.scores = np.array([0.3, 0.2, 0.1])
        group.translations = np.repeat([[0.5, 0.0, -0.2]], 3, axis=0)
        group.rotation_matrices = np.array(
            [
                np.eye(3),
                [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
                [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
            ]
        )
        processed = server._postprocess_grasps(copy.deepcopy(group))
        np.testing.assert_array_equal(processed.rotation_matrices, group.rotation_matrices)
        np.testing.assert_array_equal(processed.scores, group.scores)
        np.testing.assert_allclose(
            processed.translations,
            group.translations + group.depths[:, None] * group.rotation_matrices[:, :, 0],
        )
        reply = {
            "rotations": processed.rotation_matrices.copy(),
            "translations": processed.translations.copy(),
        }
        dynamic._constrain_tracking_reply(reply, group.depths)
        np.testing.assert_allclose(
            reply["grasp_dirs"], np.repeat([[0, 0, -1]], 3, axis=0), atol=1e-7
        )
        np.testing.assert_allclose(
            reply["translations"], group.translations + group.depths[:, None] * [0, 0, -1]
        )
        np.testing.assert_allclose(np.linalg.det(reply["rotations"]), 1)
        once = copy.deepcopy(reply)
        dynamic._constrain_tracking_reply(reply, group.depths)
        np.testing.assert_allclose(reply["translations"], once["translations"])
        np.testing.assert_allclose(reply["rotations"], once["rotations"])
        # Rotating a raw grasp and constraining it must preserve the transformed raw center.
        Trot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        shift = np.array([0.1, 0.2, 0.3])
        transformed = {
            "rotations": Trot @ processed.rotation_matrices,
            "translations": processed.translations @ Trot.T + shift,
        }
        dynamic._constrain_tracking_reply(transformed, group.depths)
        np.testing.assert_allclose(
            transformed["translations"],
            group.translations @ Trot.T + shift + group.depths[:, None] * [0, 0, -1],
        )
        print(
            "PASS: static reference depth/rotation/order; dynamic top-down, antiparallel axes and no double depth bias"
        )
    finally:
        server.stop()
        dynamic.stop()


if __name__ == "__main__":
    check_alignment()
