"""Check behaviors preserved by the repository cleanup.

Coverage includes right-hand depth offsets, width-to-joint mapping, array alignment,
input immutability, segmentation fallback order, NumPy JSON encoding, and unsupported
object errors. These are behavioral regressions, not style or dead-code checks.

Run from the repository root:
    python test/test_code_cleanup.py

Project dependencies and hand configurations are required, but recorded data, weights,
GPU inference, VLM, and display are not. Mock components replace model execution.
Split future checks by hand output, segmentation, or serialization when appropriate."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graspnetAPI import GraspGroup
from dexgraspapi import create_hand_graspgroup
from dexgraspapi.grasp_mapping import prepare_hand_output
from model import AdaRoboVLG
from utils.io import NumpyArrayEncoder


def check_hand_output():
    """Check metric depth, width-to-joint mapping, array alignment, and input immutability."""
    group = GraspGroup(np.zeros((2, 17)))
    group.widths = np.array([0.05, 0.06])
    group.depths = np.array([0.01, 0.02])
    group.heights = np.full(2, 0.02)
    group.translations = np.array([[0.5, 0.0, -0.2], [0.55, 0.0, -0.2]])
    group.rotation_matrices = np.array([np.eye(3), [[0, -1, 0], [1, 0, 0], [0, 0, 1]]])
    types = np.ones(2, dtype=int)
    for hand, dof in (("AllegroHand", 16), ("InspireHand", 6), ("DH3Hand", 2)):
        hand_group = create_hand_graspgroup(hand + "GraspGroup").from_graspgroup(
            copy.deepcopy(group), types
        )
        before = hand_group.mf_ggarray.copy()
        original = group.grasp_group_array.copy()
        visual, reply = prepare_hand_output(hand_group, group, hand, 0.04, -0.07)
        expected_positions = (
            hand_group.translations + hand_group.depths[:, None] * group.rotation_matrices[:, :, 0]
        )
        np.testing.assert_allclose(reply["translations"], expected_positions)
        np.testing.assert_array_equal(visual.translations, reply["translations"])
        np.testing.assert_array_equal(reply["grasp_widths"], group.widths)
        assert reply["hand_joint_pre_angles"].shape == (2, dof)
        for key, widths in [
            ("hand_joint_pre_angles", group.widths + 0.04),
            ("hand_joint_angles", np.zeros(2)),
        ]:
            expected = copy.deepcopy(group)
            expected.widths = widths
            expected = create_hand_graspgroup(hand + "GraspGroup").from_graspgroup(expected, types)
            np.testing.assert_allclose(reply[key], expected.angles)
        np.testing.assert_array_equal(group.grasp_group_array, original)
        np.testing.assert_array_equal(hand_group.mf_ggarray, before)


def check_segmentation_fallback():
    """Check prompt success, empty masks, and empty prompts use the correct fallback sequence."""
    points = np.array([[0.5, 0, -0.2]])
    images = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    for prompt, empty_mask in [("box", False), ("box", True), ("", True)]:
        model = AdaRoboVLG.__new__(AdaRoboVLG)
        calls = []

        def mask(*args, **kwargs):
            """Record language segmentation without loading SAM3."""
            calls.append("language")

        def project(*args, **kwargs):
            """Simulate the first projection and a successful point-prompt fallback."""
            calls.append("project")
            model.mask_3d = np.array([[int(not empty_mask or "point" in calls)]])

        def point_mask(*args, **kwargs):
            """Record the point-prompt fallback."""
            calls.append("point")

        model.generate_mask = mask
        model.mask_project = project
        model.generate_mask_with_3d_point = point_mask
        model.graspgen = SimpleNamespace(get_best_points=lambda *a, **k: points)
        model._segment_target(images, prompt, points, {}, None, False, False)
        expected = ["language", "project"] if prompt else []
        if empty_mask:
            expected += ["point", "project"]
        assert calls == expected, calls
        assert model.mask_3d[0, 0] == 1




def check_json_encoder():
    """Ensure NumPy values serialize and unsupported objects raise the standard TypeError."""
    result = json.loads(
        json.dumps({"array": np.array([1, 2]), "scalar": np.float32(0.5)}, cls=NumpyArrayEncoder)
    )
    assert result == {"array": [1, 2], "scalar": 0.5}
    try:
        json.dumps(object(), cls=NumpyArrayEncoder)
    except TypeError:
        pass
    else:
        raise AssertionError("Unsupported objects must raise TypeError")


if __name__ == "__main__":
    check_hand_output()
    check_segmentation_fallback()
    check_json_encoder()
    print("PASS: shared hand output, segmentation fallback, JSON encoding")
