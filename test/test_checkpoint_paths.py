"""Check checkpoint overrides and static projection independence from tracking.

Run from the repository root with ``python test/test_checkpoint_paths.py``.
This check requires no GPU, weights, or network; validate real weight loading separately.
"""

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.checkpoints import ROOT, checkpoint_path
from model.feat_projector import FeatProjector


def check_paths():
    """Check root/individual overrides, cwd independence, and actionable missing-file errors."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        weight = root / "model.pth"
        weight.touch()
        with patch.dict(os.environ, {"ADAROBOVLG_CHECKPOINTS_DIR": directory}):
            assert checkpoint_path("model.pth") == weight
            assert checkpoint_path(".", directory=True) == root
            with patch.dict(os.environ, {"ADAROBOVLG_TEST_WEIGHT": str(weight)}):
                assert checkpoint_path("missing.pth", "ADAROBOVLG_TEST_WEIGHT") == weight
            try:
                checkpoint_path("missing.pth")
            except FileNotFoundError as error:
                assert str(root / "missing.pth") in str(error)
                assert "checkpoints/README.md" in str(error)
            else:
                raise AssertionError("Missing weight accepted")
        with patch.dict(os.environ, {"ADAROBOVLG_CHECKPOINTS_DIR": "checkpoints"}):
            assert checkpoint_path(".", directory=True) == ROOT / "checkpoints"


def check_static_projection():
    """Ensure static projection neither requires tracking files nor constructs DINOv2."""
    with tempfile.TemporaryDirectory() as directory:
        with patch.dict(os.environ, {"ADAROBOVLG_CHECKPOINTS_DIR": directory}):
            with patch(
                "model.feat_projector.Fusion", return_value=SimpleNamespace(device="cpu")
            ) as fusion:
                FeatProjector(num_cam=4, track=False)
                fusion.assert_called_once_with(4, feat_backbone=None, just_mask=True)


if __name__ == "__main__":
    check_paths()
    check_static_projection()
    print("PASS: checkpoint overrides, missing assets and static projection independence")
