"""Validate the packaged HandDecision network against a released checkpoint.

Run from the repository root with ``python test/test_hand_decision_package.py``.
This check requires the HandDecision runtime dependencies and checkpoint, but it does not
run sparse-convolution inference or require a GPU.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.hand_decision import HandDecision
from model.hand_decision.config import config
from model.hand_decision.network import HandDecisionModel
from utils.checkpoints import checkpoint_path


def check_package(checkpoint):
    """Check the public import and exact checkpoint-to-network key compatibility."""
    assert HandDecision.__name__ == "HandDecision"
    config.ptv3["pretrained"] = None
    model = HandDecisionModel(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = payload.get("model_state_dict", payload)
    incompatible = model.load_state_dict(state_dict, strict=False)
    assert not incompatible.missing_keys, incompatible.missing_keys
    assert not incompatible.unexpected_keys, incompatible.unexpected_keys


def parse_args():
    """Parse an optional checkpoint override."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to a HandDecision state-dict checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint = args.checkpoint or checkpoint_path(
        "hand_decision/hand_decision_model_8145.pth",
        "ADAROBOVLG_HAND_CHECKPOINT",
    )
    check_package(checkpoint)
    print("PASS: HandDecision package import and checkpoint compatibility")
