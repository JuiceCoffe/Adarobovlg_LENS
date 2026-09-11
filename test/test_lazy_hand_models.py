"""Verify lazy creation and caching of dexterous-hand models.

Run from the repository root with ``python test/test_lazy_hand_models.py``.
The test uses a fake factory and fresh subprocesses, so no weights, GPU, VLM, or hand
assets are required beyond the standard project dependencies.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_lazy_loading():
    """Check import side effects, per-hand caching, concurrent access, and retry after failure."""
    child = r"""
import importlib
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import robots.hands.hand_base as hand_base

with patch.object(hand_base, "create_hand_model", side_effect=lambda *a, **k: object()) as factory:
    for mode in ("static_grasp", "dynamic_grasp"):
        for device in ("gripper", "dexhand"):
            importlib.import_module(f"server.{mode}.server_{device}")
    factory.assert_not_called()

    from dexgraspapi import AllegroHandGrasp, InspireHandGrasp, DH3HandGrasp
    for index, cls in enumerate((AllegroHandGrasp, InspireHandGrasp, DH3HandGrasp), start=1):
        # First access can come from several threads; all must share one model.
        with ThreadPoolExecutor(max_workers=4) as pool:
            models = list(pool.map(lambda _: cls.mf_model, range(8)))
        assert all(model is models[0] for model in models)
        assert cls().mf_model is models[0]
        assert factory.call_count == index
        factory.assert_called_with(cls.__name__.removesuffix("Grasp"), use_cache=True)

    from dexgraspapi.multi_finger_grasp_base import _LazyHandModel
    class RetryGrasp:
        mf_model = _LazyHandModel("AllegroHand")

    expected = object()
    factory.side_effect = [RuntimeError("temporary asset failure"), expected]
    try:
        RetryGrasp.mf_model
    except RuntimeError:
        pass
    else:
        raise AssertionError("Initialization errors must propagate")
    assert RetryGrasp.mf_model is expected
    assert RetryGrasp.mf_model is expected
    assert factory.call_count == 5
print("PASS: service imports create no hand models; per-hand cache, concurrent access and retry")
"""
    result = subprocess.run(
        [sys.executable, "-c", child], cwd=ROOT, text=True, capture_output=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hand grasp taxonomy from cache" not in result.stdout + result.stderr
    print(result.stdout.strip())


if __name__ == "__main__":
    check_lazy_loading()
