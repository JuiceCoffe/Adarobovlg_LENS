"""Check final-pose workspace filtering across all four grasp services.

Coverage includes static and dynamic candidate filtering, exclusion of boundary points,
aligned reply arrays, empty-result errors, and dynamic-session cleanup after violations.
Fixtures from ``test_service_interaction`` provide simulated observations and results.

Run from the repository root:
    python test/test_workspace_bounds.py

Project dependencies and hand configurations are required, but recorded data, weights,
GPU inference, VLM, networking, and display are not. This does not validate real bounds."""

import importlib
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test.test_service_interaction import fixture


def check_workspace():
    """Check strict workspace boundaries, aligned candidate filtering, and out-of-bounds tracking reset."""
    for mode in ("static_grasp", "dynamic_grasp"):
        for device in ("gripper", "dexhand"):
            module = importlib.import_module(f"server.{mode}.server_{device}")
            cls = getattr(
                module,
                "TargetGripperServerZMQ" if device == "gripper" else "TargetDexHandServerZMQ",
            )
            server = cls(init_socket=False, start_visualizer=False, load_model=False)
            packet = fixture(server)
            server._select_candidate = lambda request, group, reply: 0
            try:
                request = server._prepare_request(pickle.dumps(packet))
                group, reply, _ = server.run_task(request)
                positions = reply["translations"]
                cutoff = float(np.mean(positions[:, 0]))
                server.workspace_bounds["x"] = (cutoff, 0.75)
                filtered, result = server._filter_grasp_workspace(group, reply)
                assert len(filtered) == 1 and result["workspace_filtered_count"] == 1
                np.testing.assert_allclose(result["translations"], positions[[1]])
                assert result["final_gripper_width"] == result["grasp_widths"][0]
                if device == "dexhand":
                    np.testing.assert_allclose(
                        result["hand_joint_angles"], reply["hand_joint_angles"][[1]]
                    )
                low, high = server.workspace_bounds["x"]
                probe = np.repeat(positions[[1]], 3, axis=0)
                probe[:, 0] = [low, (low + high) / 2, high]
                np.testing.assert_array_equal(
                    server._grasp_workspace_mask(probe), [False, True, False]
                )
                server.workspace_bounds["x"] = (0.30, 0.75)
                if mode == "dynamic_grasp":
                    initial = server.run_request(request)[1]
                    transform = np.eye(4)
                    transform[0, 3] = 10
                    server.adarobovlg.track_featproj.track_3d = lambda obs: transform
                    tracked = server._prepare_request(
                        pickle.dumps(dict(packet, mode="track", tracking_id=initial["tracking_id"]))
                    )
                    try:
                        server.run_request(tracked)
                    except ValueError as error:
                        assert "workspace_bounds" in str(error)
                    else:
                        raise AssertionError("Out-of-bounds tracking accepted")
                    assert (
                        server._tracking_state is None and server.adarobovlg.track_featproj is None
                    )
                server.workspace_bounds["x"] = (2, 3)
                try:
                    server._filter_grasp_workspace(group, reply)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Empty candidate set accepted")
                print(
                    f"PASS: {mode}/{device}: filtering, field alignment, boundary and tracking exit"
                )
            finally:
                server.stop()


if __name__ == "__main__":
    check_workspace()
