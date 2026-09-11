"""Check LIBERO request validation, grasp conversion, and ZMQ compatibility.

Run from the repository root:
    python test/test_libero_server.py

The test uses deterministic model output and does not require weights, a GPU, LIBERO,
VLM access, or a display. It validates the legacy action-chunk interface rather than
simulator grasp success.
"""

import copy
import pickle
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.libero.server_gripper import LiberoGripperServerZMQ


class FakeGraspGroup:
    """Provide the aligned grasp fields used by the LIBERO adapter."""

    def __init__(self):
        """Create two candidates whose direction score reverses their initial order."""
        self.scores = np.array([0.9, 0.0], dtype=np.float64)
        self.depths = np.array([0.01, 0.03], dtype=np.float64)
        self.translations = np.array(
            [[0.4, 0.0, 0.1], [0.6, 0.0, 0.1]], dtype=np.float64
        )
        self.rotation_matrices = np.array(
            [np.diag([-1.0, 1.0, -1.0]), np.eye(3)], dtype=np.float64
        )

    def __len__(self):
        """Return the number of aligned grasp candidates."""
        return len(self.scores)

    def sort_by_score(self):
        """Sort every candidate field in descending score order."""
        order = np.argsort(self.scores)[::-1]
        for name in ("scores", "depths", "translations", "rotation_matrices"):
            setattr(self, name, getattr(self, name)[order])


def make_packet():
    """Build a valid two-view packet matching the AdaRoboVLG-Playground client."""
    points = np.array([[0.4, 0.0, 0.1], [0.6, 0.0, 0.1]], dtype=np.float32)
    world_from_robot = np.eye(4, dtype=np.float64)
    world_from_robot[:3, 3] = [0.1, -0.2, 0.05]
    return {
        "prompt": "bottle",
        "points": points,
        "colors": np.ones_like(points),
        "obs": {
            "color": np.zeros((2, 2, 3, 3), dtype=np.uint8),
            "depth": np.ones((2, 2, 3), dtype=np.float32),
            "K": np.repeat(np.eye(3)[None], 2, axis=0),
            "pose": np.repeat(np.eye(4)[None, :3], 2, axis=0),
            # The legacy field name is misleading; the client sends T_world_from_robot.
            "world2robot": world_from_robot,
        },
    }


def make_server(init_socket=False):
    """Create an adapter with deterministic grasp inference."""
    server = LiberoGripperServerZMQ(
        init_socket=init_socket,
        load_model=False,
        gen_box=False,
    )
    server.adarobovlg = SimpleNamespace(
        forward_gripper=lambda *args, **kwargs: copy.deepcopy(FakeGraspGroup())
    )
    return server


def check_action_chunk():
    """Verify ranking, transforms, phase lengths, and the legacy reply schema."""
    server = make_server()
    try:
        reply = server.run_request(make_packet())
    finally:
        server.stop()

    actions = np.asarray(reply["result"])
    assert actions.shape == (74, 7)
    assert actions.dtype == np.float32
    assert reply["mode"] == "abs"
    assert reply["debug"]["bbox"].shape == (2, 4)

    # The identity candidate wins after the legacy direction score is added.
    expected_grasp_xyz = np.array([0.52, 0.2, 0.05])
    expected_pre_xyz = np.array([0.42, 0.2, 0.05])
    expected_lift_xyz = np.array([0.52, 0.2, 0.25])
    expected_phases = (
        (slice(0, 20), np.array([*expected_pre_xyz[:2], 0.25])),
        (slice(20, 35), expected_pre_xyz),
        (slice(35, 54), expected_grasp_xyz),
        (slice(54, None), expected_lift_xyz),
    )
    for phase, expected in expected_phases:
        np.testing.assert_allclose(
            actions[phase, :3],
            np.repeat(expected[None], len(actions[phase]), axis=0),
            atol=1e-6,
        )
    assert np.all(actions[:50, 6] == 1.0)
    assert np.all(actions[50:, 6] == -1.0)
    np.testing.assert_allclose(
        actions[:, 3:6],
        np.repeat(actions[[0], 3:6], len(actions), axis=0),
        atol=1e-6,
    )


def check_validation_and_errors():
    """Reject malformed packets and serialize one compatible error reply."""
    server = make_server()
    try:
        packet = make_packet()
        malformed = [
            [],
            dict(packet, prompt=""),
            dict(packet, points=np.zeros((2, 4))),
            dict(packet, colors=np.zeros((1, 3))),
            dict(packet, obs={}),
        ]
        for value in malformed:
            try:
                server.run_request(value)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Accepted malformed LIBERO request: {type(value)}")

        sent = []
        server.socket = SimpleNamespace(send=sent.append, close=lambda: None)
        server.process_message(pickle.dumps({"prompt": "invalid"}))
        assert len(sent) == 1
        error = pickle.loads(sent[0])
        assert error["error_code"] == "request_failed" and "error" in error
    finally:
        server.stop()


def check_pyobj_network_exchange():
    """Exchange one request with send_pyobj/recv_pyobj over a ZMQ REQ/REP socket."""
    server = make_server(init_socket=True)
    server.address = f"inproc://libero-test-{id(server)}"
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()

    socket = server.context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    try:
        socket.connect(server.address)
        socket.send_pyobj(make_packet())
        reply = socket.recv_pyobj()
        assert np.asarray(reply["result"]).shape == (74, 7)
    finally:
        socket.close()
        server.running = False
        thread.join(timeout=2)
        if thread.is_alive():
            raise AssertionError("LIBERO server did not stop after its receive timeout")


def main():
    """Run all deterministic LIBERO adapter regression checks."""
    check_action_chunk()
    check_validation_and_errors()
    check_pyobj_network_exchange()
    print(
        "PASS: LIBERO validation, 74-step action conversion, errors, "
        "and ZMQ pyobj exchange"
    )


if __name__ == "__main__":
    main()
