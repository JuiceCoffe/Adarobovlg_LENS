"""Check service interaction, error replies, and client request construction.

Coverage includes candidate preview, confirmation and cancellation, static automatic
replies, dynamic initialization, session inheritance, invalid requests, single error
replies, client overrides, and timeout cleanup. Models, displays, and sockets are mocked;
terminal input uses real pipe descriptors. Fixtures are shared with workspace tests.

Run from the repository root:
    python test/test_service_interaction.py

The Linux project environment and hand configurations are required, but recorded data,
weights, GPU inference, VLM, real networking, and display are not. Input is simulated and
temporary data is removed. This does not replace real ZMQ or Open3D validation."""

import argparse
import copy
import importlib
import pickle
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from graspnetAPI import GraspGroup
from dexgraspapi import create_hand_graspgroup
from server.common import SelectionCancelled


def fixture(server):
    """Attach deterministic inference/tracking to a service and return a valid observation packet."""
    points = np.array([[0.5, 0, -0.2], [0.55, 0, -0.2]])
    packet = dict(
        points=points,
        colors=np.ones_like(points),
        prompt="box",
        obs=dict(
            color=np.zeros((1, 2, 2, 3), dtype=np.uint8),
            depth=np.ones((1, 2, 2)),
            K=np.eye(3)[None],
            pose=np.eye(4)[None],
        ),
    )
    group = GraspGroup(np.zeros((2, 17)))
    group.scores = np.array([0.9, 0.7])
    group.widths = np.array([0.05, 0.06])
    group.heights = np.full(2, 0.02)
    group.depths = np.full(2, 0.01)
    group.rotation_matrices = np.repeat(np.eye(3)[None], 2, axis=0)
    group.translations = points.copy()

    def infer_hand(*args, **kwargs):
        """Return right-hand fixture candidates in the same order as their gripper candidates."""
        hand = create_hand_graspgroup(kwargs["hand_name"] + "GraspGroup").from_graspgroup(
            copy.deepcopy(group), np.ones(2, dtype=int)
        )
        return hand, copy.deepcopy(group)

    model = SimpleNamespace(
        forward_gripper=lambda *a, **k: copy.deepcopy(group),
        forward=infer_hand,
        mask_3d=None,
        obj_pts=points,
        obj_colors=np.ones_like(points),
    )
    transform = np.eye(4)
    transform[0, 3] = 0.01

    def prepare(obs):
        """Install the deterministic first-frame tracker on the model fixture."""
        model.track_featproj = SimpleNamespace(track_3d=lambda obs: transform.copy())

    model.prepare_grasp_tracking = prepare
    server.adarobovlg = model
    return packet


class FakeDisplay:
    closed = False

    def __init__(self):
        """Create an empty list of captured display updates."""
        self.frames = []

    def build_payload(self, points, colors, group, mask, reply):
        """Copy the reply into a fake display payload without constructing meshes."""
        return dict(points=points, reply=copy.deepcopy(reply), selected_pose="all")

    def show(self, payload):
        """Record a display update without opening a window."""
        self.frames.append(payload)

    def close(self):
        """Mark the fake display closed."""
        self.closed = True


def check_interactions():
    """Verify preview, confirmation, cancellation, and tracking behavior for all four services."""
    for dynamic in (False, True):
        for hand in (False, True):
            mode = "dynamic_grasp" if dynamic else "static_grasp"
            device = "dexhand" if hand else "gripper"
            module = importlib.import_module(f"server.{mode}.server_{device}")
            cls = getattr(module, "TargetDexHandServerZMQ" if hand else "TargetGripperServerZMQ")
            for visual in (False, True):
                server = cls(init_socket=False, load_model=False, start_visualizer=False)
                packet = fixture(server)
                if visual:
                    server.visualizer = FakeDisplay()
                commands = iter(["999", "all", "0", "1", "select 1"])
                reads = []

                def read(deadline):
                    """Return scripted selection commands while counting preview interactions."""
                    assert not sent, "Reply sent before operator confirmation"
                    command = next(commands)
                    reads.append(command)
                    return command

                server._read_selection_command = read
                sent = []
                server.socket = SimpleNamespace(send=sent.append, close=lambda: None)
                try:
                    server.process_message(pickle.dumps(packet))
                    assert len(sent) == 1
                    reply = pickle.loads(sent[0])
                    assert "error" not in reply, reply
                    selected = dynamic or visual
                    assert len(reply["translations"]) == (1 if selected else 2)
                    assert bool(reads) == selected
                    if selected:
                        assert reply["grasp_index"] == 1
                        assert reply["final_gripper_width"] == reply["grasp_widths"][0]
                        np.testing.assert_allclose(reply["grasp_widths"], [0.06])
                        if hand:
                            assert reply["hand_joint_angles"].shape[0] == 1
                    if dynamic:
                        track = dict(packet, mode="track", tracking_id=reply["tracking_id"])
                        server._read_selection_command = lambda deadline: (_ for _ in ()).throw(
                            AssertionError("Selected twice")
                        )
                        server.process_message(pickle.dumps(track))
                        updated = pickle.loads(sent[-1])
                        assert updated["frame_index"] == 1 and updated["grasp_index"] == 1
                        np.testing.assert_allclose(
                            updated["translations"], reply["translations"] + [0.01, 0, 0]
                        )
                        if visual:
                            np.testing.assert_allclose(
                                server.visualizer.frames[-1]["reply"]["translations"],
                                updated["translations"],
                            )
                    if visual:
                        assert [p["selected_pose"] for p in server.visualizer.frames[:4]] == [
                            "all",
                            "all",
                            0,
                            1,
                        ]
                    print(f"PASS: {mode}/{device}, visual={visual}")
                finally:
                    server.stop()


def check_errors():
    """Reject malformed requests and ensure display errors never produce duplicate replies."""
    from server.static_grasp.server_gripper import TargetGripperServerZMQ
    from server.dynamic_grasp.server_dexhand import TargetDexHandServerZMQ

    server = TargetGripperServerZMQ(init_socket=False, load_model=False, start_visualizer=False)
    packet = fixture(server)
    sent = []
    server.socket = SimpleNamespace(send=sent.append, close=lambda: None)
    try:
        malformed = [
            [],
            dict(packet, points=np.zeros((2, 4))),
            dict(packet, colors=np.zeros((1, 3))),
            dict(packet, obs={}),
            dict(packet, mode="init_track"),
        ]
        for value in malformed:
            sent.clear()
            server.process_message(pickle.dumps(value))
            assert len(sent) == 1 and "error" in pickle.loads(sent[0])
        server.visualizer = FakeDisplay()
        for command in ("cancel",):
            server._read_selection_command = lambda deadline: command
            sent.clear()
            server.process_message(pickle.dumps(packet))
            assert len(sent) == 1 and pickle.loads(sent[0])["error_code"] == "selection_cancelled"
        server._read_selection_command = lambda deadline: "select 0"
        calls = []

        def fail_after_selection(payload):
            """Simulate a visualization error after a candidate has already been confirmed."""
            calls.append(payload)
            if len(calls) > 1:
                raise RuntimeError("display failed after reply")

        server.visualizer.show = fail_after_selection
        sent.clear()
        server.process_message(pickle.dumps(packet))
        assert len(sent) == 1 and "error" not in pickle.loads(sent[0])
    finally:
        server.stop()
    # Overridden initial convention/hand are inherited when omitted by a track frame.
    server = TargetDexHandServerZMQ(init_socket=False, load_model=False, start_visualizer=False)
    packet = fixture(server)
    commands = iter(["select 0"])
    server._read_selection_command = lambda deadline: next(commands)
    packet.update(invert_obs_pose=False, hand_name="DH3Hand")
    try:
        initial = pickle.loads(server._compute_reply_and_payload(pickle.dumps(packet))[0])
        packet.pop("invert_obs_pose")
        packet.pop("hand_name")
        packet.update(mode="track", tracking_id=initial["tracking_id"])
        reply = pickle.loads(server._compute_reply_and_payload(pickle.dumps(packet))[0])
        assert reply["hand_name"] == "DH3Hand" and not reply["invert_obs_pose"]
        packet.update(mode="init_track")
        server._read_selection_command = lambda deadline: (_ for _ in ()).throw(
            SelectionCancelled("EOF")
        )
        try:
            server._compute_reply_and_payload(pickle.dumps(packet))
        except SelectionCancelled:
            pass
        else:
            raise AssertionError("EOF accepted")
        assert server._tracking_state is None
    finally:
        server.stop()
    print("PASS: invalid packets, cancellation, one reply on display failure, session inheritance")


def check_clients():
    """Check device options and explicit packet overrides in each client entrypoint."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "packet.pkl"
        path.write_bytes(
            pickle.dumps(dict(hand_name="InspireHand", grasp_index=0, tracking_id="old"))
        )
        args = argparse.Namespace(
            data=path, prompt=None, obs_pose=None, hand=None, mode="init_track", tracking_id=None
        )
        for mode in ("static_grasp", "dynamic_grasp"):
            gripper = importlib.import_module(f"client.{mode}.client_gripper")
            dexhand = importlib.import_module(f"client.{mode}.client_dexhand")
            assert "hand_name" not in gripper.build_request(args)
            request = dexhand.build_request(args)
            assert request["hand_name"] == "InspireHand"
            assert "grasp_index" not in request and "tracking_id" not in request
            args.hand = "DH3Hand"
            assert dexhand.build_request(args)["hand_name"] == "DH3Hand"
            args.hand = None
    print("PASS: client device separation, saved hand preserved, explicit hand override")


def check_terminal_and_timeout():
    """Check terminal selection through real pipes and client timeout cleanup with a fake socket."""
    import os
    import time
    from unittest.mock import patch

    import zmq
    from client.common import run_client
    from server.dynamic_grasp.server_gripper import TargetGripperServerZMQ

    server = TargetGripperServerZMQ(init_socket=False, load_model=False, start_visualizer=False)
    reader, writer = os.pipe()
    try:
        with os.fdopen(reader, "r") as terminal, patch("sys.stdin", terminal):
            os.write(writer, b"all\nselect 1\n")
            assert server._read_selection_command(None) == "all"
            assert server._read_selection_command(None) == "select 1"
            try:
                server._read_selection_command(time.monotonic() - 1)
            except SelectionCancelled:
                pass
            else:
                raise AssertionError("Expired selection accepted")
            os.close(writer)
            writer = None
            try:
                server._read_selection_command(None)
            except SelectionCancelled:
                pass
            else:
                raise AssertionError("EOF accepted")
        server.visualizer = FakeDisplay()
        server.visualizer.closed = True
        try:
            server._read_selection_command(None)
        except SelectionCancelled:
            pass
        else:
            raise AssertionError("Closed preview accepted")
    finally:
        if writer is not None:
            os.close(writer)
        server.stop()

    sent, closed, options = [], [], {}

    def receive():
        """Simulate a receive timeout by raising zmq.Again without network traffic."""
        raise zmq.Again()

    socket = SimpleNamespace(
        send=sent.append,
        recv=receive,
        connect=lambda address: None,
        setsockopt=lambda key, value: options.update({key: value}),
        close=lambda: closed.append("socket"),
    )
    context = SimpleNamespace(socket=lambda kind: socket, term=lambda: closed.append("context"))
    args = argparse.Namespace(
        host="127.0.0.1",
        port=5557,
        timeout=0,
        data=Path("unused"),
        output=None,
        verbose=False,
        next_data=None,
    )
    with patch("client.common.zmq.Context", return_value=context):
        try:
            run_client(args, lambda args: {"mode": "init_track"})
        except TimeoutError:
            pass
        else:
            raise AssertionError("Timeout not propagated")
    assert len(sent) == 1 and closed == ["socket", "context"]
    assert options[zmq.RCVTIMEO] == -1, "Operator wait must not expire after the old 300s default"
    print(
        "PASS: terminal preview/confirm, selection timeout, EOF, window close, client cleanup/no retry"
    )


if __name__ == "__main__":
    check_interactions()
    check_errors()
    check_clients()
    check_terminal_and_timeout()
