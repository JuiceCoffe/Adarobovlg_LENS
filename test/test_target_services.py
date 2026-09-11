"""Check static grasp services and provide an optional real-inference entrypoint.

Coverage includes parallel-jaw replies, Allegro/Inspire/DH3 right-hand mapping, invalid
arguments, and empty candidates. By default, in-memory observations and mock inference
avoid model weights, VLM calls, and display windows. Project dependencies and hand assets
are still required.

Run from the repository root:
    python test/test_target_services.py

Real VLM inference requires a GPU, weights, trusted request data, and DASHSCOPE_API_KEY:
    python test/test_target_services.py --data /path/to/request.pkl --prompt bottle --task both --box-prompt

Add ``--network`` to launch static-service subprocesses and verify local ZMQ exchange.
Using ``--data`` without ``--box-prompt`` selects local English text segmentation.
Packets must follow the protocol and default workspace; ``--obs-pose`` is unavailable.
The test prints PASS on success and does not validate robot execution."""

import argparse
import copy
import gc
import pickle
import socket as socket_module
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graspnetAPI import GraspGroup
from dexgraspapi import create_hand_graspgroup
from server.static_grasp.server_gripper import TargetGripperServerZMQ
from server.static_grasp.server_dexhand import TargetDexHandServerZMQ


def check_reply(reply, task):
    """Check candidate counts, finite arrays, and the device-specific reply shape."""
    count = len(reply["translations"])
    assert count > 0
    assert reply["task_type"] == task
    assert reply["rotations"].shape == (count, 3, 3)
    assert reply["translations"].shape == (count, 3)
    assert reply["grasp_widths"].shape == (count,)
    for key in ("rotations", "translations", "grasp_widths"):
        assert np.isfinite(reply[key]).all(), key
    if task == "target_hand_grasp":
        assert reply["left_hand_mode"] is False
        for key in ("hand_joint_pre_angles", "hand_joint_angles"):
            assert reply[key].shape[0] == count
            assert np.isfinite(reply[key]).all(), key


def run_regressions():
    """Exercise both devices, all right-hand mappings, invalid options, and empty results."""
    points = np.array([[0.5, 0.0, -0.2], [0.55, 0.0, -0.2]], dtype=np.float32)
    packet = {
        "points": points,
        "colors": np.ones_like(points),
        "prompt": "bottle",
        "obs": {
            "pose": np.eye(4)[None],
            "color": np.zeros((1, 2, 2, 3), dtype=np.uint8),
            "depth": np.ones((1, 2, 2)),
            "K": np.eye(3)[None],
        },
        "invert_obs_pose": True,
    }
    gg = GraspGroup()
    gg.grasp_group_array = np.zeros((2, 17))
    gg.scores = np.array([0.9, 0.7])
    gg.widths = np.array([0.05, 0.06])
    gg.heights = np.full(2, 0.02)
    gg.depths = np.full(2, 0.01)
    gg.rotation_matrices = np.repeat(np.eye(3)[None], 2, axis=0)
    gg.translations = points.copy()

    server = TargetGripperServerZMQ(init_socket=False, load_model=False, start_visualizer=False)
    try:
        request = server._prepare_request(pickle.dumps(packet))
        assert request.obs["pose"].shape == (1, 3, 4)
        server.adarobovlg = SimpleNamespace(forward_gripper=lambda *a, **k: copy.deepcopy(gg))
        _, reply, _ = server.run_task(request)
        check_reply(reply, server.task_type)
        assert reply["workspace_filtered_count"] == 0
        assert np.array_equal(packet["points"], points)
    finally:
        server.stop()

    for hand_name in TargetDexHandServerZMQ.supported_hands:
        server = TargetDexHandServerZMQ(
            hand_name=hand_name,
            init_socket=False,
            load_model=False,
            start_visualizer=False,
        )
        try:
            assert server.port == 5556 and server.mask_conf == 0.9
            request = server._prepare_request(pickle.dumps(packet))
            group = create_hand_graspgroup(hand_name + "GraspGroup").from_graspgroup(
                copy.deepcopy(gg),
                np.ones(2, dtype=int),
            )

            def infer(*args, **kwargs):
                """Capture hand-selection options and return deterministic matching candidate groups."""
                assert kwargs["hand_name"] == hand_name
                assert kwargs["allowed_grasp_types"] == (1, 2, 3, 4, 5)
                return copy.deepcopy(group), copy.deepcopy(gg)

            server.adarobovlg = SimpleNamespace(forward=infer)
            _, reply, _ = server.run_task(request)
            check_reply(reply, server.task_type)
            for option in (
                {"left_hand": True},
                {"left_hand_mode": "true"},
                {"hand_name": "RohandHand"},
                {"top_k": 0},
            ):
                invalid = copy.copy(request)
                invalid.messages = option
                try:
                    server.run_task(invalid)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"Accepted unsupported options: {option}")
            server.adarobovlg = SimpleNamespace(forward=lambda *a, **k: (None, None))
            try:
                server.run_task(request)
            except ValueError as error:
                assert "No target_hand_grasp" in str(error)
            else:
                raise AssertionError("Empty model result was accepted")
        finally:
            server.stop()
    print(
        "PASS: gripper response, all three right-hand mappings, invalid options and empty results."
    )


def run_inference(args):
    """Run real model inference against the explicitly supplied observation file."""
    import torch

    data_path = Path(args.data).expanduser()
    packet = pickle.loads(data_path.read_bytes())
    if args.prompt is not None:
        packet["prompt"] = args.prompt
    tasks = ("gripper", "hand") if args.task == "both" else (args.task,)
    for task in tasks:
        cls = TargetGripperServerZMQ if task == "gripper" else TargetDexHandServerZMQ
        options = {"hand_name": args.hand} if task == "hand" else {}
        server = cls(init_socket=False, start_visualizer=False, gen_box=args.box_prompt, **options)
        try:
            reply_bytes, _ = server._compute_reply_and_payload(pickle.dumps(packet))
            reply = pickle.loads(reply_bytes)
            check_reply(reply, server.task_type)
            print(f"PASS: {server.task_type}: {len(reply['translations'])} grasps")
        finally:
            server.stop()
            del server
            gc.collect()
            torch.cuda.empty_cache()


def run_network_inference(args):
    """Send real observations to the configured service and validate its response."""
    import zmq

    packet = pickle.loads(Path(args.data).expanduser().read_bytes())
    if args.prompt is not None:
        packet["prompt"] = args.prompt
    tasks = ("gripper", "hand") if args.task == "both" else (args.task,)
    for task in tasks:
        with socket_module.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        name = "target_grasp" if task == "gripper" else "target_hand_grasp"
        command = [
            sys.executable,
            str(
                ROOT
                / "server"
                / "static_grasp"
                / ("server_gripper.py" if task == "gripper" else "server_dexhand.py")
            ),
            "--no-visualizer",
            "--port",
            str(port),
        ]
        if not args.box_prompt:
            command.append("--text-prompt")
        if task == "hand":
            command.extend(["--hand", args.hand, "--right-hand"])
        process = subprocess.Popen(command, cwd=ROOT)
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 300000)
        try:
            socket.connect(f"tcp://127.0.0.1:{port}")
            socket.send(pickle.dumps(packet))
            reply = pickle.loads(socket.recv())
            assert "error" not in reply, reply
            check_reply(reply, name)
            print(f"PASS: ZMQ {name}: {len(reply['translations'])} grasps")
        finally:
            socket.close()
            context.term()
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        help="Trusted request pickle; omitted: regression checks without neural inference.",
    )
    parser.add_argument(
        "--prompt", help="Override packet prompt; use English for local text prompts."
    )
    parser.add_argument("--task", choices=("gripper", "hand", "both"), default="both")
    parser.add_argument(
        "--hand", choices=TargetDexHandServerZMQ.supported_hands, default="AllegroHand"
    )
    parser.add_argument(
        "--box-prompt", action="store_true", help="Use VLM boxes (requires DASHSCOPE_API_KEY)."
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="Start each service and verify a real ZMQ request/reply.",
    )
    args = parser.parse_args()
    if args.network:
        if not args.data:
            parser.error("--network requires --data")
        run_network_inference(args)
    elif args.data:
        run_inference(args)
    else:
        run_regressions()
