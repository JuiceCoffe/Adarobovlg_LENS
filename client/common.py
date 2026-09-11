"""Shared packet I/O and synchronous transport; device options belong to entrypoints."""

import argparse
import math
import pickle
from pathlib import Path

import zmq


def create_parser(description, port):
    """Create a parser with arguments shared by all client entrypoints."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, required=True, help="Trusted RGB-D packet pickle")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--prompt", help="Override the saved target prompt")
    parser.add_argument("--obs-pose", choices=("camera-to-world", "world-to-camera"))
    parser.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="Reply wait in seconds; 0 waits indefinitely for operator selection",
    )
    parser.add_argument("--output", type=Path, help="Save reply pickle")
    parser.add_argument("--verbose", action="store_true", help="Print all reply fields")
    return parser


def add_tracking_arguments(parser):
    """Add initialization, session-ID, and ordered follow-up frame options to a client CLI."""
    parser.add_argument("--mode", choices=("init_track", "track"), default="init_track")
    parser.add_argument(
        "--next-data", type=Path, nargs="+", help="Ordered following frames after initialization"
    )
    parser.add_argument("--tracking-id", help="Required for a standalone track request")


def parse_args(parser):
    """Parse and validate command-line arguments before connecting."""
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout < 0:
        parser.error("--timeout must be finite and nonnegative")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if getattr(args, "mode", None) == "track":
        if not args.tracking_id:
            parser.error("--mode track requires --tracking-id")
        if args.next_data:
            parser.error("--next-data requires --mode init_track")
    return args


def load_request(args):
    """Load a trusted packet and apply explicit command-line overrides."""
    request = pickle.loads(args.data.expanduser().read_bytes())
    if not isinstance(request, dict):
        raise ValueError(f"Request packet must be a dictionary: {args.data}")
    request = dict(request)
    # Old recordings may contain a previous session or candidate selection.
    for key in ("mode", "track", "tracking_id", "grasp_index"):
        request.pop(key, None)
    if args.prompt is not None:
        request["prompt"] = args.prompt
    if args.obs_pose is not None:
        request["invert_obs_pose"] = args.obs_pose == "camera-to-world"
    return request


def exchange(socket, request):
    """Send one frame. Caller owns the socket and must close it after a timeout."""
    try:
        socket.send(pickle.dumps(request))
        reply = pickle.loads(socket.recv())
    except zmq.Again as error:
        raise TimeoutError(
            "Reply timed out; server may still be selecting/computing. "
            "Close the socket; do not automatically resend this frame."
        ) from error
    if not isinstance(reply, dict):
        raise ValueError("Server reply must be a dictionary")
    if "error" in reply:
        raise RuntimeError(
            f"{reply['error']} (requires_reinit={reply.get('requires_reinit', False)})"
        )
    for key in ("task_type", "mode", "translations", "rotations", "grasp_widths"):
        if key not in reply:
            raise ValueError(f"Server reply missing {key}")
    return reply


def run_client(args, build_request):
    """Replay packets in order with one connection and explicit request construction."""
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDTIMEO, 10000)
    socket.setsockopt(zmq.RCVTIMEO, -1 if args.timeout == 0 else max(1, int(args.timeout * 1000)))

    # Each request has one reply; the initial reply may await operator confirmation.
    try:
        socket.connect(f"tcp://{args.host}:{args.port}")
        previous = None
        paths = [args.data, *(getattr(args, "next_data", None) or [])]
        for index, path in enumerate(paths):
            frame_args = argparse.Namespace(**vars(args))
            frame_args.data = path
            request = build_request(frame_args)
            if previous is not None:
                request.update(mode="track", tracking_id=previous["tracking_id"])
                # These define the selected session, independent of saved frame metadata.
                request["prompt"] = initial_prompt
                request["invert_obs_pose"] = previous["invert_obs_pose"]
                if "hand_name" in previous:
                    request["hand_name"] = previous["hand_name"]
            else:
                initial_prompt = request.get("prompt")

            print(f"Sending {path} ({request['mode']})", flush=True)
            if request["mode"] != "track":
                print("Confirm a pose in the server terminal when prompted.", flush=True)
            reply = exchange(socket, request)
            if request["mode"] in {"init_track", "track"} and not reply.get("tracking_id"):
                raise ValueError("Dynamic reply missing tracking_id")
            previous = reply
            print(
                f"{reply['task_type']} / {reply['mode']}: {len(reply['translations'])} pose(s)",
                flush=True,
            )
            for key in ("grasp_index", "tracking_id", "frame_index"):
                if key in reply:
                    print(f"  {key}: {reply[key]}")
            if args.verbose:
                for key, value in reply.items():
                    print(f"  {key}: {value}")

            # Add the frame index so multi-frame output does not overwrite initialization.
            if args.output:
                output = args.output.expanduser()
                if len(paths) > 1:
                    output = output.with_name(f"{output.stem}_{index:03d}{output.suffix}")
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(pickle.dumps(reply))
                print(f"Saved: {output}", flush=True)
    finally:
        socket.close()
        context.term()
