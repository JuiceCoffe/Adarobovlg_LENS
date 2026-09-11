"""Synchronous request handling and terminal candidate selection."""

import argparse
import copy
import os
import pickle
import select
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import zmq

from model.adarobovlg import AdaRoboVLG
from server.visualization import GraspVisualizer

ROOT = Path(__file__).resolve().parents[1]
# These fields align by candidate and must always be filtered or selected together.
CANDIDATE_FIELDS = (
    "rotations",
    "translations",
    "grasp_dirs",
    "approach_dirs",
    "grasp_widths",
    "hand_joint_pre_angles",
    "hand_joint_angles",
    "grasp_types",
)


@dataclass
class PreparedRequest:
    """Store a validated request in normalized world and camera coordinates."""

    messages: dict
    prompt: str
    points: np.ndarray
    colors: np.ndarray
    obs: dict
    invert_obs_pose: bool
    task_name: str
    mode: str


class SelectionCancelled(ValueError):
    """The operator cancelled or the selection interface became unavailable."""


class SingleTaskGraspServerZMQ:
    """One synchronous service; device subclasses own inference and CLI options."""

    task_type = "target_grasp"
    default_port = 5555
    default_invert_obs_pose = True
    default_top_k = 10
    default_mask_conf = 0.1
    model_num_cam = 3
    tracking_service = False
    visualization_selection = "all"

    def __init__(
        self,
        host="127.0.0.1",
        port=None,
        protocol="tcp",
        save_request_data=False,
        init_socket=True,
        start_visualizer=True,
        load_model=True,
        gen_box=True,
        mask_conf=None,
        selection_timeout=0,
        verbose=False,
    ):
        """Configure one synchronous service with optional socket, model, and visualizer initialization."""
        self.port = self.default_port if port is None else port
        self.address = f"{protocol}://{host}:{self.port}"
        self.gen_box = gen_box
        self.mask_conf = self.default_mask_conf if mask_conf is None else mask_conf
        if not np.isfinite(selection_timeout) or selection_timeout < 0:
            raise ValueError("selection_timeout must be finite and nonnegative")
        self.selection_timeout = selection_timeout
        self.verbose = verbose
        self.save_request_data = save_request_data
        self.server_data_root = ROOT / "data" / "server"

        # Input points and final grasp positions share this open workspace in meters.
        self.workspace_bounds = {
            "x": (0.30, 0.75),
            "y": (-0.25, 0.25),
            "z": (-0.30, 0.0),
        }

        self.context = zmq.Context() if init_socket else None
        self.socket = None
        self.socket_ready = False
        self.running = False

        # Each process stores one target session; visualization is independent of selection.
        self._tracking_state = None
        self.visualizer = GraspVisualizer() if start_visualizer else None
        self.adarobovlg = None
        if load_model:
            self._ensure_model()

    def start(self):
        """Bind the REP socket and handle one request at a time until shutdown."""
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 500)
        try:
            self.socket.bind(self.address)
            self.socket_ready = True
            self.running = True
            print(f"ZMQ Server started on {self.address}", flush=True)
            while self.running:
                try:
                    message = self.socket.recv()
                except zmq.Again:
                    continue
                self.process_message(message)
        finally:
            self.stop()

    def stop(self):
        """Close the visualizer, socket, and ZMQ context; safe to call repeatedly."""
        self.running = False
        if self.visualizer is not None:
            self.visualizer.close()
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None
        self.socket_ready = False

    def process_message(self, message):
        # Log receipt immediately to distinguish transport waits from later processing.
        """Send exactly one success/error reply for a trusted pickle request, then update display."""
        print(
            f"[{datetime.now().isoformat(timespec='milliseconds')}] "
            f"Received request: {self.task_type}, {len(message) / (1024 * 1024):.2f} MiB",
            flush=True,
        )
        # Compute one response before sending it. Display failures never send again.
        try:
            reply_bytes, payload = self._compute_reply_and_payload(message)
        except Exception as error:
            print(f"Request failed: {error}", flush=True)
            reply_bytes = pickle.dumps(
                {
                    "error": str(error),
                    "error_code": (
                        "selection_cancelled"
                        if isinstance(error, SelectionCancelled)
                        else "request_failed"
                    ),
                    "requires_reinit": self.tracking_service and self._tracking_state is None,
                }
            )
            payload = None
        self.socket.send(reply_bytes)
        if payload is not None:
            self._display(payload)

    def _compute_reply_and_payload(self, message, debug=False):
        """Validate and infer a request; return serialized reply bytes and optional display payload."""
        request = self._prepare_request(message)
        self._ensure_model()
        started = time.monotonic()
        gg, reply, _ = self.run_request(request, debug=debug)
        count = len(reply["translations"])
        for key in CANDIDATE_FIELDS:
            if key in reply:
                values = np.asarray(reply[key])
                if len(values) != count or not np.isfinite(values).all():
                    raise ValueError(f"Invalid reply field: {key}")
        print(
            f"{self.task_type} {request.mode}: {count} pose(s), "
            f"{time.monotonic() - started:.2f}s including selection",
            flush=True,
        )
        if self.verbose:
            self._log_reply_poses(reply, count)
        payload = self._preview_payload(request, gg, reply)
        return pickle.dumps(reply), payload

    def run_request(self, request, debug=False):
        """Run static inference and optionally return one explicitly confirmed candidate."""
        gg, reply, vis_top_k = self.run_task(request, debug=debug)
        reply["mode"] = "predict"
        if self.visualizer is not None:
            index = self._select_candidate(request, gg, reply)
            gg, reply = self._slice_candidate(gg, reply, index)
        return gg, reply, vis_top_k

    def run_task(self, request, debug=False):
        """Implement device inference; return (grasp_group, reply_dict, visualization_selection)."""
        raise NotImplementedError

    def _ensure_model(self):
        """Lazily construct and return the AdaRoboVLG inference instance."""
        if self.adarobovlg is None:
            self.adarobovlg = AdaRoboVLG(
                mask_conf=self.mask_conf,
                track=False,
                num_cam=self.model_num_cam,
                gen_box=self.gen_box,
            )
        return self.adarobovlg

    def _prepare_request(self, message):
        # Validate the mode and payload before coordinate conversion and workspace cropping.
        """Decode a trusted pickle packet into a validated PreparedRequest.

        Normalizes obs poses to world-to-camera (V, 3, 4), crops point/color arrays
        to the open workspace bounds, and optionally saves the original packet.
        Raises ValueError for invalid fields or an empty cropped cloud."""
        messages = pickle.loads(message)
        if not isinstance(messages, dict):
            raise ValueError("Request must be a dictionary")
        messages = dict(messages)
        mode = messages.get("mode")
        if mode is None:
            legacy = messages.get("track", False)
            enabled = (
                legacy.lower() in {"true", "1", "yes"} if isinstance(legacy, str) else bool(legacy)
            )
            mode = "track" if enabled else ("init_track" if self.tracking_service else "predict")
        allowed = {"init_track", "track"} if self.tracking_service else {"predict"}
        if mode not in allowed:
            raise ValueError(f"This service accepts {sorted(allowed)}, got {mode!r}")
        messages["mode"] = mode
        prompt = messages.get("prompt")
        if mode != "track" and (not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError("prompt must be a nonempty string")
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        points = np.asarray(messages.get("points"))
        colors = np.asarray(messages.get("colors"))
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
            raise ValueError("points must have shape (N, 3), N > 0")
        if colors.shape != points.shape:
            raise ValueError("colors must match points shape (N, 3)")
        if not np.isfinite(points).all() or not np.isfinite(colors).all():
            raise ValueError("points/colors must be finite")
        if np.any(colors < 0) or np.any(colors > 1):
            raise ValueError("colors must be in [0, 1]")

        # Images, depths, intrinsics, and extrinsics must share the same camera order.
        obs = messages.get("obs")
        if not isinstance(obs, dict):
            raise ValueError("obs must be a dictionary")
        obs = {**obs, **{key: np.asarray(obs.get(key)) for key in ("color", "depth", "K", "pose")}}
        color, depth, intrinsic, pose = (obs[key] for key in ("color", "depth", "K", "pose"))
        if (
            color.ndim != 4
            or color.shape[-1] != 3
            or min(color.shape[:3]) < 1
            or color.dtype != np.uint8
        ):
            raise ValueError("obs.color must be uint8 BGR (V, H, W, 3)")
        views = color.shape[0]
        if depth.shape != color.shape[:3] or not np.isfinite(depth).all() or np.any(depth < 0):
            raise ValueError(
                "obs.depth must match (V, H, W), finite nonnegative meters; use zero for invalid depth"
            )
        if intrinsic.shape != (views, 3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("obs.K must be finite (V, 3, 3)")
        if np.any(intrinsic[:, (0, 1), (0, 1)] <= 0):
            raise ValueError("obs.K focal lengths must be positive")
        if pose.shape not in ((views, 3, 4), (views, 4, 4)) or not np.isfinite(pose).all():
            raise ValueError("obs.pose must be finite (V, 3, 4) or (V, 4, 4)")
        if pose.shape[1] == 4 and not np.allclose(pose[:, 3], [0, 0, 0, 1]):
            raise ValueError("obs.pose has an invalid homogeneous last row")
        rotations = pose[:, :3, :3]
        if not np.allclose(
            rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-3
        ) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-3):
            raise ValueError("obs.pose must contain proper rigid rotations")

        # Accept both extrinsic conventions and normalize projection to world-to-camera.
        invert_pose = self._resolve_invert_obs_pose_for_task(messages)
        obs = self._invert_obs_pose(obs) if invert_pose else dict(obs, pose=pose[:, :3, :4].copy())
        points, colors = self._apply_workspace_filter(points, colors)
        if not len(points):
            raise ValueError("No points remain after workspace filtering")
        task_name = messages.get("task_name") or self.task_type
        if self.save_request_data:
            self._save_request_snapshot(messages, task_name)
        return PreparedRequest(messages, prompt, points, colors, obs, invert_pose, task_name, mode)

    def _resolve_invert_obs_pose_for_task(self, messages):
        """Resolve the external pose convention, inheriting the active session for track requests."""
        state = self._tracking_state if messages.get("mode") == "track" else None
        default = state["invert_obs_pose"] if state else self.default_invert_obs_pose
        value = messages.get("invert_obs_pose", default)
        if isinstance(value, str):
            if value.lower() not in {"true", "false", "1", "0"}:
                raise ValueError("invert_obs_pose must be a boolean")
            return value.lower() in {"true", "1"}
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError("invert_obs_pose must be a boolean")
        return bool(value)

    def _invert_obs_pose(self, obs):
        """Copy observations and invert camera-to-world poses into (V, 3, 4) projection matrices."""
        if obs is None or "pose" not in obs or obs["pose"] is None:
            return obs

        pose_array = np.asarray(obs["pose"], dtype=np.float64)
        if pose_array.ndim != 3 or pose_array.shape[1:] not in ((3, 4), (4, 4)):
            raise ValueError(
                f"obs['pose'] must have shape (N, 3, 4) or (N, 4, 4), got {pose_array.shape}"
            )

        inverse_poses = []
        for pose in pose_array:
            if pose.shape == (3, 4):
                pose_4x4 = np.eye(4, dtype=np.float64)
                pose_4x4[:3, :4] = pose
            else:
                pose_4x4 = pose
            inverse_poses.append(np.linalg.inv(pose_4x4)[:3, :4])

        inverted_obs = dict(obs)
        inverted_obs["pose"] = np.asarray(inverse_poses, dtype=pose_array.dtype)
        return inverted_obs

    def _apply_workspace_filter(self, points, colors):
        """Return corresponding points/colors strictly inside each world-space bound, in meters."""
        if points is None or len(points) == 0:
            return points, colors

        mask = np.ones(len(points), dtype=bool)
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            low, high = self.workspace_bounds[axis_name]
            mask &= (points[:, axis_index] > low) & (points[:, axis_index] < high)

        filtered_points = points[mask]
        filtered_colors = colors[mask] if colors is not None else None
        return filtered_points, filtered_colors

    def _grasp_workspace_mask(self, translations):
        """Check final gripper positions / hand roots in the input world frame."""
        positions = np.asarray(translations)
        valid = np.isfinite(positions).all(axis=1)
        for axis, name in enumerate(("x", "y", "z")):
            low, high = self.workspace_bounds[name]
            valid &= (positions[:, axis] > low) & (positions[:, axis] < high)
        return valid

    def _filter_grasp_workspace(self, group, reply):
        # Filter before limiting the count, so valid lower-ranked poses are retained.
        """Filter final poses before applying the candidate cap; keep all candidate arrays aligned."""
        valid = self._grasp_workspace_mask(reply["translations"])
        indices = np.flatnonzero(valid)[: self.max_return_grasps]
        if not len(indices):
            raise ValueError(f"No {self.task_type} grasps remain inside workspace_bounds")
        result = dict(reply)
        for key in CANDIDATE_FIELDS:
            if key in result:
                result[key] = np.asarray(result[key])[indices].copy()
        result["final_gripper_width"] = float(result["grasp_widths"][0])
        result["workspace_bounds"] = {
            key: list(value) for key, value in self.workspace_bounds.items()
        }
        result["workspace_filtered_count"] = int(len(valid) - np.count_nonzero(valid))
        return group[indices], result

    def _build_reply_data(self, gg, request):
        """Copy two-finger poses and widths into the response; local X is the world approach direction."""
        grasp_widths = gg.widths.copy()
        reply_data = {
            "rotations": gg.rotation_matrices.copy(),
            "translations": gg.translations.copy(),
            "grasp_dirs": gg.rotation_matrices[:, :, 0].copy(),
            "grasp_widths": grasp_widths,
            "final_gripper_width": float(grasp_widths[0]),
            "task_type": self.task_type,
            "invert_obs_pose": request.invert_obs_pose,
        }
        return reply_data

    def _slice_candidate(self, gg, reply, index):
        """Copy one confirmed candidate across all arrays and update its index and first width."""
        result = copy.deepcopy(reply)
        for key in CANDIDATE_FIELDS:
            if key in result:
                result[key] = np.asarray(result[key])[[index]].copy()
        result["grasp_index"] = int(index)
        result["final_gripper_width"] = float(result["grasp_widths"][0])
        return copy.deepcopy(gg[[index]]), result

    def _select_candidate(self, request, gg, reply):
        """Preview candidates until ``select <index>`` confirms one candidate."""
        count = len(gg)
        if not count:
            raise ValueError("No candidates available")
        payload = self._preview_payload(request, gg, reply)
        if self.visualizer is not None:
            if payload is None:
                raise SelectionCancelled(
                    "Cannot preview candidates; restart with --no-visualizer for terminal selection"
                )
            self._display(payload)
        self._log_reply_poses(reply, count)
        print("Preview: all or <index>; confirm: select <index>; cancel: cancel", flush=True)
        deadline = time.monotonic() + self.selection_timeout if self.selection_timeout else None
        while True:
            command = self._read_selection_command(deadline).strip().lower()
            if command in {"cancel", "q", "quit", "exit"}:
                raise SelectionCancelled("Selection cancelled")
            parts = command.split()
            confirmed = len(parts) == 2 and parts[0] == "select"
            value = parts[1] if confirmed else command
            if value == "all" and not confirmed:
                selected = "all"
            elif value.isdigit() and 0 <= int(value) < count:
                selected = int(value)
                if confirmed:
                    print(f"Selected pose {selected}", flush=True)
                    return selected
            else:
                print(f"Use all, 0-{count - 1}, select <index>, or cancel", flush=True)
                continue
            if payload is not None:
                self._display(dict(payload, selected_pose=selected))
            elif selected == "all":
                self._log_reply_poses(reply, count)
            else:
                self._log_reply_poses(
                    {
                        key: np.asarray(reply[key])[[selected]]
                        for key in ("translations", "rotations", "grasp_widths")
                    },
                    1,
                )

    def _read_selection_command(self, deadline):
        # One reader, no background input thread. Poll to notice window close/timeout.
        """Read one terminal line while polling cancellation, window close, and a monotonic deadline."""
        line = bytearray()
        while True:
            if self.visualizer is not None and self.visualizer.closed:
                raise SelectionCancelled("Preview window closed")
            if deadline is not None and time.monotonic() >= deadline:
                raise SelectionCancelled("Selection timed out")
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                char = os.read(sys.stdin.fileno(), 1)
                if not char:
                    raise SelectionCancelled("Terminal input ended (EOF); no pose selected")
                if char == b"\n":
                    return line.decode("utf-8", errors="replace")
                line.extend(char)

    def _log_reply_poses(self, reply_data, top_k=10):
        """Print up to top_k candidate positions, rotation matrices, and widths."""
        for index in range(min(top_k, len(reply_data["translations"]))):
            position = np.round(reply_data["translations"][index], 4).tolist()
            rotation = np.round(reply_data["rotations"][index], 3).tolist()
            width = float(reply_data["grasp_widths"][index])
            print(f"  [{index}] xyz={position}, width={width:.4f}, R={rotation}", flush=True)
        if "hand_name" in reply_data:
            print(f"  right hand: {reply_data['hand_name']}", flush=True)

    def _preview_payload(self, request, gg, reply):
        """Build an optional display payload; report geometry errors without changing inference results."""
        if self.visualizer is None or self.visualizer.closed:
            return None
        try:
            return self.visualizer.build_payload(
                request.points,
                request.colors,
                gg,
                getattr(self.adarobovlg, "mask_3d", None),
                reply,
            )
        except Exception as error:
            print(f"Visualization unavailable: {error}", flush=True)
            return None

    def _display(self, payload):
        """Update the optional window and report display failures without sending another reply."""
        try:
            self.visualizer.show(payload)
        except Exception as error:
            print(f"Visualization failed: {error}", flush=True)

    def _sanitize_task_name(self, task_name):
        """Replace unsafe task-name characters so request snapshots stay in one task directory."""
        task_name = str(task_name or "").strip().replace("\\", "_").replace("/", "_")
        task_name = task_name.replace(" ", "_")
        task_name = "".join(
            ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in task_name
        )
        task_name = task_name.strip("._")
        return task_name or "unknown_task"

    def _save_request_snapshot(self, messages, task_name):
        """Write the original trusted request under a timestamped task directory."""
        task_dir = self.server_data_root / self._sanitize_task_name(task_name)
        task_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        file_path = task_dir / f"{timestamp}.pkl"
        suffix = 1
        while file_path.exists():
            file_path = task_dir / f"{timestamp}-{suffix}.pkl"
            suffix += 1

        with open(file_path, "wb") as f:
            pickle.dump(messages, f)
        print(f"Saved request snapshot: {file_path}")
        return file_path


def create_server_parser(server_cls, description):
    """Create shared CLI options from the selected service class defaults."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind address. Keep the default for local use; binding to 0.0.0.0 is safe "
            "only on a trusted, isolated network because the pickle protocol has no "
            "authentication or encryption."
        ),
    )
    parser.add_argument("--port", type=int, default=server_cls.default_port)
    parser.add_argument("--save-request-data", action="store_true")
    parser.add_argument(
        "--no-visualizer",
        action="store_true",
        help="Disable the window; dynamic initialization still requires terminal selection",
    )
    parser.add_argument("--text-prompt", action="store_true", help="Use local SAM3 English prompts")
    parser.add_argument(
        "--selection-timeout",
        type=float,
        default=0,
        help="Operator selection timeout in seconds; 0 waits indefinitely",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--workspace-bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    return parser


def run_server(server_cls, parser):
    """Apply CLI overrides, install shutdown handlers, and run the selected service."""
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if not np.isfinite(args.selection_timeout) or args.selection_timeout < 0:
        parser.error("--selection-timeout must be finite and nonnegative")
    for name in ("workspace_bounds",):
        bounds = getattr(args, name, None)
        if bounds is not None and (
            not np.isfinite(bounds).all() or any(bounds[i] >= bounds[i + 1] for i in (0, 2, 4))
        ):
            parser.error(f"--{name.replace('_', '-')} requires finite min < max for each axis")
    device_options = {"hand_name": args.hand_name} if hasattr(args, "hand_name") else {}
    server = server_cls(
        host=args.host,
        port=args.port,
        save_request_data=args.save_request_data,
        start_visualizer=not args.no_visualizer,
        gen_box=not args.text_prompt,
        selection_timeout=args.selection_timeout,
        verbose=args.verbose,
        **device_options,
    )
    if args.workspace_bounds is not None:
        bounds = args.workspace_bounds
        server.workspace_bounds = dict(zip(("x", "y", "z"), zip(bounds[::2], bounds[1::2])))
    if hasattr(args, "top_down"):
        server.top_down = args.top_down

    def handle_shutdown_signal(signum, frame):
        """Stop the service when a process shutdown signal arrives."""
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
