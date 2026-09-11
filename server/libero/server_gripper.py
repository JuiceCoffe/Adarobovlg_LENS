"""Serve AdaRoboVLG grasps as LIBERO Franka action chunks."""

import argparse
import pickle
import signal
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import zmq

# Keep this repository ahead of paths inherited from editable third-party packages.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))


class LiberoGripperServerZMQ:
    """Convert language-guided grasps into legacy LIBERO absolute actions."""

    default_port = 6666
    model_num_cam = 2
    default_mask_conf = 0.02
    default_top_k = 20

    # Convert a ContactGraspNet frame to the LIBERO Franka end-effector frame.
    _GRIPPER_TO_EEF = np.array(
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    _PHASE_LENGTHS = (20, 15, 15, 4, 20)

    def __init__(
        self,
        host="127.0.0.1",
        port=None,
        protocol="tcp",
        init_socket=True,
        load_model=True,
        gen_box=True,
        mask_conf=None,
        verbose=False,
    ):
        """Configure the synchronous LIBERO service and optionally load the model."""
        self.port = self.default_port if port is None else port
        self.address = f"{protocol}://{host}:{self.port}"
        self.gen_box = gen_box
        self.mask_conf = self.default_mask_conf if mask_conf is None else mask_conf
        self.verbose = verbose
        self.context = zmq.Context() if init_socket else None
        self.socket = None
        self.running = False
        self.adarobovlg = None
        if load_model:
            self._ensure_model()

    def _ensure_model(self):
        """Lazily construct the two-view AdaRoboVLG inference model."""
        if self.adarobovlg is None:
            from model.adarobovlg import AdaRoboVLG

            self.adarobovlg = AdaRoboVLG(
                mask_conf=self.mask_conf,
                track=False,
                num_cam=self.model_num_cam,
                gen_box=self.gen_box,
            )
        return self.adarobovlg

    def start(self):
        """Bind the REP socket and serve requests until shutdown."""
        if self.context is None:
            self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 500)
        self.socket.bind(self.address)
        self.running = True
        print(f"LIBERO ZMQ server started on {self.address}", flush=True)
        try:
            while self.running:
                try:
                    message = self.socket.recv()
                except zmq.Again:
                    continue
                self.process_message(message)
        finally:
            self.stop()

    def stop(self):
        """Close the socket and context; this method is safe to call repeatedly."""
        self.running = False
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None

    def process_message(self, message):
        """Process one trusted pickle request and send exactly one reply."""
        print(
            f"[{datetime.now().isoformat(timespec='milliseconds')}] "
            f"Received LIBERO request: {len(message) / (1024 * 1024):.2f} MiB",
            flush=True,
        )
        try:
            request = pickle.loads(message)
            reply = self.run_request(request)
        except Exception as error:
            print(f"LIBERO request failed: {error}", flush=True)
            reply = {
                "error": str(error),
                "error_code": "request_failed",
            }
        self.socket.send(pickle.dumps(reply))

    def run_request(self, request):
        """Validate one LIBERO packet, infer a grasp, and build its action chunk."""
        prompt, points, colors, obs = self._prepare_request(request)
        model = self._ensure_model()
        grasps = model.forward_gripper(
            prompt,
            points,
            color=colors,
            obs=obs,
            flip_z=True,
            top_k=self.default_top_k,
            debug=False,
            collision_thresh=0,
        )
        grasp_world = self._select_grasp_pose(grasps)
        actions = self._build_action_chunk(grasp_world, obs["world2robot"])
        if self.verbose:
            print(
                "Selected LIBERO grasp: "
                f"world_xyz={np.round(grasp_world[:3, 3], 5).tolist()}, "
                f"actions={len(actions)}",
                flush=True,
            )
        return {
            "result": [action.copy() for action in actions],
            "mode": "abs",
            "debug": {
                "bbox": np.array(
                    [[0, 0, 255, 255], [0, 0, 255, 255]], dtype=np.float32
                )
            },
        }

    def _prepare_request(self, request):
        """Validate and normalize fields used by the legacy LIBERO protocol."""
        if not isinstance(request, dict):
            raise ValueError("Request must be a dictionary")
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")

        points = np.asarray(request.get("points"))
        colors = np.asarray(request.get("colors"))
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
            raise ValueError("points must have shape (N, 3), N > 0")
        if colors.shape != points.shape:
            raise ValueError("colors must match points shape (N, 3)")
        if not np.isfinite(points).all() or not np.isfinite(colors).all():
            raise ValueError("points/colors must be finite")
        if np.any(colors < 0) or np.any(colors > 1):
            raise ValueError("colors must be in [0, 1]")

        raw_obs = request.get("obs")
        if not isinstance(raw_obs, dict):
            raise ValueError("obs must be a dictionary")
        required = ("color", "depth", "K", "pose", "world2robot")
        if any(raw_obs.get(key) is None for key in required):
            raise ValueError(f"obs must contain {required}")
        obs = {key: np.asarray(raw_obs[key]) for key in required}
        image, depth, intrinsic, pose = (
            obs["color"],
            obs["depth"],
            obs["K"],
            obs["pose"],
        )
        if (
            image.ndim != 4
            or image.shape[0] != self.model_num_cam
            or image.shape[-1] != 3
        ):
            raise ValueError("obs.color must have shape (2, H, W, 3)")
        if image.dtype != np.uint8:
            raise ValueError("obs.color must use uint8 values")
        if (
            depth.shape != image.shape[:3]
            or not np.isfinite(depth).all()
            or np.any(depth < 0)
        ):
            raise ValueError("obs.depth must be finite nonnegative meters with shape (2, H, W)")
        if intrinsic.shape != (self.model_num_cam, 3, 3) or not np.isfinite(
            intrinsic
        ).all():
            raise ValueError("obs.K must be finite with shape (2, 3, 3)")
        if np.any(intrinsic[:, (0, 1), (0, 1)] <= 0):
            raise ValueError("obs.K focal lengths must be positive")
        if pose.shape not in (
            (self.model_num_cam, 3, 4),
            (self.model_num_cam, 4, 4),
        ) or not np.isfinite(pose).all():
            raise ValueError(
                "obs.pose must be finite with shape (2, 3, 4) or (2, 4, 4)"
            )
        if pose.shape[1:] == (4, 4):
            if not np.allclose(pose[:, 3], [0, 0, 0, 1]):
                raise ValueError("obs.pose has an invalid homogeneous last row")
            obs["pose"] = pose[:, :3, :4].copy()
        self._validate_rotation_batch(obs["pose"][:, :3, :3], "obs.pose")

        world_from_robot = obs["world2robot"]
        if world_from_robot.shape != (4, 4) or not np.isfinite(
            world_from_robot
        ).all():
            raise ValueError("obs.world2robot must be a finite (4, 4) matrix")
        if not np.allclose(world_from_robot[3], [0, 0, 0, 1]):
            raise ValueError("obs.world2robot has an invalid homogeneous last row")
        self._validate_rotation_batch(
            world_from_robot[None, :3, :3], "obs.world2robot"
        )
        return prompt.strip(), points, colors, obs

    @staticmethod
    def _validate_rotation_batch(rotations, field_name):
        """Reject matrices that are not proper rigid rotations."""
        rotations = np.asarray(rotations)
        if not np.allclose(
            rotations.transpose(0, 2, 1) @ rotations,
            np.eye(3),
            atol=1e-3,
        ) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-3):
            raise ValueError(f"{field_name} must contain proper rigid rotations")

    def _select_grasp_pose(self, grasps):
        """Apply the legacy LIBERO ranking and return one world-frame EEF pose."""
        if grasps is None or len(grasps) == 0:
            raise ValueError("No LIBERO grasp generated")
        count = len(grasps)
        rotations = np.asarray(grasps.rotation_matrices)
        translations = np.asarray(grasps.translations)
        depths = np.asarray(grasps.depths)
        scores = np.asarray(grasps.scores)
        if rotations.shape != (count, 3, 3):
            raise ValueError("Generated grasp rotations have an invalid shape")
        if (
            translations.shape != (count, 3)
            or depths.shape != (count,)
            or scores.shape != (count,)
        ):
            raise ValueError("Generated grasp fields have invalid shapes")
        if not all(
            np.isfinite(values).all()
            for values in (rotations, translations, depths, scores)
        ):
            raise ValueError("Generated grasp fields must be finite")

        direction_score = np.clip(rotations[:, 0, 0], -1.0, 0.0)
        grasps.scores = scores + (direction_score + 1.0) ** 2
        grasps.sort_by_score()

        adjusted_depth = max(float(grasps.depths[0]) - 0.01, 0.0)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = (
            np.asarray(grasps.rotation_matrices[0]) @ self._GRIPPER_TO_EEF
        )
        pose[:3, 3] = np.asarray(grasps.translations[0]) + adjusted_depth * np.asarray(
            grasps.rotation_matrices[0]
        )[:, 0]
        return pose

    def _build_action_chunk(self, grasp_world, world_from_robot):
        """Create the legacy five-phase, 74-step absolute Franka trajectory."""
        robot_from_world = np.linalg.inv(world_from_robot)
        grasp_robot = robot_from_world @ grasp_world

        retreat = np.eye(4, dtype=np.float64)
        retreat[2, 3] = -0.10
        pre_grasp_robot = grasp_robot @ retreat

        lift = np.eye(4, dtype=np.float64)
        lift[2, 3] = 0.20
        lift_grasp_robot = lift @ grasp_robot

        pre_pre_action = self._pose_to_action(pre_grasp_robot, 1.0)
        pre_pre_action[2] = 0.25
        pre_action = self._pose_to_action(pre_grasp_robot, 1.0)
        grasp_action = self._pose_to_action(grasp_robot, 1.0)
        close_action = self._pose_to_action(grasp_robot, -1.0)
        lift_action = self._pose_to_action(lift_grasp_robot, -1.0)
        phases = (
            pre_pre_action,
            pre_action,
            grasp_action,
            close_action,
            lift_action,
        )
        return np.concatenate(
            [
                np.repeat(action[None], count, axis=0)
                for action, count in zip(phases, self._PHASE_LENGTHS)
            ],
            axis=0,
        ).astype(np.float32)

    @staticmethod
    def _pose_to_action(pose, gripper):
        """Convert a homogeneous pose to xyz, fixed-axis xyz RPY, and gripper state."""
        rotation = pose[:3, :3]
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        pitch = np.arctan2(
            -rotation[2, 0],
            np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2),
        )
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        return np.asarray(
            [*pose[:3, 3], roll, pitch, yaw, gripper],
            dtype=np.float64,
        )


def create_parser():
    """Create the LIBERO server command-line parser."""
    parser = argparse.ArgumentParser(description="AdaRoboVLG LIBERO gripper server")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind address. Keep the default for local use; binding to 0.0.0.0 is safe "
            "only on a trusted, isolated network because the pickle protocol has no "
            "authentication or encryption."
        ),
    )
    parser.add_argument("--port", type=int, default=LiberoGripperServerZMQ.default_port)
    parser.add_argument(
        "--text-prompt",
        action="store_true",
        help="Use local SAM3 English prompts",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main():
    """Run the LIBERO gripper server command-line entrypoint."""
    parser = create_parser()
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    server = LiberoGripperServerZMQ(
        host=args.host,
        port=args.port,
        gen_box=not args.text_prompt,
        verbose=args.verbose,
    )

    def handle_shutdown_signal(signum, frame):
        """Convert process shutdown signals into normal server cleanup."""
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
