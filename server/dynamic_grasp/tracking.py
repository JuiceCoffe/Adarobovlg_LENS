"""Shared state and rigid updates for the two independent dynamic services."""

import copy
import uuid

import numpy as np


def _run_task_with_tracking(self, request, debug=False):
    """Keep one selected, final grasp anchored to the initial object frame."""
    mode = request.mode
    model = self.adarobovlg

    # Later frames update the confirmed candidate without regeneration or reselection.
    if mode == "track":
        state = getattr(self, "_tracking_state", None)
        if state is None:
            raise ValueError("Run init_track before track; no active tracking session.")
        if request.messages.get("tracking_id") != state["id"]:
            raise ValueError("tracking_id does not match the active session; run init_track again.")
        if request.invert_obs_pose != state["invert_obs_pose"]:
            raise ValueError("Camera pose convention changed; run init_track again.")
        if request.prompt and request.prompt != state["prompt"]:
            raise ValueError("Target prompt changed; run init_track again.")
        if "hand_name" in state["reply"]:
            hand_name, _ = self._resolve_hand_options(request)
            if hand_name != state["reply"]["hand_name"]:
                raise ValueError("Hand type changed; run init_track again.")
        if tuple(request.obs["color"].shape) != state["image_shape"]:
            raise ValueError("Camera count or image size changed; run init_track again.")
        try:
            # track_3d returns initial-frame -> current-frame, not a frame increment.
            transform = np.asarray(model.track_featproj.track_3d(request.obs))
            if (
                transform.shape != (4, 4)
                or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-4)
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-3)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-3)
            ):
                raise ValueError("Tracker returned an invalid rigid transform.")
            rotation, translation = transform[:3, :3], transform[:3, 3]
            reply = copy.deepcopy(state["reply"])
            reply["rotations"] = rotation @ reply["rotations"]
            reply["translations"] = reply["translations"] @ rotation.T + translation
            for key in ("grasp_dirs", "approach_dirs"):
                if key in reply:
                    reply[key] = reply[key] @ rotation.T
            if getattr(self, "top_down", False):
                self._constrain_tracking_reply(reply, state["gg"].depths)
            if not self._grasp_workspace_mask(reply["translations"]).all():
                raise ValueError("Tracked grasp left workspace_bounds; run init_track again.")
            gg = copy.deepcopy(state["gg"])
            gg.rotation_matrices = reply["rotations"].copy()
            gg.translations = reply["translations"].copy()
            model.last_tracking_T = transform.copy()
            model.last_tracked_obj_pts = model.obj_pts @ rotation.T + translation
            model.last_tracked_obj_colors = model.obj_colors
            # The first-frame mask is not aligned with the current point cloud.
            model.mask_3d = None
            state["frame_index"] += 1
            reply.update(
                mode="track",
                tracking_id=state["id"],
                frame_index=state["frame_index"],
                tracking_transform=transform.copy(),
            )
            return gg, reply, "all"
        except Exception:
            self._tracking_state = None
            model.track_featproj = None
            raise

    # A fresh prediction or initialization invalidates the previous target.
    self._tracking_state = None
    model.track_featproj = None
    model.track = False
    gg, reply, vis_top_k = self.run_task(request, debug=debug)
    reply["mode"] = mode

    # Create the session only after confirmation while retaining the initial observation.
    index = self._select_candidate(request, gg, reply)
    gg, reply = self._slice_candidate(gg, reply, index)
    try:
        model.track = True
        model.prepare_grasp_tracking(request.obs)
    except Exception:
        model.track_featproj = None
        raise
    finally:
        model.track = False
    session_id = uuid.uuid4().hex
    reply.update(
        tracking_id=session_id, frame_index=0, grasp_index=int(index), tracking_transform=np.eye(4)
    )
    # Preserve the unconstrained initial grasp to prevent transform and offset accumulation.
    self._tracking_state = {
        "id": session_id,
        "gg": copy.deepcopy(gg),
        "reply": copy.deepcopy(reply),
        "prompt": request.prompt,
        "invert_obs_pose": request.invert_obs_pose,
        "image_shape": tuple(request.obs["color"].shape),
        "frame_index": 0,
    }
    if getattr(self, "top_down", False):
        try:
            self._constrain_tracking_reply(reply, gg.depths)
            gg.rotation_matrices = reply["rotations"].copy()
            gg.translations = reply["translations"].copy()
            if not self._grasp_workspace_mask(reply["translations"]).all():
                raise ValueError("Constrained grasp left workspace_bounds; run init_track again.")
        except Exception:
            self._tracking_state = None
            model.track_featproj = None
            raise
    return gg, reply, "all"
