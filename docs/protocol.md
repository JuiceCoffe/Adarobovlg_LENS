# Server/Client Protocol

This document defines requests and replies for the four supported services. See [Service Usage](../README.md#-service-usage) for launch commands.

## Transport and Services

- Transport: ZeroMQ REQ/REP over TCP. A client must receive each reply before sending the next frame.
- Encoding: `pickle.dumps(request_dict)` and `pickle.loads(reply_bytes)`, with NumPy arrays. This is not a JSON or HTTP API.
- Default host: `127.0.0.1`. For remote clients, bind the server with `--host 0.0.0.0` and give the client the server IP.
- Pickle deserialization executes Python reconstruction logic. Load only trusted packets and connect only to trusted endpoints. The services provide no authentication or encryption and must not be exposed directly to the public internet.

| Entrypoint | Port | Accepted modes | Default mode |
| --- | --- | --- | --- |
| `server/static_grasp/server_gripper.py` | 5555 | `predict` | `predict` |
| `server/static_grasp/server_dexhand.py` | 5556 | `predict` | `predict` |
| `server/dynamic_grasp/server_gripper.py` | 5557 | `init_track`, `track` | `init_track` |
| `server/dynamic_grasp/server_dexhand.py` | 5558 | `init_track`, `track` | `init_track` |

Use `--port` to override a default port. Each dynamic-service process stores one target session and does not isolate multiple clients. The protocol currently has no version negotiation, so field changes must be synchronized with clients.

## Request Fields

Let N be the number of points, V the number of views, and H × W the common image size. Every mode, including `track`, requires `points`, `colors`, and a complete `obs` object. The leading array dimension determines the camera count; retain `V=1` for one camera. Camera count and image size cannot change during a tracking session.

| Field | Type/shape | Requirement and meaning |
| --- | --- | --- |
| `mode` | `str` | Optional; defaults according to the service. |
| `prompt` | `str` | Nonempty for `predict` and `init_track`. May be omitted or empty for `track`; a nonempty value must match initialization. |
| `points` | float array `(N,3)` | N > 0; finite scene points in the world/robot frame, in meters. |
| `colors` | array `(N,3)` | Finite RGB values in [0,1], aligned with `points`. |
| `obs.color` | `uint8 (V,H,W,3)` | BGR images with positive V, H, and W. |
| `obs.depth` | array `(V,H,W)` | Finite, nonnegative depth in meters; use 0 rather than NaN for invalid pixels. |
| `obs.K` | array `(V,3,3)` | Finite camera intrinsics with fx, fy > 0 and matching resolution. |
| `obs.pose` | array `(V,3,4)` or `(V,4,4)` | Rigid extrinsics; see the next section. A 4×4 matrix must end with `[0,0,0,1]`. |
| `invert_obs_pose` | `bool` | Defaults to True on an initial request; omitted `track` values inherit the session setting. |
| `hand_name` | `str` | Dexterous hand only: `AllegroHand`, `InspireHand`, or `DH3Hand`. Defaults to server `--hand`; omitted `track` values inherit the session setting. |
| `top_k` | positive integer | Dexterous hand only: parallel-jaw proposal count, default 10. This is not the final dexterous-candidate limit. |
| `tracking_id` | nonempty string | Required for `track` and must exactly match the active session. |
| `task_name` | `str` | Optional sanitized output-directory name when `--save-request-data` is enabled. |

The parallel-jaw service ignores request `top_k` and uses `default_top_k=10`. Dexterous services support right-hand configurations only and reject legacy `left_hand` or `left_hand_mode` values that evaluate to true.

Explicit client options `--prompt`, `--hand`, and `--obs-pose` override recorded values. Otherwise, recorded values remain and the server applies defaults or session values to missing fields. While loading a packet, the client removes stale `mode`, `track`, `tracking_id`, and `grasp_index` fields before building the current request.

For compatibility, omitted `mode` maps legacy `track=True` to `track` and still requires `tracking_id`; `track=False` uses the service default. `invert_obs_pose` also accepts `true`, `false`, `1`, and `0` strings, though new clients should send a boolean. A request-side `grasp_index` never bypasses manual selection.

## Coordinates, Units, and Workspace

`points`, workspace bounds, and returned poses share the world frame W.

| Input extrinsic | `invert_obs_pose` | Client option |
| --- | --- | --- |
| Camera-to-world `T_world_from_camera` | True | `--obs-pose camera-to-world` |
| World-to-camera `T_camera_from_world` | False | `--obs-pose world-to-camera` |

Internally, every pose becomes `T_camera_from_world`, satisfying `p_camera = R @ p_world + t`. Never infer the direction from a filename or end-effector type.

The columns of a returned rotation R are the local grasp axes expressed in the world frame, satisfying `p_world = R @ p_grasp + translation`. A gripper's approach direction is column 0 of R. A dexterous-hand pose is the corresponding URDF hand-base pose; its `approach_dirs` follows the associated parallel-jaw approach axis and need not equal column 0 of the hand-base R.

Translations, depths, and widths are in meters; joint angles are in radians. Observation images are BGR, point colors are RGB, and the model converts images to RGB internally.

`--workspace-bounds xmin xmax ymin ymax zmin zmax` crops input points and filters final gripper or hand-base positions using open bounds. Defaults are x=(0.30,0.75), y=(-0.25,0.25), and z=(-0.30,0.0), intended only as experimental examples. Workspace filtering occurs before truncation to at most 10 gripper or 20 dexterous candidates. This position test does not guarantee that the entire hand geometry remains inside the workspace.

## Successful Replies

Let G be the returned candidate count. A headless static service returns between one candidate and its limit; all other successful modes return G=1. Row i across candidate arrays always describes the same grasp. Success means that inference or tracking completed, not that target recognition or physical execution is guaranteed.

The current parallel-jaw entrypoints do not perform collision filtering. Dexterous-hand candidates are collision-filtered during generation, but later dynamic frames are not checked again. See [Target Detection and Full-Scene Fallback](#target-detection-and-full-scene-fallback) for the behavior when language-guided segmentation produces no target points.

| Common field | Type/shape | Meaning |
| --- | --- | --- |
| `task_type` | `str` | `target_grasp` for gripper; `target_hand_grasp` for dexterous hand. |
| `mode` | `str` | Executed mode. |
| `rotations` | float array `(G,3,3)` | Local-grasp-to-world rotation matrices. |
| `translations` | float array `(G,3)` | Final positions after server-side depth offset; do not apply the offset again. |
| `grasp_widths` | float array `(G,)` | Associated parallel-jaw widths in meters. |
| `final_gripper_width` | `float` | Convenience copy of `grasp_widths[0]`, not post-closure finger spacing. |
| `invert_obs_pose` | `bool` | Extrinsic convention used by the request/session. |
| `workspace_bounds` | `dict` | x, y, and z `[min,max]` bounds in meters. |
| `workspace_filtered_count` | `int` | Candidates excluded by pose bounds during static/initial inference, before count truncation; tracking retains the initial value. |
| `grasp_index` | `int`, conditional | Confirmed index in the sorted, filtered preselection list. |

After manual selection, read the single entry at `translations[0]`, `rotations[0]`, and corresponding fields. `grasp_index` preserves the original candidate index and must not index the one-entry reply arrays.

| Device field | Type/shape | Meaning |
| --- | --- | --- |
| `grasp_dirs` | `(G,3)` | Gripper-only world-frame approach directions. |
| `approach_dirs` | `(G,3)` | Dexterous-only world-frame approach directions. |
| `hand_joint_pre_angles` | `(G,D)` | Pre-grasp joints mapped from original width + 0.04 m. |
| `hand_joint_angles` | `(G,D)` | Closed joints mapped from max(original width − 0.07 m, 0). |
| `grasp_types` | `(G,)` | Dexterous taxonomy IDs; services retain available types among 1–5. |
| `hand_name` | `str` | Right-hand model actually used. |
| `left_hand_mode` | `bool` | Always False for dexterous services. |
| `model_entrypoint` | `str` | Currently `forward` for dexterous services; intended for diagnostics. |
| `orientation_policy` | `str`, conditional | `top_down` when a dynamic gripper uses `--top-down`. |

Joint arrays follow the non-fixed, non-mimic URDF joint order:

- Allegro (D=16): `joint_0.0` through `joint_15.0`, numerically ordered.
- Inspire (D=6): `thumb_proximal_yaw_joint`, `thumb_proximal_pitch_joint`, `index_proximal_joint`, `middle_proximal_joint`, `ring_proximal_joint`, `pinky_proximal_joint`.
- DH3 (D=2): `finger1_base_joint`, `finger1_joint`.

These are model joint angles, not vendor motor commands. Recheck ordering after changing a URDF or mapping table.

### Target Detection and Full-Scene Fallback

Target segmentation and grasp availability are separate outcomes. If language-guided mask projection produces no target points, the model uses its highest-ranked predicted grasp point as a SAM point prompt and projects the resulting mask again. The following behavior applies if this retry still produces no target points:

- A static parallel-jaw or dexterous-hand service generates candidates from the complete workspace point cloud. If usable candidates remain, it returns a normal success reply, but those grasps are not guaranteed to belong to the requested language target.
- A dynamic service may generate and request confirmation of a full-scene fallback candidate, but tracking initialization subsequently fails because it requires target points. The request returns `request_failed`, no tracking session is created, and `requires_reinit` is `True`.
- If grasp generation returns no candidates, dexterous-hand mapping returns no candidates, or workspace filtering removes every candidate, the service returns an error reply rather than a successful reply with zero-length pose arrays.

Successful replies currently contain no `target_detected` or `fallback_mode` field. A client therefore cannot determine from the reply alone whether a static grasp came from the requested target or from the full-scene fallback. Applications that require verified target identity should not treat protocol success as recognition confirmation.

## Selection and Dynamic Sessions

Headless static services return candidates automatically. Static services with visualization and every dynamic initialization require selection in the server terminal:

```text
all         Preview all candidates
0           Preview candidate 0 without confirming
select 0    Confirm candidate 0
cancel      Cancel the request
```

Initialization proceeds as first-frame inference, final filtering, manual confirmation, tracker initialization, and successful reply. The same initial observation is retained while waiting. After confirmation, send the newest camera frame; reinitialize from a new observation if the target moved substantially.

| Dynamic field | Type/shape | Meaning |
| --- | --- | --- |
| `tracking_id` | `str` | Active process-local session ID; a new `init_track` replaces the old session. |
| `frame_index` | `int` | Starts at 0 and increments after successful tracking; not a camera frame number. |
| `tracking_transform` | `(4,4)` | Rigid initial-object-to-current-object transform; identity at initialization. |

Tracking updates only the selected pose, rotation, and direction. Width, grasp type, joint angles, and `grasp_index` remain from the first frame. Poses follow object rotation by default; gripper `--top-down` realigns the approach direction with world −Z and applies the depth offset once. Transforms always use the first frame as reference rather than accumulating frame by frame.

The server rejects incorrect IDs, changed targets or hand models, changed image dimensions or view counts, and changed extrinsic conventions. Exceptions after tracking starts, invalid rigid transforms, and out-of-bounds poses clear the session. Closing the tracking window disables display only; closing it during selection cancels the request.

## Errors and Timeouts

Error replies contain error metadata only; clients must not read grasp arrays from them:

```python
{'error': 'Selection cancelled',
 'error_code': 'selection_cancelled',
 'requires_reinit': True}
```

For example, a static parallel-jaw request with no usable pose returns:

```python
{'error': 'No target_grasp grasps generated.',
 'error_code': 'request_failed',
 'requires_reinit': False}
```

The corresponding failure during dynamic initialization has `requires_reinit=True` because no tracking session was created.

| Field | Meaning |
| --- | --- |
| `error` | Human-readable message; do not use its contents as a stable branch condition. |
| `error_code` | `selection_cancelled` for cancellation, selection timeout, EOF, or selection-UI failure; otherwise `request_failed`. |
| `requires_reinit` | True when a dynamic service has no valid session; always False for static services. |

No-candidate outcomes use `request_failed`; they are never represented by empty `translations`, `rotations`, or other candidate arrays in a successful reply. Typical `error` values are `No target_grasp grasps generated.` or `No target_hand_grasp grasps generated.` when generation or hand mapping produces no usable pose, and `No <task_type> grasps remain inside workspace_bounds` when final pose filtering removes every candidate. Treat `error` as human-readable diagnostics rather than a stable machine-readable value.

For these failures, `requires_reinit` is `False` for static services. It is `True` when a dynamic initialization has no valid target or pose, and after any dynamic failure that clears the active session. An erroneous request does not always clear an existing session, so dynamic clients must follow the returned `requires_reinit` value. Unknown exceptions currently have no finer error classification.

A custom ZMQ client receives the raw error dictionary above. The reference clients in `client/` instead detect the `error` field and raise `RuntimeError`; they do not return the dictionary as a normal result, and callers must not attempt to read grasp arrays after that exception.

Server `--selection-timeout` is in seconds; 0 waits indefinitely and limits only manual selection. Client `--timeout` is also in seconds; 0 waits indefinitely for a reply, while sending has a 10-second limit. A timeout does not cancel server inference or session advancement. Close and reconnect a timed-out REQ socket, reinitialize when required, and do not automatically resend the same frame.

## Python Client Example

The `packet` below must come from a camera pipeline and satisfy this document. Initialization and subsequent frames use the same connection. For a dynamic dexterous hand, use port 5558 and set `hand_name`.

```python
import pickle

import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.LINGER, 0)
socket.setsockopt(zmq.SNDTIMEO, 10000)
socket.setsockopt(zmq.RCVTIMEO, -1)
socket.connect("tcp://127.0.0.1:5557")
tracking_id = None
try:
    while True:
        packet = read_latest_packet()  # Implement this to return points, colors, and obs.
        packet.update(
            prompt="white box",
            invert_obs_pose=False,
            mode="init_track" if tracking_id is None else "track",
        )
        if tracking_id is not None:
            packet["tracking_id"] = tracking_id
        socket.send(pickle.dumps(packet))
        reply = pickle.loads(socket.recv())
        if "error" in reply:
            raise RuntimeError(f"{reply['error_code']}: {reply['error']}")
        tracking_id = reply["tracking_id"]
        print(reply["frame_index"], reply["translations"], reply["rotations"])
finally:
    socket.close()
    context.term()
```

For static integration, use port 5555 or 5556 with `mode='predict'`, send one request, and read the common and device-specific fields. The model does not capture later frames automatically, predict motion, repeat collision checks during tracking, or control a robot.
