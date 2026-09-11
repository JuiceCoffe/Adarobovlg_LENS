"""Open3D display in a separate process; terminal selection stays in the server."""

import multiprocessing as mp
from queue import Empty, Full

import numpy as np


def _render(queue, ready, closed):
    """Render queued geometry in a child process; signal readiness and window closure."""
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    try:
        if not vis.create_window(window_name="AdaRoboVLG", width=1280, height=720):
            return
        vis.get_render_option().point_size = 4.0
        ready.set()
        first_frame = True
        while vis.poll_events():
            try:
                payload = queue.get(timeout=0.03)
            except Empty:
                vis.update_renderer()
                continue
            if payload is None:
                break
            # Preserve the user's viewing angle while poses move between frames.
            camera = (
                None
                if first_frame
                else vis.get_view_control().convert_to_pinhole_camera_parameters()
            )
            vis.clear_geometries()
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(payload["points"])
            cloud.colors = o3d.utility.Vector3dVector(payload["colors"])
            vis.add_geometry(cloud)
            vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
            selected = payload.get("selected_pose", "all")
            for index, item in enumerate(payload["grasps"]):
                if selected != "all" and index != selected:
                    continue
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(item["vertices"])
                mesh.triangles = o3d.utility.Vector3iVector(item["triangles"])
                mesh.vertex_colors = o3d.utility.Vector3dVector(item["colors"])
                mesh.compute_vertex_normals()
                vis.add_geometry(mesh)
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                frame.transform(item["pose"])
                vis.add_geometry(frame)
            if camera is not None:
                vis.get_view_control().convert_from_pinhole_camera_parameters(
                    camera, allow_arbitrary=True
                )
            first_frame = False
            vis.update_renderer()
    finally:
        closed.set()
        vis.destroy_window()


class GraspVisualizer:
    """Own the display process and its bounded latest-frame queue."""

    def __init__(self):
        # Use spawn to isolate the window from the main GPU and rendering contexts.
        """Prepare a spawn-based display process and a one-frame queue without opening a window."""
        context = mp.get_context("spawn")
        # Retain only the latest frame so slow rendering cannot queue stale observations.
        self.queue = context.Queue(maxsize=1)
        self.ready = context.Event()
        self.closed_event = context.Event()
        self.process = context.Process(
            target=_render,
            args=(self.queue, self.ready, self.closed_event),
            daemon=True,
        )
        self.started = False
        self.stopped = False

    @property
    def closed(self):
        """Return whether the display window closed or its process exited."""
        return self.closed_event.is_set() or (self.started and not self.process.is_alive())

    def build_payload(self, points, colors, group, mask, reply):
        """Copy point and mesh arrays for interprocess display using the same poses as the reply."""
        colors = np.array(colors, copy=True)
        if mask is not None and len(mask) == len(colors):
            colors[np.asarray(mask)[:, 0] < 0.5] *= 0.3
        grasps = []
        for index, grasp in enumerate(group):
            mesh = grasp.to_open3d_geometry()
            vertices = np.asarray(mesh.vertices).copy()
            mesh_colors = np.asarray(mesh.vertex_colors).copy()
            if not len(mesh_colors):
                mesh_colors = np.tile([0.7, 0.7, 0.7], (len(vertices), 1))
            pose = np.eye(4)
            pose[:3, :3] = reply["rotations"][index]
            pose[:3, 3] = reply["translations"][index]
            grasps.append(
                {
                    "vertices": vertices,
                    "triangles": np.asarray(mesh.triangles).copy(),
                    "colors": mesh_colors,
                    "pose": pose,
                }
            )
        return {
            "points": np.array(points, copy=True),
            "colors": colors,
            "grasps": grasps,
            "selected_pose": "all",
        }

    def show(self, payload):
        """Start the window if needed and queue the latest display frame without blocking inference."""
        if self.closed:
            raise RuntimeError("Open3D window is closed")
        if not self.started:
            self.process.start()
            self.started = True
            if not self.ready.wait(timeout=10) or self.closed:
                self.closed_event.set()
                raise RuntimeError("Open3D window could not start")
        try:
            self.queue.put_nowait(payload)
        except Full:
            try:
                self.queue.get(timeout=0.05)
            except Empty:
                pass
            try:
                self.queue.put_nowait(payload)
            except Full:
                pass  # A newer display update can replace this frame.

    def close(self):
        """Stop the display process and release its bounded queue; repeated calls are harmless."""
        if self.stopped:
            return
        self.stopped = True
        self.closed_event.set()
        if self.started:
            try:
                self.queue.put_nowait(None)
            except Full:
                pass
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
        self.queue.cancel_join_thread()
        self.queue.close()
