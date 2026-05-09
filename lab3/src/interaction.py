import numpy as np
import taichi as ti
from scene import Scene
from util import get_camera_ray_dir


class InteractionHandler:
    def __init__(self, scene: Scene, force_scale=200.0, pick_radius=0.12):
        self.scene = scene
        self.force_scale = float(force_scale)
        self.pick_radius = float(pick_radius)
        self.selected_vertex = -1
        self.original_mouse_pos = None

    def _pick_nearest_vertex(self, mouse_pos, camera):
        cam_pos = np.asarray(camera.curr_position, dtype=np.float32)
        cam_lookat = np.asarray(camera.curr_lookat, dtype=np.float32)
        cam_up = np.asarray(camera.curr_up, dtype=np.float32)
        ray_dir = get_camera_ray_dir(
            mouse_pos[0], mouse_pos[1], cam_pos, cam_lookat, cam_up
        )

        positions = self.scene.pos.to_numpy()
        rel = positions - cam_pos.reshape(1, 3)
        t = rel @ ray_dir
        in_front = t > 0.0
        closest = cam_pos.reshape(1, 3) + t.reshape(-1, 1) * ray_dir.reshape(1, 3)
        dist = np.linalg.norm(positions - closest, axis=1)
        dist[~in_front] = np.inf
        vid = int(np.argmin(dist))
        if vid >= 0 and float(dist[vid]) <= self.pick_radius:
            return vid
        return -1

    def process_inputs(self, window, camera):
        applied_forces = np.zeros((self.scene.num_vertices, 3), dtype=np.float32)
        ctrl_pressed = window.is_pressed(ti.ui.CTRL)
        lmb_pressed = window.is_pressed(ti.ui.LMB)
        mouse_pos = window.get_cursor_pos()

        if not ctrl_pressed or not lmb_pressed:
            self.selected_vertex = -1
            self.original_mouse_pos = None
            return applied_forces

        if self.selected_vertex == -1:
            self.selected_vertex = self._pick_nearest_vertex(mouse_pos, camera)
            self.original_mouse_pos = mouse_pos

        if self.selected_vertex != -1:
            cam_pos = np.asarray(camera.curr_position, dtype=np.float32)
            cam_lookat = np.asarray(camera.curr_lookat, dtype=np.float32)
            cam_up = np.asarray(camera.curr_up, dtype=np.float32)
            fwd = cam_lookat - cam_pos
            fwd /= np.linalg.norm(fwd) + 1e-8
            up0 = cam_up / (np.linalg.norm(cam_up) + 1e-8)
            right = np.cross(fwd, up0)
            right /= np.linalg.norm(right) + 1e-8
            up = np.cross(right, fwd)
            up /= np.linalg.norm(up) + 1e-8

            dx = mouse_pos[0] - self.original_mouse_pos[0]
            dy = mouse_pos[1] - self.original_mouse_pos[1]
            applied_forces[self.selected_vertex] = (
                dx * right + dy * up
            ) * self.force_scale

        return applied_forces
