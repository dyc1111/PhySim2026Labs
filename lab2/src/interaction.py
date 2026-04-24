import numpy as np
import taichi as ti
from util import get_camera_ray_dir


class InteractionHandler:
    def __init__(self, scene):
        self.scene = scene
        self.n_bodies = scene.num_rigidbodies
        self.selected_body = -1
        self.interaction_mode = None
        self.click_pos_w = None
        self.click_normal_w = None
        self.original_mouse_pos = None

    def process_inputs(self, window, camera):
        applied_forces = np.zeros((self.n_bodies, 3), dtype=np.float32)
        applied_torques = np.zeros((self.n_bodies, 3), dtype=np.float32)
        if self.n_bodies == 0:
            return applied_forces, applied_torques

        lmb_pressed = window.is_pressed(ti.ui.LMB)
        rmb_pressed = window.is_pressed(ti.ui.RMB)
        ctrl_pressed = window.is_pressed(ti.ui.CTRL)
        mouse_pos = window.get_cursor_pos()

        if not ctrl_pressed:
            self.selected_body = -1
            self.interaction_mode = None
            return applied_forces, applied_torques

        if lmb_pressed or rmb_pressed:
            cam_pos = np.array(camera.curr_position, dtype=np.float32)
            cam_lookat = np.array(camera.curr_lookat, dtype=np.float32)
            cam_up = np.array(camera.curr_up, dtype=np.float32)

            if self.selected_body == -1:
                ray_dir = get_camera_ray_dir(
                    mouse_pos[0],
                    mouse_pos[1],
                    cam_pos,
                    cam_lookat,
                    cam_up,
                    45.0,
                    1280.0 / 720.0,
                )
                min_t = np.inf
                pos, _, rot, _ = self.scene.get_rigidbody_state()
                for i in range(self.n_bodies):
                    orig_l = rot[i].T @ (cam_pos - pos[i])
                    dir_l = rot[i].T @ ray_dir
                    hit, t, n_l = self.scene.rigid_bodies[i].ray_intersect(
                        orig_l, dir_l
                    )
                    if hit and t < min_t:
                        min_t = t
                        self.selected_body = i
                        self.click_pos_w = cam_pos + t * ray_dir
                        self.click_normal_w = rot[i] @ n_l
                if self.selected_body != -1:
                    self.original_mouse_pos = mouse_pos
                    self.interaction_mode = "TRANSLATE" if lmb_pressed else "ROTATE"

            if self.selected_body != -1:
                if self.interaction_mode == "TRANSLATE":
                    fwd = cam_lookat - cam_pos
                    fwd /= np.linalg.norm(fwd) + 1e-8
                    up0 = cam_up / (np.linalg.norm(cam_up) + 1e-8)
                    right = np.cross(fwd, up0)
                    right /= np.linalg.norm(right) + 1e-8
                    up = np.cross(right, fwd)
                    up /= np.linalg.norm(up) + 1e-8

                    dx = mouse_pos[0] - self.original_mouse_pos[0]
                    dy = mouse_pos[1] - self.original_mouse_pos[1]
                    force = (dx * right + dy * up) * 100
                    applied_forces[self.selected_body] = force
                elif self.interaction_mode == "ROTATE":
                    curr_ray_dir = get_camera_ray_dir(
                        mouse_pos[0],
                        mouse_pos[1],
                        cam_pos,
                        cam_lookat,
                        cam_up,
                        45.0,
                        1280.0 / 720.0,
                    )
                    prev_ray_dir = get_camera_ray_dir(
                        self.original_mouse_pos[0],
                        self.original_mouse_pos[1],
                        cam_pos,
                        cam_lookat,
                        cam_up,
                        45.0,
                        1280.0 / 720.0,
                    )

                    drag = (curr_ray_dir - prev_ray_dir) * 100
                    normal = self.click_normal_w
                    tangential_drag = drag - np.dot(drag, normal) * normal
                    pos, _, _, _ = self.scene.get_rigidbody_state()
                    arm = self.click_pos_w - pos[self.selected_body]
                    torque = np.cross(arm, tangential_drag)
                    applied_torques[self.selected_body] = torque
        else:
            self.selected_body = -1
            self.interaction_mode = None

        return applied_forces, applied_torques
