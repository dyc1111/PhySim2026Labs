import numpy as np
import taichi as ti
from util import get_camera_ray_dir


class InteractionHandler:
    def __init__(self, scene):
        self.scene = scene
        self.n_bodies = scene.num_bodies[None]
        self.selected_body = -1
        self.interaction_mode = None
        self.click_pos_w = None
        self.click_normal_w = None
        self.original_mouse_pos = None

    def process_inputs(self, window, camera):
        applied_forces = np.zeros((self.n_bodies, 3), dtype=np.float32)
        applied_torques = np.zeros((self.n_bodies, 3), dtype=np.float32)

        lmb_pressed = window.is_pressed(ti.ui.LMB)
        rmb_pressed = window.is_pressed(ti.ui.RMB)
        ctrl_pressed = window.is_pressed(ti.ui.CTRL)
        mouse_pos = window.get_cursor_pos()

        if not ctrl_pressed:
            self.selected_body = -1
            self.interaction_mode = None
            return applied_forces, applied_torques

        if lmb_pressed or rmb_pressed:
            cam_p = np.array(camera.curr_position, dtype=np.float32)
            cam_look = np.array(camera.curr_lookat, dtype=np.float32)
            cam_up = np.array(camera.curr_up, dtype=np.float32)

            if self.selected_body == -1:
                ray_dir = get_camera_ray_dir(
                    mouse_pos[0],
                    mouse_pos[1],
                    cam_p,
                    cam_look,
                    cam_up,
                    45.0,
                    1280.0 / 720.0,
                )
                min_t = np.inf
                pos, _, rot, _ = self.scene.get_state()
                for i in range(self.n_bodies):
                    orig_l = rot[i].T @ (cam_p - pos[i])
                    dir_l = rot[i].T @ ray_dir
                    hit, t, n_l = self.scene.bodies[i].ray_intersect(orig_l, dir_l)
                    if hit and t < min_t:
                        min_t = t
                        self.selected_body = i
                        self.click_pos_w = cam_p + t * ray_dir
                        self.click_normal_w = rot[i] @ n_l
                if self.selected_body != -1:
                    self.original_mouse_pos = mouse_pos
                    self.interaction_mode = "TRANSLATE" if lmb_pressed else "ROTATE"

            if self.selected_body != -1:
                if self.interaction_mode == "TRANSLATE":
                    F_dir = cam_look - cam_p
                    F_dir /= np.linalg.norm(F_dir) + 1e-8
                    U0 = cam_up / (np.linalg.norm(cam_up) + 1e-8)
                    R_vec = np.cross(F_dir, U0)
                    R_vec /= np.linalg.norm(R_vec) + 1e-8
                    U_vec = np.cross(R_vec, F_dir)
                    U_vec /= np.linalg.norm(U_vec) + 1e-8

                    dx = mouse_pos[0] - self.original_mouse_pos[0]
                    dy = mouse_pos[1] - self.original_mouse_pos[1]

                    F = (dx * R_vec + dy * U_vec) * 100.0
                    applied_forces[self.selected_body] = F

                elif self.interaction_mode == "ROTATE":
                    curr_ray_dir = get_camera_ray_dir(
                        mouse_pos[0],
                        mouse_pos[1],
                        cam_p,
                        cam_look,
                        cam_up,
                        45.0,
                        1280.0 / 720.0,
                    )
                    orig_ray_dir = get_camera_ray_dir(
                        self.original_mouse_pos[0],
                        self.original_mouse_pos[1],
                        cam_p,
                        cam_look,
                        cam_up,
                        45.0,
                        1280.0 / 720.0,
                    )

                    V_drag = (curr_ray_dir - orig_ray_dir) * 100.0
                    N = self.click_normal_w
                    V_tangent = V_drag - np.dot(V_drag, N) * N

                    pos, _, _, _ = self.scene.get_state()
                    r = self.click_pos_w - pos[self.selected_body]
                    tau = np.cross(r, V_tangent)

                    applied_torques[self.selected_body] = tau
        else:
            self.selected_body = -1
            self.interaction_mode = None

        return applied_forces, applied_torques