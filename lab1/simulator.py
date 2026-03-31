import numpy as np
import taichi as ti
from util import skew_symmetric, get_camera_ray_dir
from scene import Scene


class Simulator:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.masses = self.scene.mass.to_numpy()
        self.inv_inertia = self.scene.inv_inertia_body.to_numpy()
        self.inertia = self.scene.inertia_body.to_numpy()
        self.n_bodies = self.scene.num_bodies[None]

    def render(self, window, camera, canvas, scene_3d):
        scene_3d.set_camera(camera)
        scene_3d.ambient_light((0.6, 0.6, 0.6))
        scene_3d.point_light((5, 5, 5), (1.2, 1.2, 1.2))
        for i in range(self.scene.num_bodies[None]):
            scene_3d.mesh(
                self.scene.mesh_vertices,
                self.scene.mesh_indices,
                color=tuple(self.scene.mesh_colors[i]),
                index_offset=self.scene.index_offset[i],
                index_count=self.scene.index_count[i],
            )
        canvas.scene(scene_3d)
        window.get_canvas().set_background_color((0.8, 0.8, 0.85))
        window.show()

    def step(self, applied_forces=None, applied_torques=None):
        substeps = self.scene.substeps
        dt = self.scene.dt / substeps
        pos, vel, rot, ang_vel = self.scene.get_state()

        for _ in range(substeps):
            if applied_forces is not None:
                vel += dt * (applied_forces / self.masses[:, None])
            vel += dt * self.scene.gravity[None, :]
            vel *= max(0.0, 1.0 - self.scene.linear_damping * dt)

            for i in range(self.n_bodies):
                R = rot[i]
                I_inv = R @ self.inv_inertia[i] @ R.T
                I_curr = R @ self.inertia[i] @ R.T

                tau = (
                    applied_torques[i]
                    if applied_torques is not None
                    else np.zeros(3, dtype=np.float32)
                )
                omega = ang_vel[i]

                # Including the gyroscopic term: tau - omega x (I * omega)
                tau_total = tau - np.cross(omega, I_curr @ omega)
                ang_vel[i] += dt * (I_inv @ tau_total)

            ang_vel *= max(0.0, 1.0 - self.scene.angular_damping * dt)
            pos += dt * vel

            for i in range(self.n_bodies):
                rot[i] = self._integrate_rotation(rot[i], ang_vel[i], dt)

        self.scene.set_state(pos, vel, rot, ang_vel)
        self.scene.update_mesh_vertices()

    def _integrate_rotation(self, r, omega, dt):
        dtheta = omega * dt
        theta = np.linalg.norm(dtheta)
        if theta < 1e-7:
            delta = np.eye(3, dtype=np.float32) + skew_symmetric(dtheta)
        else:
            axis = dtheta / theta
            k = skew_symmetric(axis)
            delta = (
                np.eye(3, dtype=np.float32)
                + np.sin(theta) * k
                + (1.0 - np.cos(theta)) * (k @ k)
            )
        return delta @ r

    def run(self, steps):
        window = ti.ui.Window("Rigid Body Simulation", (1280, 720), vsync=True)
        canvas = window.get_canvas()
        scene_3d = window.get_scene()
        camera = ti.ui.Camera()
        camera.position(3, 2, 3)
        camera.lookat(0, 0.5, 0)
        camera.up(0, 1, 0)

        selected_body = -1
        click_pos_w = None
        click_normal_w = None
        original_mouse_pos = None

        frame = 0
        while window.running and frame < (steps if steps > 0 else float("inf")):
            ctrl_pressed = window.is_pressed(ti.ui.CTRL)
            lmb_pressed = window.is_pressed(ti.ui.LMB)
            mouse_pos = window.get_cursor_pos()

            if not ctrl_pressed:
                camera.track_user_inputs(
                    window, movement_speed=0.03, hold_key=ti.ui.RMB
                )

            applied_forces = np.zeros((self.n_bodies, 3), dtype=np.float32)
            applied_torques = np.zeros((self.n_bodies, 3), dtype=np.float32)

            if ctrl_pressed and lmb_pressed:
                cam_p = np.array(camera.curr_position, dtype=np.float32)
                cam_look = np.array(camera.curr_lookat, dtype=np.float32)
                cam_up = np.array(camera.curr_up, dtype=np.float32)

                if selected_body == -1:
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
                            selected_body = i
                            click_pos_w = cam_p + t * ray_dir
                            click_normal_w = rot[i] @ n_l
                    if selected_body != -1:
                        original_mouse_pos = mouse_pos

                if selected_body != -1:
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
                        original_mouse_pos[0],
                        original_mouse_pos[1],
                        cam_p,
                        cam_look,
                        cam_up,
                        45.0,
                        1280.0 / 720.0,
                    )

                    V_drag = (curr_ray_dir - orig_ray_dir) * 500.0
                    N = click_normal_w
                    V_tangent = V_drag - np.dot(V_drag, N) * N

                    F = V_tangent * 0.2

                    pos, _, _, _ = self.scene.get_state()
                    r = click_pos_w - pos[selected_body]
                    tau = np.cross(r, F)

                    applied_forces[selected_body] = F
                    applied_torques[selected_body] = tau
            else:
                selected_body = -1

            self.step(applied_forces, applied_torques)
            self.render(window, camera, canvas, scene_3d)
            frame += 1
