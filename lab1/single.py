import taichi as ti
import numpy as np
import hydra
from omegaconf import OmegaConf
from constants import *
from util import *


ti.init(arch=ti.gpu)


class RigidBody:
    def __init__(self, cfg):
        self.mass = float(cfg["mass"])
        self.position = np.array(cfg["position"], dtype=np.float32)
        self.velocity = np.array(cfg["velocity"], dtype=np.float32)
        self.angular_velocity = np.array(cfg["angular_velocity"], dtype=np.float32)
        self.rotation = euler_angle_to_matrix(cfg["rotation_deg"])
        self.color = np.array(cfg["color"], dtype=np.float32)

    @property
    def vertex_count(self):
        raise NotImplementedError

    @property
    def index_count(self):
        raise NotImplementedError

    def get_inertia_diag(self):
        raise NotImplementedError

    def setup_mesh(self, indices_field, i_offset, v_offset):
        raise NotImplementedError

    def update_mesh_vertices(
        self, pos_field, rot_field, body_id, vertices_field, v_offset
    ):
        raise NotImplementedError

    def ray_intersect(self, orig_l, dir_l):
        raise NotImplementedError


@ti.data_oriented
class Cuboid(RigidBody):
    vertex_count = 8
    index_count = 36

    local_verts = ti.Vector.field(3, dtype=ti.f32, shape=8)
    indices = ti.field(dtype=ti.i32, shape=36)
    _is_initialized = False

    @classmethod
    def initialize(cls):
        if not cls._is_initialized:
            cls.local_verts.from_numpy(CUBE_LOCAL_VERTS_NP)
            cls.indices.from_numpy(CUBE_INDICES_NP)
            cls._is_initialized = True

    def __init__(self, cfg):
        super().__init__(cfg)
        Cuboid.initialize()
        self.size = np.array(cfg["size"], dtype=np.float32)
        self.half_extent = 0.5 * self.size

    def get_inertia_diag(self):
        lx, ly, lz = self.size
        ixx = (self.mass / 12.0) * (ly * ly + lz * lz)
        iyy = (self.mass / 12.0) * (lx * lx + lz * lz)
        izz = (self.mass / 12.0) * (lx * lx + ly * ly)
        return np.array([ixx, iyy, izz], dtype=np.float32)

    def ray_intersect(self, orig_l, dir_l):
        return ray_aabb_intersect(orig_l, dir_l, self.half_extent)

    def setup_mesh(self, indices_field, i_offset, v_offset):
        self._setup_mesh(indices_field, i_offset, v_offset)

    @classmethod
    @ti.kernel
    def _setup_mesh(
        cls, indices_field: ti.template(), i_offset: ti.i32, v_offset: ti.i32  # type: ignore
    ):
        for i in range(cls.index_count):
            indices_field[i_offset + i] = v_offset + cls.indices[i]

    def update_mesh_vertices(
        self, pos_field, rot_field, body_id, vertices_field, v_offset
    ):
        self._update_mesh_vertices(
            pos_field,
            rot_field,
            body_id,
            ti.Vector(self.half_extent),
            vertices_field,
            v_offset,
        )

    @classmethod
    @ti.kernel
    def _update_mesh_vertices(
        cls,
        pos_field: ti.template(),  # type: ignore
        rot_field: ti.template(),  # type: ignore
        body_id: ti.i32,  # type: ignore
        half_ext: ti.types.vector(3, ti.f32),  # type: ignore
        vertices_field: ti.template(),  # type: ignore
        v_offset: ti.i32,  # type: ignore
    ):
        pos = pos_field[body_id]
        rot = rot_field[body_id]
        for k in range(cls.vertex_count):
            lv = cls.local_verts[k]
            local = ti.Vector(
                [lv[0] * half_ext[0], lv[1] * half_ext[1], lv[2] * half_ext[2]]
            )
            world = rot @ local + pos
            vertices_field[v_offset + k] = world


@ti.data_oriented
class Scene:
    def __init__(self, scene_cfg):
        self.dt = scene_cfg["dt"]
        self.substeps = scene_cfg["substeps"]
        self.gravity = np.array(scene_cfg["gravity"], dtype=np.float32)
        self.linear_damping = scene_cfg["linear_damping"]
        self.angular_damping = scene_cfg["angular_damping"]

        objects = scene_cfg["objects"]

        self.bodies = []
        for cfg in objects:
            body_type = cfg["type"]
            if body_type == "cuboid":
                self.bodies.append(Cuboid(cfg))
            else:
                raise NotImplementedError(f"Unsupported body type: {body_type}")

        self.num_bodies = ti.field(dtype=ti.i32, shape=())
        n_bodies = len(self.bodies)
        self.num_bodies[None] = n_bodies

        self.position = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.rotation = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_bodies)
        self.angular_velocity = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.mass = ti.field(dtype=ti.f32, shape=n_bodies)
        self.inertia_body = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_bodies)
        self.inv_inertia_body = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_bodies)

        total_vertices = sum(b.vertex_count for b in self.bodies)
        total_indices = sum(b.index_count for b in self.bodies)

        self.mesh_vertices = ti.Vector.field(3, dtype=ti.f32, shape=total_vertices)
        self.mesh_indices = ti.field(dtype=ti.i32, shape=total_indices)
        self.mesh_colors = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.index_offset = ti.field(dtype=ti.i32, shape=n_bodies)
        self.index_count = ti.field(dtype=ti.i32, shape=n_bodies)

        v_offset = 0
        i_offset = 0

        for i, body in enumerate(self.bodies):
            self.position[i] = body.position
            self.velocity[i] = body.velocity
            self.angular_velocity[i] = body.angular_velocity
            self.rotation[i] = body.rotation
            self.mass[i] = body.mass

            inertia_diag = body.get_inertia_diag()
            self.inertia_body[i] = np.diag(inertia_diag)
            self.inv_inertia_body[i] = np.diag(1.0 / np.maximum(inertia_diag, 1e-8))

            self.mesh_colors[i] = body.color

            self.index_count[i] = body.index_count
            self.index_offset[i] = i_offset

            body.setup_mesh(self.mesh_indices, i_offset, v_offset)

            v_offset += body.vertex_count
            i_offset += body.index_count

        self.update_mesh_vertices()

    def get_state(self):
        pos = self.position.to_numpy().copy()
        vel = self.velocity.to_numpy().copy()
        rot = self.rotation.to_numpy().copy()
        ang_vel = self.angular_velocity.to_numpy().copy()
        return pos, vel, rot, ang_vel

    def set_state(self, pos, vel, rot, ang_vel):
        n = self.num_bodies[None]
        for i in range(n):
            self.position[i] = pos[i]
            self.velocity[i] = vel[i]
            self.rotation[i] = rot[i]
            self.angular_velocity[i] = ang_vel[i]

    def update_mesh_vertices(self):
        v_offset = 0
        for i, body in enumerate(self.bodies):
            body.update_mesh_vertices(
                self.position, self.rotation, i, self.mesh_vertices, v_offset
            )
            v_offset += body.vertex_count


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

                    F = V_tangent * 20.0

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


@hydra.main(config_path="cfg", config_name="single", version_base=None)
def main(cfg):
    scene_cfg = OmegaConf.to_container(cfg.scene, resolve=True)
    scene = Scene(scene_cfg)
    simulator = Simulator(scene)

    steps = scene_cfg["steps"]
    simulator.run(steps)


if __name__ == "__main__":
    main()
