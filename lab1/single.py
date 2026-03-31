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

    def cuboid_inertia_diag(m, size):
        lx, ly, lz = size
        ixx = (m / 12.0) * (ly * ly + lz * lz)
        iyy = (m / 12.0) * (lx * lx + lz * lz)
        izz = (m / 12.0) * (lx * lx + ly * ly)
        return np.array([ixx, iyy, izz], dtype=np.float32)

    def get_inertia_diag(self):
        lx, ly, lz = self.size
        ixx = (self.mass / 12.0) * (ly * ly + lz * lz)
        iyy = (self.mass / 12.0) * (lx * lx + lz * lz)
        izz = (self.mass / 12.0) * (lx * lx + ly * ly)
        return np.array([ixx, iyy, izz], dtype=np.float32)

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
        return {
            "position": self.position.to_numpy().copy(),
            "velocity": self.velocity.to_numpy().copy(),
            "rotation": self.rotation.to_numpy().copy(),
            "angular_velocity": self.angular_velocity.to_numpy().copy(),
        }

    def set_state(self, state):
        n = self.num_bodies[None]
        for i in range(n):
            self.position[i] = state["position"][i]
            self.velocity[i] = state["velocity"][i]
            self.rotation[i] = state["rotation"][i]
            self.angular_velocity[i] = state["angular_velocity"][i]

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

    def step(self):
        substeps = self.scene.substeps
        dt = self.scene.dt / substeps

        state = self.scene.get_state()
        pos = state["position"]
        vel = state["velocity"]
        rot = state["rotation"]
        ang_vel = state["angular_velocity"]

        for _ in range(substeps):
            vel += dt * self.scene.gravity[None, :]
            vel *= max(0.0, 1.0 - self.scene.linear_damping * dt)
            ang_vel *= max(0.0, 1.0 - self.scene.angular_damping * dt)
            pos += dt * vel

            for i in range(self.scene.num_bodies[None]):
                rot[i] = self._integrate_rotation(rot[i], ang_vel[i], dt)

        self.scene.set_state(
            {
                "position": pos,
                "velocity": vel,
                "rotation": rot,
                "angular_velocity": ang_vel,
            }
        )
        self.scene.update_mesh_vertices()

    def _integrate_rotation(r, omega, dt):
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

        frame = 0
        while window.running and frame < (steps if steps > 0 else float("inf")):
            self.step()
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
