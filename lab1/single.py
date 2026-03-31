import taichi as ti
import numpy as np
import hydra
from omegaconf import OmegaConf
from constants import *


ti.init(arch=ti.gpu)


def euler_xyz_deg_to_matrix(rot_deg):
    rx, ry, rz = np.radians(np.array(rot_deg, dtype=np.float32))

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz_m @ ry_m @ rx_m


def skew_symmetric(v):
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float32,
    )


def integrate_rotation(r, omega, dt):
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


def cuboid_inertia_diag(m, size):
    lx, ly, lz = size
    ixx = (m / 12.0) * (ly * ly + lz * lz)
    iyy = (m / 12.0) * (lx * lx + lz * lz)
    izz = (m / 12.0) * (lx * lx + ly * ly)
    return np.array([ixx, iyy, izz], dtype=np.float32)


@ti.data_oriented
class Scene:
    def __init__(self, scene_cfg):
        self.num_bodies = ti.field(dtype=ti.i32, shape=())
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
        self.rotation = ti.Matrix.field(3, 3, dtype=ti.f32, shape=MAX_BODIES)
        self.angular_velocity = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
        self.half_extent = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES)
        self.mass = ti.field(dtype=ti.f32, shape=MAX_BODIES)
        self.inertia_body = ti.Matrix.field(3, 3, dtype=ti.f32, shape=MAX_BODIES)
        self.inv_inertia_body = ti.Matrix.field(3, 3, dtype=ti.f32, shape=MAX_BODIES)

        self.mesh_vertices = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BODIES * 8)
        self.mesh_indices = ti.field(dtype=ti.i32, shape=MAX_BODIES * 36)
        self.cube_local_verts = ti.Vector.field(3, dtype=ti.f32, shape=8)
        self.cube_indices_ti = ti.field(dtype=ti.i32, shape=36)
        self.cube_local_verts.from_numpy(CUBE_LOCAL_VERTS_NP)
        self.cube_indices_ti.from_numpy(CUBE_INDICES_NP)

        self.dt = 1.0 / 240.0
        self.substeps = 1
        self.gravity = np.array([0.0, -9.8, 0.0], dtype=np.float32)
        self.linear_damping = 0.0
        self.angular_damping = 0.0

        self.initialize(scene_cfg)

    def initialize(self, scene_cfg):
        parsed = self.parse_scene_cfg(scene_cfg)
        (
            n_bodies,
            pos_np,
            vel_np,
            rot_np,
            ang_vel_np,
            half_ext_np,
            mass_np,
            inertia_np,
            inv_inertia_np,
            sim_cfg,
        ) = parsed

        self.dt = sim_cfg["dt"]
        self.substeps = sim_cfg["substeps"]
        self.gravity = sim_cfg["gravity"]
        self.linear_damping = sim_cfg["linear_damping"]
        self.angular_damping = sim_cfg["angular_damping"]

        self.num_bodies[None] = n_bodies
        for i in range(n_bodies):
            self.position[i] = pos_np[i]
            self.velocity[i] = vel_np[i]
            self.rotation[i] = rot_np[i]
            self.angular_velocity[i] = ang_vel_np[i]
            self.half_extent[i] = half_ext_np[i]
            self.mass[i] = float(mass_np[i])
            self.inertia_body[i] = inertia_np[i]
            self.inv_inertia_body[i] = inv_inertia_np[i]

        self.update_mesh_vertices()

    def parse_scene_cfg(self, scene_cfg):
        objects = list(scene_cfg.get("objects", scene_cfg.get("cubes", [])))
        if not objects:
            raise ValueError(
                "scene.objects (or scene.cubes) must contain at least one body"
            )

        n_bodies = len(objects)
        if n_bodies > MAX_BODIES:
            raise ValueError(
                f"scene has {n_bodies} bodies, exceeds MAX_BODIES={MAX_BODIES}"
            )

        default_side = float(scene_cfg.get("cube_side_length", 0.6))
        default_size = np.array(
            scene_cfg.get("default_size", [default_side, default_side, default_side]),
            dtype=np.float32,
        )
        default_mass = float(scene_cfg.get("default_mass", 1.0))

        pos_np = np.zeros((n_bodies, 3), dtype=np.float32)
        vel_np = np.zeros((n_bodies, 3), dtype=np.float32)
        ang_vel_np = np.zeros((n_bodies, 3), dtype=np.float32)
        half_ext_np = np.zeros((n_bodies, 3), dtype=np.float32)
        rot_np = np.zeros((n_bodies, 3, 3), dtype=np.float32)
        mass_np = np.zeros((n_bodies,), dtype=np.float32)
        inertia_np = np.zeros((n_bodies, 3, 3), dtype=np.float32)
        inv_inertia_np = np.zeros((n_bodies, 3, 3), dtype=np.float32)

        for i, body in enumerate(objects):
            if "size" in body:
                size = np.array(body["size"], dtype=np.float32)
            elif "side_length" in body:
                s = float(body["side_length"])
                size = np.array([s, s, s], dtype=np.float32)
            else:
                size = default_size.copy()

            if size.shape != (3,):
                raise ValueError(f"Body {i} size must be length-3, got {size}")

            pos_np[i] = np.array(
                body.get("position", [0.0, 0.0, 0.0]), dtype=np.float32
            )
            vel_np[i] = np.array(
                body.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32
            )
            ang_vel_np[i] = np.array(
                body.get("angular_velocity", [0.0, 0.0, 0.0]), dtype=np.float32
            )
            rot_np[i] = euler_xyz_deg_to_matrix(
                body.get("rotation_deg", [0.0, 0.0, 0.0])
            )
            half_ext_np[i] = 0.5 * size
            mass_np[i] = float(body.get("mass", default_mass))

            inertia_diag = cuboid_inertia_diag(mass_np[i], size)
            inertia_np[i] = np.diag(inertia_diag)
            inv_inertia_diag = 1.0 / np.maximum(inertia_diag, 1e-8)
            inv_inertia_np[i] = np.diag(inv_inertia_diag)

        sim_cfg = {
            "dt": float(scene_cfg.get("dt", 1.0 / 240.0)),
            "substeps": int(scene_cfg.get("substeps", 1)),
            "gravity": np.array(
                scene_cfg.get("gravity", [0.0, -9.8, 0.0]), dtype=np.float32
            ),
            "linear_damping": float(scene_cfg.get("linear_damping", 0.0)),
            "angular_damping": float(scene_cfg.get("angular_damping", 0.0)),
        }

        return (
            n_bodies,
            pos_np,
            vel_np,
            rot_np,
            ang_vel_np,
            half_ext_np,
            mass_np,
            inertia_np,
            inv_inertia_np,
            sim_cfg,
        )

    def get_state_numpy(self):
        n = self.num_bodies[None]
        return {
            "position": self.position.to_numpy()[:n].copy(),
            "velocity": self.velocity.to_numpy()[:n].copy(),
            "rotation": self.rotation.to_numpy()[:n].copy(),
            "angular_velocity": self.angular_velocity.to_numpy()[:n].copy(),
        }

    def set_state_numpy(self, state):
        n = self.num_bodies[None]
        for i in range(n):
            self.position[i] = state["position"][i]
            self.velocity[i] = state["velocity"][i]
            self.rotation[i] = state["rotation"][i]
            self.angular_velocity[i] = state["angular_velocity"][i]

    @ti.kernel
    def update_mesh_vertices(self):
        for i in range(self.num_bodies[None]):
            pos = self.position[i]
            rot = self.rotation[i]
            ext = self.half_extent[i]
            for k in range(8):
                lv = self.cube_local_verts[k]
                local = ti.Vector([lv[0] * ext[0], lv[1] * ext[1], lv[2] * ext[2]])
                world = rot @ local + pos
                self.mesh_vertices[i * 8 + k] = world
            for t in range(12):
                for v in range(3):
                    self.mesh_indices[i * 36 + t * 3 + v] = (
                        i * 8 + self.cube_indices_ti[t * 3 + v]
                    )


class Simulator:
    def __init__(self, scene):
        self.scene = scene

    def render(self, window, camera, canvas, scene_3d):
        scene_3d.set_camera(camera)
        scene_3d.ambient_light((0.6, 0.6, 0.6))
        scene_3d.point_light((5, 5, 5), (1.2, 1.2, 1.2))
        scene_3d.mesh(
            self.scene.mesh_vertices,
            indices=self.scene.mesh_indices,
            color=(0.6, 0.7, 0.9),
        )
        canvas.scene(scene_3d)
        window.show()

    def step(self):
        substeps = max(1, int(self.scene.substeps))
        dt = self.scene.dt / substeps

        state = self.scene.get_state_numpy()
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
                rot[i] = integrate_rotation(rot[i], ang_vel[i], dt)

        self.scene.set_state_numpy(
            {
                "position": pos,
                "velocity": vel,
                "rotation": rot,
                "angular_velocity": ang_vel,
            }
        )
        self.scene.update_mesh_vertices()

    def run(self, steps=1):
        window = ti.ui.Window("Rigid Body Simulation", (1280, 720), vsync=True)
        canvas = window.get_canvas()
        scene_3d = window.get_scene()
        camera = ti.ui.Camera()
        camera.position(0.0, 1.8, 4.5)
        camera.lookat(0.0, 0.0, 0.0)
        camera.up(0.0, 1.0, 0.0)

        frame = 0
        while window.running and frame < steps:
            self.step()
            self.render(window, camera, canvas, scene_3d)
            frame += 1


@hydra.main(config_path="cfg", config_name="single", version_base=None)
def main(cfg):
    scene_cfg = OmegaConf.to_container(cfg.scene, resolve=True)
    scene = Scene(scene_cfg)
    simulator = Simulator(scene)

    steps = int(scene_cfg.get("steps", 1))
    simulator.run(steps)

    n_bodies = scene.num_bodies[None]
    print(f"Initialized {n_bodies} body(ies)")
    print(
        f"dt={scene.dt}, substeps={scene.substeps}, gravity={scene.gravity.tolist()}, "
        f"linear_damping={scene.linear_damping}, angular_damping={scene.angular_damping}, steps={steps}"
    )


if __name__ == "__main__":
    main()
