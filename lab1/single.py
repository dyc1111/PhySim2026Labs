import taichi as ti
import numpy as np
import hydra
from omegaconf import OmegaConf
from constants import *
from util import *


ti.init(arch=ti.gpu)


@ti.data_oriented
class Scene:
    def __init__(self, scene_cfg):
        self.dt = scene_cfg["dt"]
        self.substeps = scene_cfg["substeps"]
        self.gravity = np.array(scene_cfg["gravity"], dtype=np.float32)
        self.linear_damping = scene_cfg["linear_damping"]
        self.angular_damping = scene_cfg["angular_damping"]

        objects = scene_cfg["objects"]
        self.num_bodies = ti.field(dtype=ti.i32, shape=())
        n_bodies = len(objects)
        self.num_bodies[None] = n_bodies

        self.position = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.rotation = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_bodies)
        self.angular_velocity = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.half_extent = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.mass = ti.field(dtype=ti.f32, shape=n_bodies)
        self.inertia_body = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_bodies)
        self.inv_inertia_body = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_bodies)

        self.mesh_vertices = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies * 8)
        self.mesh_indices = ti.field(dtype=ti.i32, shape=n_bodies * 36)
        self.mesh_colors = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.index_offset = ti.field(dtype=ti.i32, shape=n_bodies)
        self.index_count = ti.field(dtype=ti.i32, shape=n_bodies)

        self.cube_local_verts = ti.Vector.field(3, dtype=ti.f32, shape=8)
        self.cube_indices_ti = ti.field(dtype=ti.i32, shape=36)
        self.cube_local_verts.from_numpy(CUBE_LOCAL_VERTS_NP)
        self.cube_indices_ti.from_numpy(CUBE_INDICES_NP)

        for i, body in enumerate(objects):
            size = np.array(body["size"], dtype=np.float32)
            self.position[i] = np.array(body["position"], dtype=np.float32)
            self.velocity[i] = np.array(body["velocity"], dtype=np.float32)
            self.angular_velocity[i] = np.array(
                body["angular_velocity"], dtype=np.float32
            )
            self.rotation[i] = euler_angle_to_matrix(body["rotation_deg"])
            self.half_extent[i] = 0.5 * size
            self.mass[i] = float(body["mass"])
            inertia_diag = cuboid_inertia_diag(self.mass[i], size)
            self.inertia_body[i] = np.diag(inertia_diag)
            inv_inertia_diag = 1.0 / np.maximum(inertia_diag, 1e-8)
            self.inv_inertia_body[i] = np.diag(inv_inertia_diag)
            self.mesh_colors[i] = np.array(body["color"], dtype=np.float32)
            self.index_count[i] = 36
            if i == 0:
                self.index_offset[i] = 0
            else:
                self.index_offset[i] = self.index_offset[i - 1] + self.index_count[i]

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
                rot[i] = integrate_rotation(rot[i], ang_vel[i], dt)

        self.scene.set_state(
            {
                "position": pos,
                "velocity": vel,
                "rotation": rot,
                "angular_velocity": ang_vel,
            }
        )
        self.scene.update_mesh_vertices()

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
