import taichi as ti
import numpy as np
from rigidbody import Cuboid


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
        self.inv_mass = ti.field(dtype=ti.f32, shape=n_bodies)
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
            self.inv_mass[i] = body.inv_mass

            inertia_diag = body.get_inertia_diag()
            self.inertia_body[i] = np.diag(inertia_diag)
            
            if body.dyn_type == "freeze":
                self.inv_inertia_body[i] = np.zeros((3, 3), dtype=np.float32)
            else:
                self.inv_inertia_body[i] = np.diag(1.0 / np.maximum(inertia_diag, 1e-8))

            self.mesh_colors[i] = body.color

            self.index_count[i] = body.index_count
            self.index_offset[i] = i_offset

            body.setup_mesh(self.mesh_indices, i_offset, v_offset)

            v_offset += body.vertex_count
            i_offset += body.index_count

        self.init_pos = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.init_vel = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.init_rot = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_bodies)
        self.init_ang_vel = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.init_pos.copy_from(self.position)
        self.init_vel.copy_from(self.velocity)
        self.init_rot.copy_from(self.rotation)
        self.init_ang_vel.copy_from(self.angular_velocity)

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

    def reset(self):
        self.position.copy_from(self.init_pos)
        self.velocity.copy_from(self.init_vel)
        self.rotation.copy_from(self.init_rot)
        self.angular_velocity.copy_from(self.init_ang_vel)
