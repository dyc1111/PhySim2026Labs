import math
import numpy as np
import taichi as ti

from constants import CellType, bbox_indices, bbox_verts
from rigidbody import create_rigid_body
from util import HashTable, skew_symmetric


@ti.data_oriented
class Scene:
    def __init__(self, scene_cfg):
        gravity = np.array(scene_cfg["gravity"], dtype=np.float32)
        self.gravity = ti.Vector([gravity[0], gravity[1], gravity[2]])
        particles_cfg = scene_cfg["particles"]
        grid_cfg = scene_cfg["grid"]
        obj_cfg = scene_cfg.get("object", [])
        if obj_cfg is None:
            obj_cfg = []

        self._init_grid(grid_cfg)
        self._init_rigidbody(obj_cfg)
        self._init_particle(particles_cfg)

    def _init_grid(self, grid_cfg):
        self.grid_size = tuple(float(x) for x in grid_cfg["size"])
        self.grid_resolution = tuple(int(x) for x in grid_cfg["resolution"])
        x, y, z = self.grid_size
        nx, ny, nz = self.grid_resolution
        self.grid_dx = x / nx
        self.grid_dy = y / ny
        self.grid_dz = z / nz
        assert math.isclose(self.grid_dx, self.grid_dy) and math.isclose(
            self.grid_dy, self.grid_dz
        ), "scene.grid.size and scene.grid.resolution incompatible"

        self.grid_u = ti.field(dtype=ti.f32, shape=(nx + 1, ny, nz))
        self.grid_v = ti.field(dtype=ti.f32, shape=(nx, ny + 1, nz))
        self.grid_w = ti.field(dtype=ti.f32, shape=(nx, ny, nz + 1))

        self.grid_u_prev = ti.field(dtype=ti.f32, shape=(nx + 1, ny, nz))
        self.grid_v_prev = ti.field(dtype=ti.f32, shape=(nx, ny + 1, nz))
        self.grid_w_prev = ti.field(dtype=ti.f32, shape=(nx, ny, nz + 1))

        self.grid_u_num = ti.field(dtype=ti.f32, shape=(nx + 1, ny, nz))
        self.grid_v_num = ti.field(dtype=ti.f32, shape=(nx, ny + 1, nz))
        self.grid_w_num = ti.field(dtype=ti.f32, shape=(nx, ny, nz + 1))

        self.grid_u_denom = ti.field(dtype=ti.f32, shape=(nx + 1, ny, nz))
        self.grid_v_denom = ti.field(dtype=ti.f32, shape=(nx, ny + 1, nz))
        self.grid_w_denom = ti.field(dtype=ti.f32, shape=(nx, ny, nz + 1))

        self.grid_density = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
        self.grid_cell_type = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
        self.grid_solid_velocity = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))

        self.bbox_vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
        self.bbox_indices = ti.field(dtype=ti.i32, shape=24)

        self._init_cell_type_cache()
        self._initialize_grid()
        self._init_bbox()

    def _init_cell_type_cache(self):
        nx, ny, nz = self.grid_resolution

        x = (np.arange(nx, dtype=np.float32) + 0.5) * self.grid_dx
        y = (np.arange(ny, dtype=np.float32) + 0.5) * self.grid_dy
        z = (np.arange(nz, dtype=np.float32) + 0.5) * self.grid_dz
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        centers = np.stack([xx, yy, zz], axis=-1).astype(np.float32)
        self._grid_cell_centers_flat = centers.reshape(-1, 3)

        cell_type_base = np.full((nx, ny, nz), CellType.CELL_AIR.value, dtype=np.int32)
        cell_type_base[0, :, :] = CellType.CELL_SOLID.value
        cell_type_base[nx - 1, :, :] = CellType.CELL_SOLID.value
        cell_type_base[:, 0, :] = CellType.CELL_SOLID.value
        cell_type_base[:, ny - 1, :] = CellType.CELL_SOLID.value
        cell_type_base[:, :, 0] = CellType.CELL_SOLID.value
        cell_type_base[:, :, nz - 1] = CellType.CELL_SOLID.value
        self._grid_cell_type_base = cell_type_base

        self._grid_cell_inflate = np.float32(
            0.5
            * math.sqrt(
                self.grid_dx * self.grid_dx
                + self.grid_dy * self.grid_dy
                + self.grid_dz * self.grid_dz
            )
        )

    def _init_particle(self, particles_cfg):
        self.particle_radius = self.grid_dx * 0.25
        positions = self._build_initial_particle_positions(particles_cfg)
        self.num_particles = int(positions.shape[0])

        self.particle_init_pos = ti.Vector.field(
            3, dtype=ti.f32, shape=self.num_particles
        )
        self.particle_init_pos.from_numpy(positions)
        self.particle_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles)
        self.particle_vel = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles)
        self.particle_color = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles)

        self.hashtable = HashTable(
            self.grid_size, self.particle_radius, self.num_particles
        )
        self.num_water_grid = ti.field(dtype=ti.i32, shape=())
        self.avg_density = ti.field(dtype=ti.f32, shape=())
        self.density_sum = ti.field(dtype=ti.f32, shape=())
        self.avg_density[None] = 0.0
        self.density_sum[None] = 0.0

        self._initialize_particles()
        self.update_cell_type()

    def _init_rigidbody(self, obj_cfg):
        self.rigid_bodies = [create_rigid_body(cfg) for cfg in obj_cfg]
        self.num_rigidbodies = len(self.rigid_bodies)
        self.rigidbody_capacity = max(1, self.num_rigidbodies)

        self.rigid_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.rigidbody_capacity)
        self.rigid_vel = ti.Vector.field(3, dtype=ti.f32, shape=self.rigidbody_capacity)
        self.rigid_rot = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=self.rigidbody_capacity
        )
        self.rigid_ang_vel = ti.Vector.field(
            3, dtype=ti.f32, shape=self.rigidbody_capacity
        )
        self.rigid_mass = ti.field(dtype=ti.f32, shape=self.rigidbody_capacity)
        self.rigid_inv_mass = ti.field(dtype=ti.f32, shape=self.rigidbody_capacity)
        self.rigid_inertia_body = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=self.rigidbody_capacity
        )
        self.rigid_inv_inertia_body = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=self.rigidbody_capacity
        )
        self.rigid_count = ti.field(dtype=ti.i32, shape=())
        self.rigid_count[None] = self.num_rigidbodies

        self.rigid_init_pos = ti.Vector.field(
            3, dtype=ti.f32, shape=self.rigidbody_capacity
        )
        self.rigid_init_vel = ti.Vector.field(
            3, dtype=ti.f32, shape=self.rigidbody_capacity
        )
        self.rigid_init_rot = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=self.rigidbody_capacity
        )
        self.rigid_init_ang_vel = ti.Vector.field(
            3, dtype=ti.f32, shape=self.rigidbody_capacity
        )

        self._initialize_rigidbodies()
        self._load_rigidbodies()
        self.rigid_init_pos.copy_from(self.rigid_pos)
        self.rigid_init_vel.copy_from(self.rigid_vel)
        self.rigid_init_rot.copy_from(self.rigid_rot)
        self.rigid_init_ang_vel.copy_from(self.rigid_ang_vel)
        self._init_rigidbody_meshes()

    @ti.kernel
    def _initialize_grid(self):
        nx, ny, nz = self.grid_resolution
        for I in ti.grouped(self.grid_u):
            self.grid_u[I] = 0.0
            self.grid_u_prev[I] = 0.0
            self.grid_u_num[I] = 0.0
            self.grid_u_denom[I] = 0.0
        for I in ti.grouped(self.grid_v):
            self.grid_v[I] = 0.0
            self.grid_v_prev[I] = 0.0
            self.grid_v_num[I] = 0.0
            self.grid_v_denom[I] = 0.0
        for I in ti.grouped(self.grid_w):
            self.grid_w[I] = 0.0
            self.grid_w_prev[I] = 0.0
            self.grid_w_num[I] = 0.0
            self.grid_w_denom[I] = 0.0

        for I in ti.grouped(self.grid_density):
            i, j, k = I
            self.grid_density[I] = 0.0
            self.grid_solid_velocity[I] = ti.Vector([0.0, 0.0, 0.0])
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
                self.grid_cell_type[I] = CellType.CELL_SOLID.value
            else:
                self.grid_cell_type[I] = CellType.CELL_AIR.value

    @ti.kernel
    def _initialize_particles(self):
        for p in range(self.num_particles):
            self.particle_pos[p] = self.particle_init_pos[p]
            self.particle_vel[p] = ti.Vector([0.0, 0.0, 0.0])
            self.particle_color[p] = ti.Vector([0.0, 0.0, 1.0])

    @ti.kernel
    def _initialize_rigidbodies(self):
        for i in range(self.rigidbody_capacity):
            self.rigid_pos[i] = ti.Vector([0.0, 0.0, 0.0])
            self.rigid_vel[i] = ti.Vector([0.0, 0.0, 0.0])
            self.rigid_rot[i] = ti.Matrix.identity(ti.f32, 3)
            self.rigid_ang_vel[i] = ti.Vector([0.0, 0.0, 0.0])
            self.rigid_mass[i] = 0.0
            self.rigid_inv_mass[i] = 0.0
            self.rigid_inertia_body[i] = ti.Matrix.identity(ti.f32, 3)
            self.rigid_inv_inertia_body[i] = ti.Matrix.zero(ti.f32, 3, 3)

    def _load_rigidbodies(self):
        cap = self.rigidbody_capacity
        pos = np.zeros((cap, 3), dtype=np.float32)
        vel = np.zeros((cap, 3), dtype=np.float32)
        rot = np.tile(np.eye(3, dtype=np.float32), (cap, 1, 1))
        ang_vel = np.zeros((cap, 3), dtype=np.float32)
        mass = np.zeros((cap,), dtype=np.float32)
        inv_mass = np.zeros((cap,), dtype=np.float32)
        inertia_body = np.tile(np.eye(3, dtype=np.float32), (cap, 1, 1))
        inv_inertia_body = np.zeros((cap, 3, 3), dtype=np.float32)

        for i, body in enumerate(self.rigid_bodies):
            pos[i] = body.position
            vel[i] = body.velocity
            rot[i] = body.rotation
            ang_vel[i] = body.angular_velocity

            if body.inv_mass > 0.0:
                mass[i] = float(body.mass)
                inv_mass[i] = float(body.inv_mass)
                inertia_diag = body.get_inertia_diag()
                inertia_body[i] = np.diag(inertia_diag)
                inv_inertia_body[i] = np.diag(1.0 / np.maximum(inertia_diag, 1e-8))
            else:
                mass[i] = 0.0
                inv_mass[i] = 0.0
                inertia_body[i] = np.eye(3, dtype=np.float32)
                inv_inertia_body[i] = np.zeros((3, 3), dtype=np.float32)

        self.rigid_pos.from_numpy(pos)
        self.rigid_vel.from_numpy(vel)
        self.rigid_rot.from_numpy(rot)
        self.rigid_ang_vel.from_numpy(ang_vel)
        self.rigid_mass.from_numpy(mass)
        self.rigid_inv_mass.from_numpy(inv_mass)
        self.rigid_inertia_body.from_numpy(inertia_body)
        self.rigid_inv_inertia_body.from_numpy(inv_inertia_body)

    def _init_rigidbody_meshes(self):
        n = self.num_rigidbodies
        self._rb_local_vertices = []
        self._rb_vertex_offsets = []
        self._mesh_total_vertices = 0

        if n == 0:
            self.mesh_vertices = ti.Vector.field(3, dtype=ti.f32, shape=1)
            self.mesh_indices = ti.field(dtype=ti.i32, shape=1)
            self.mesh_colors = ti.Vector.field(3, dtype=ti.f32, shape=1)
            self.index_offset = ti.field(dtype=ti.i32, shape=1)
            self.index_count = ti.field(dtype=ti.i32, shape=1)
            self.mesh_vertices[0] = ti.Vector([0.0, 0.0, 0.0])
            self.mesh_indices[0] = 0
            self.mesh_colors[0] = ti.Vector([0.0, 0.0, 0.0])
            self.index_offset[0] = 0
            self.index_count[0] = 0
            return

        total_vertices = 0
        total_indices = 0
        for body in self.rigid_bodies:
            local_verts, local_indices = body.get_local_mesh()
            self._rb_local_vertices.append(local_verts.astype(np.float32))
            self._rb_vertex_offsets.append(total_vertices)
            total_vertices += int(local_verts.shape[0])
            total_indices += int(local_indices.shape[0])

        self._mesh_total_vertices = total_vertices
        self.mesh_vertices = ti.Vector.field(3, dtype=ti.f32, shape=total_vertices)
        self.mesh_indices = ti.field(dtype=ti.i32, shape=total_indices)
        self.mesh_colors = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.index_offset = ti.field(dtype=ti.i32, shape=n)
        self.index_count = ti.field(dtype=ti.i32, shape=n)

        mesh_indices = np.zeros((total_indices,), dtype=np.int32)
        mesh_colors = np.zeros((n, 3), dtype=np.float32)
        index_offset = np.zeros((n,), dtype=np.int32)
        index_count = np.zeros((n,), dtype=np.int32)

        idx_cursor = 0
        vtx_cursor = 0
        for i, body in enumerate(self.rigid_bodies):
            local_verts, local_indices = body.get_local_mesh()
            count = int(local_indices.shape[0])
            mesh_indices[idx_cursor : idx_cursor + count] = local_indices + vtx_cursor
            index_offset[i] = idx_cursor
            index_count[i] = count
            mesh_colors[i] = body.color
            idx_cursor += count
            vtx_cursor += int(local_verts.shape[0])

        self.mesh_indices.from_numpy(mesh_indices)
        self.mesh_colors.from_numpy(mesh_colors)
        self.index_offset.from_numpy(index_offset)
        self.index_count.from_numpy(index_count)
        self.update_rigidbody_mesh_vertices()

    def update_rigidbody_mesh_vertices(self):
        if self.num_rigidbodies == 0:
            return
        pos, _, rot, _ = self.get_rigidbody_state()
        world_vertices = np.zeros((self._mesh_total_vertices, 3), dtype=np.float32)
        for i, local_verts in enumerate(self._rb_local_vertices):
            offset = self._rb_vertex_offsets[i]
            count = local_verts.shape[0]
            world_vertices[offset : offset + count] = (rot[i] @ local_verts.T).T + pos[
                i
            ].reshape(1, 3)
        self.mesh_vertices.from_numpy(world_vertices)

    def _init_bbox(self):
        sx, sy, sz = self.grid_size
        sx -= 2 * self.grid_dx
        sy -= 2 * self.grid_dy
        sz -= 2 * self.grid_dz
        verts = bbox_verts * np.array([sx, sy, sz], dtype=np.float32).reshape(1, -1)
        verts += np.ones_like(verts) * self.grid_dx
        self.bbox_vertices.from_numpy(verts)
        self.bbox_indices.from_numpy(bbox_indices)

    def _build_initial_particle_positions(self, particles_cfg):
        range_low = np.array(particles_cfg["low"], dtype=np.float32)
        range_high = np.array(particles_cfg["high"], dtype=np.float32)
        assert np.all(
            range_low <= range_high
        ), "scene.particles.low must be <= scene.particles.high"

        r = self.particle_radius
        a = 2.0 * r
        row_pitch = math.sqrt(3.0) * 0.5 * a
        layer_pitch = math.sqrt(2.0 / 3.0) * a
        layer_y_shift = row_pitch / 3.0
        eps = 1e-6

        domain_low = np.array(
            [self.grid_dx + r, self.grid_dy + r, self.grid_dz + r], dtype=np.float32
        )
        domain_high = np.array(
            [
                self.grid_size[0] - self.grid_dx - r,
                self.grid_size[1] - self.grid_dy - r,
                self.grid_size[2] - self.grid_dz - r,
            ],
            dtype=np.float32,
        )
        spawn_low = np.maximum(range_low, domain_low)
        spawn_high = np.minimum(range_high, domain_high)
        if np.any(spawn_low > spawn_high):
            return np.zeros((0, 3), dtype=np.float32)

        positions = []
        layer_idx = 0
        z = float(spawn_low[2])
        while z <= float(spawn_high[2]) + eps:
            layer_shift_x = 0.5 * a if (layer_idx & 1) else 0.0
            layer_shift_y = layer_y_shift if (layer_idx & 1) else 0.0

            row_idx = 0
            y = float(spawn_low[1]) + layer_shift_y
            while y <= float(spawn_high[1]) + eps:
                row_shift_x = 0.5 * a if (row_idx & 1) else 0.0
                x = float(spawn_low[0]) + layer_shift_x + row_shift_x
                while x <= float(spawn_high[0]) + eps:
                    positions.append([x, y, z])
                    x += a
                row_idx += 1
                y = float(spawn_low[1]) + layer_shift_y + row_idx * row_pitch

            layer_idx += 1
            z = float(spawn_low[2]) + layer_idx * layer_pitch

        return np.array(positions, dtype=np.float32)

    @ti.kernel
    def _mark_water_cells(self):
        nx, ny, nz = self.grid_resolution
        self.num_water_grid[None] = 0

        for p in range(self.num_particles):
            x = ti.cast(ti.floor(self.particle_pos[p][0] / self.grid_dx), ti.i32)
            y = ti.cast(ti.floor(self.particle_pos[p][1] / self.grid_dy), ti.i32)
            z = ti.cast(ti.floor(self.particle_pos[p][2] / self.grid_dz), ti.i32)
            x = ti.max(0, ti.min(nx - 1, x))
            y = ti.max(0, ti.min(ny - 1, y))
            z = ti.max(0, ti.min(nz - 1, z))
            if self.grid_cell_type[x, y, z] == CellType.CELL_AIR.value:
                self.grid_cell_type[x, y, z] = CellType.CELL_WATER.value

        for I in ti.grouped(self.grid_cell_type):
            if self.grid_cell_type[I] == CellType.CELL_WATER.value:
                ti.atomic_add(self.num_water_grid[None], 1)

    def update_cell_type(self):
        nx, ny, nz = self.grid_resolution
        cell_type = self._grid_cell_type_base.copy()
        solid_velocity = np.zeros((nx, ny, nz, 3), dtype=np.float32)

        if self.num_rigidbodies > 0:
            centers = self._grid_cell_centers_flat
            cell_type_flat = cell_type.reshape(-1)
            solid_velocity_flat = solid_velocity.reshape(-1, 3)
            pos, vel, rot, ang_vel = self.get_rigidbody_state()

            for body_id, body in enumerate(self.rigid_bodies):
                mask = body.intersects_grid_cells(
                    centers,
                    pos[body_id],
                    rot[body_id],
                    self._grid_cell_inflate,
                )
                if not np.any(mask):
                    continue
                cell_type_flat[mask] = CellType.CELL_SOLID.value
                solid_velocity_flat[mask] = body.sample_solid_velocity(
                    centers[mask],
                    pos[body_id],
                    vel[body_id],
                    ang_vel[body_id],
                )

        self.grid_cell_type.from_numpy(cell_type)
        self.grid_solid_velocity.from_numpy(solid_velocity)
        self._mark_water_cells()

    @ti.kernel
    def pre_solve_kinematics(
        self, dt: ti.f32, forces: ti.types.ndarray(), torques: ti.types.ndarray()  # type: ignore
    ):
        for i in range(self.rigid_count[None]):
            if self.rigid_inv_mass[i] > 0.0:
                force = ti.Vector([forces[i, 0], forces[i, 1], forces[i, 2]])
                self.rigid_vel[i] += dt * force * self.rigid_inv_mass[i]

                torque = ti.Vector([torques[i, 0], torques[i, 1], torques[i, 2]])
                rot = self.rigid_rot[i]
                i_inv_world = rot @ self.rigid_inv_inertia_body[i] @ rot.transpose()
                i_world = rot @ self.rigid_inertia_body[i] @ rot.transpose()
                omega = self.rigid_ang_vel[i]
                torque_total = torque - omega.cross(i_world @ omega)
                self.rigid_ang_vel[i] += dt * (i_inv_world @ torque_total)

    @ti.kernel
    def post_solve_kinematics(self, dt: ti.f32):  # type: ignore
        for i in range(self.rigid_count[None]):
            if self.rigid_inv_mass[i] > 0.0:
                self.rigid_pos[i] += dt * self.rigid_vel[i]

                omega = self.rigid_ang_vel[i]
                dtheta = omega * dt
                theta = dtheta.norm()
                delta = ti.Matrix.identity(ti.f32, 3)
                if theta < 1e-7:
                    delta += skew_symmetric(dtheta)
                else:
                    axis = dtheta / theta
                    k = skew_symmetric(axis)
                    delta += ti.sin(theta) * k + (1.0 - ti.cos(theta)) * (k @ k)
                self.rigid_rot[i] = delta @ self.rigid_rot[i]

    def get_rigidbody_state(self):
        n = self.num_rigidbodies
        pos = self.rigid_pos.to_numpy()[:n]
        vel = self.rigid_vel.to_numpy()[:n]
        rot = self.rigid_rot.to_numpy()[:n]
        ang_vel = self.rigid_ang_vel.to_numpy()[:n]
        return pos, vel, rot, ang_vel

    def reset(self):
        self._initialize_grid()
        self._initialize_particles()
        if self.num_rigidbodies > 0:
            self.rigid_pos.copy_from(self.rigid_init_pos)
            self.rigid_vel.copy_from(self.rigid_init_vel)
            self.rigid_rot.copy_from(self.rigid_init_rot)
            self.rigid_ang_vel.copy_from(self.rigid_init_ang_vel)
            self.update_rigidbody_mesh_vertices()
        self.avg_density[None] = 0.0
        self.density_sum[None] = 0.0
        self.update_cell_type()

    @ti.kernel
    def reflect(self):
        for I in ti.grouped(self.grid_u):
            self.grid_u[I] = 2 * self.grid_u[I] - self.grid_u_prev[I]
        for I in ti.grouped(self.grid_v):
            self.grid_v[I] = 2 * self.grid_v[I] - self.grid_v_prev[I]
        for I in ti.grouped(self.grid_w):
            self.grid_w[I] = 2 * self.grid_w[I] - self.grid_w_prev[I]
