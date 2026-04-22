import numpy as np
import taichi as ti
import math
from constants import CellType, bbox_verts, bbox_indices
from util import HashTable


@ti.data_oriented
class Scene:
    def __init__(self, scene_cfg):
        gravity = np.array(scene_cfg["gravity"], dtype=np.float32)
        self.gravity = ti.Vector([gravity[0], gravity[1], gravity[2]])
        particles_cfg = scene_cfg["particles"]
        grid_cfg = scene_cfg["grid"]
        rigid_cfg = scene_cfg["rigidbodies"]

        self._init_grid(grid_cfg)
        self._init_particle(particles_cfg)
        self._init_rigidbody(rigid_cfg)

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

        self.grid_pressure = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
        self.grid_divergence = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
        self.grid_density = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
        self.grid_cell_type = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
        self.grid_solid_velocity = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))

        self.bbox_vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
        self.bbox_indices = ti.field(dtype=ti.i32, shape=24)

        self._initialize_grid()
        self._init_bbox()

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

        self.hashtable = HashTable(
            self.grid_size, self.particle_radius, self.num_particles
        )
        self.num_water_grid = ti.field(dtype=ti.i32, shape=())
        self.avg_density = ti.field(dtype=ti.f32, shape=())
        self.density_sum = ti.field(dtype=ti.f32, shape=())
        self.avg_density[None] = 0

        self._initialize_particles()
        self.update_cell_type()

    def _init_rigidbody(self, rigid_cfg):
        self.num_rigidbodies = len(rigid_cfg)
        self.rigidbody_capacity = max(1, self.num_rigidbodies)

        self.rigid_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.rigidbody_capacity)
        self.rigid_vel = ti.Vector.field(3, dtype=ti.f32, shape=self.rigidbody_capacity)
        self.rigid_orientation = ti.Vector.field(
            4, dtype=ti.f32, shape=self.rigidbody_capacity
        )
        self.rigid_angular_vel = ti.Vector.field(
            3, dtype=ti.f32, shape=self.rigidbody_capacity
        )
        self.rigid_dynamic = ti.field(dtype=ti.i32, shape=self.rigidbody_capacity)

        # Shape tags reserved for future fluid-solid coupling:
        # 0: none, 1: sphere, 2: box, 3: capsule, ...
        self.rigid_shape_type = ti.field(dtype=ti.i32, shape=self.rigidbody_capacity)
        self.rigid_radius = ti.field(dtype=ti.f32, shape=self.rigidbody_capacity)
        self.rigid_half_extent = ti.Vector.field(
            3, dtype=ti.f32, shape=self.rigidbody_capacity
        )

        self.rigid_count = ti.field(dtype=ti.i32, shape=())
        self.rigid_count[None] = self.num_rigidbodies
        self._initialize_rigidbodies()

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
        for I in ti.grouped(self.grid_pressure):
            self.grid_pressure[I] = 0.0
            self.grid_divergence[I] = 0.0
            self.grid_density[I] = 0.0
            i, j, k = I[0], I[1], I[2]
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
                self.grid_cell_type[I] = CellType.CELL_SOLID.value
            else:
                self.grid_cell_type[I] = CellType.CELL_AIR.value
            self.grid_solid_velocity[I] = ti.Vector([0.0, 0.0, 0.0])

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

        # Keep initialization away from the 1-cell solid boundary shell.
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
    def _initialize_particles(self):
        for p in range(self.num_particles):
            self.particle_pos[p] = self.particle_init_pos[p]
            self.particle_vel[p] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def update_cell_type(self):
        self.num_water_grid[None] = 0
        for I in ti.grouped(self.grid_cell_type):
            if self.grid_cell_type[I] == CellType.CELL_WATER.value:
                self.grid_cell_type[I] = CellType.CELL_AIR.value
        for p in range(self.num_particles):
            x = ti.cast(ti.floor(self.particle_pos[p][0] / self.grid_dx), ti.i32)
            y = ti.cast(ti.floor(self.particle_pos[p][1] / self.grid_dy), ti.i32)
            z = ti.cast(ti.floor(self.particle_pos[p][2] / self.grid_dz), ti.i32)
            if self.grid_cell_type[x, y, z] == CellType.CELL_AIR.value:
                self.grid_cell_type[x, y, z] = CellType.CELL_WATER.value
        for I in ti.grouped(self.grid_cell_type):
            if self.grid_cell_type[I] == CellType.CELL_WATER.value:
                ti.atomic_add(self.num_water_grid[None], 1)

    @ti.kernel
    def _initialize_rigidbodies(self):
        for i in range(self.rigidbody_capacity):
            self.rigid_pos[i] = ti.Vector([0.0, 0.0, 0.0])
            self.rigid_vel[i] = ti.Vector([0.0, 0.0, 0.0])
            self.rigid_orientation[i] = ti.Vector([1.0, 0.0, 0.0, 0.0])
            self.rigid_angular_vel[i] = ti.Vector([0.0, 0.0, 0.0])
            self.rigid_dynamic[i] = 0
            self.rigid_shape_type[i] = 0
            self.rigid_radius[i] = 0.0
            self.rigid_half_extent[i] = ti.Vector([0.0, 0.0, 0.0])

    def reset(self):
        self._initialize_grid()
        self._initialize_particles()
        self.update_cell_type()
        self._initialize_rigidbodies()
