import numpy as np
import taichi as ti
import math
from constants import CellType, particle_offset, bbox_verts, bbox_indices
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
        self._init_renderer()

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
        self.grid_particle_density = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
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
            self.grid_particle_density[I] = 0.0
            i, j, k = I[0], I[1], I[2]
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
                self.grid_cell_type[I] = CellType.CELL_SOLID.value
            else:
                self.grid_cell_type[I] = CellType.CELL_AIR.value
            self.grid_solid_velocity[I] = ti.Vector([0.0, 0.0, 0.0])

    def _init_renderer(self):
        self.window = ti.ui.Window("Fluid Simulation", (1280, 720), vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene_3d = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self._init_camera()

    def _init_camera(self):
        sx, sy, sz = self.grid_size
        center = (0.5 * sx, 0.5 * sy, 0.5 * sz)
        span = max(sx, sy, sz)
        self.camera.position(-0.9 * span, 0.9 * span, -0.9 * span)
        self.camera.lookat(center[0], center[1], center[2])
        self.camera.up(0.0, 1.0, 0.0)

    def _init_bbox(self):
        sx, sy, sz = self.grid_size
        verts = bbox_verts * np.array([sx, sy, sz], dtype=np.float32).reshape(1, -1)
        self.bbox_vertices.from_numpy(verts)
        self.bbox_indices.from_numpy(bbox_indices)

    def _build_initial_particle_positions(self, particles_cfg):
        range_low = np.array(particles_cfg["low"], dtype=np.float32)
        range_high = np.array(particles_cfg["high"], dtype=np.float32)
        assert np.all(
            range_low <= range_high
        ), "scene.particles.range_low must be <= range_high"

        nx, ny, nz = self.grid_resolution
        positions = []
        for i in range(1, nx - 1):
            cx = (i + 0.5) * self.grid_dx
            if cx < range_low[0] or cx > range_high[0]:
                continue
            for j in range(1, ny - 1):
                cy = (j + 0.5) * self.grid_dy
                if cy < range_low[1] or cy > range_high[1]:
                    continue
                for k in range(1, nz - 1):
                    cz = (k + 0.5) * self.grid_dz
                    if cz < range_low[2] or cz > range_high[2]:
                        continue

                    for ox, oy, oz in particle_offset:
                        positions.append(
                            [
                                (i + ox) * self.grid_dx,
                                (j + oy) * self.grid_dy,
                                (k + oz) * self.grid_dz,
                            ]
                        )

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
            if self.grid_cell_type[I] != CellType.CELL_SOLID.value:
                self.grid_cell_type[I] = CellType.CELL_AIR.value
        for p in range(self.num_particles):
            x = ti.cast(ti.floor(self.particle_pos[p][0] / self.grid_dx), ti.i32)
            y = ti.cast(ti.floor(self.particle_pos[p][1] / self.grid_dy), ti.i32)
            z = ti.cast(ti.floor(self.particle_pos[p][2] / self.grid_dz), ti.i32)
            if self.grid_cell_type[x, y, z] != CellType.CELL_SOLID.value:
                self.grid_cell_type[x, y, z] = CellType.CELL_WATER.value
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

    def _reset(self):
        self._initialize_grid()
        self._initialize_particles()
        self.update_cell_type()
        self._initialize_rigidbodies()
        self._init_camera()

    def render(self):
        if not self.window.running or self.window.is_pressed(ti.ui.ESCAPE):
            return False
        elif self.window.is_pressed(ti.ui.SPACE):
            self._reset()

        if not self.window.is_pressed(ti.ui.CTRL):
            self.camera.track_user_inputs(
                self.window, movement_speed=0.03, hold_key=ti.ui.LMB
            )

        self.scene_3d.set_camera(self.camera)
        self.scene_3d.ambient_light((0.6, 0.6, 0.6))
        self.scene_3d.point_light((5, 5, 5), (1.2, 1.2, 1.2))

        self.scene_3d.particles(
            self.particle_pos,
            radius=self.particle_radius,
            color=(0.1, 0.35, 0.95),
        )
        self.scene_3d.lines(
            self.bbox_vertices,
            width=2.0,
            indices=self.bbox_indices,
            color=(0.05, 0.05, 0.05),
        )

        self.canvas.scene(self.scene_3d)
        self.canvas.set_background_color((0.8, 0.8, 0.85))
        self.window.show()
        return True
