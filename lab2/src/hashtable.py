import math
import taichi as ti


@ti.data_oriented
class HashTable:
    def __init__(
        self,
        domain_size: tuple[float, float, float],
        particle_radius: float,
        max_particles: int,
    ):
        self.cell_size = float(particle_radius) * 2.2
        sx, sy, sz = (float(v) for v in domain_size)
        self.resolution = (
            max(1, int(math.ceil(sx / self.cell_size))),
            max(1, int(math.ceil(sy / self.cell_size))),
            max(1, int(math.ceil(sz / self.cell_size))),
        )
        hx, hy, hz = self.resolution

        self.max_particles = int(max_particles)
        self.num_cells = hx * hy * hz

        self.particle_cell_id = ti.field(dtype=ti.i32, shape=self.max_particles)
        self.cell_count = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.cell_offset = ti.field(dtype=ti.i32, shape=self.num_cells + 1)
        self.cell_cursor = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.particle_id = ti.field(dtype=ti.i32, shape=self.max_particles)

    @ti.func
    def _coord_from_pos(self, pos: ti.types.vector(3, ti.f32)):  # type: ignore
        hx, hy, hz = self.resolution
        x = ti.cast(ti.floor(pos[0] / self.cell_size), ti.i32)
        y = ti.cast(ti.floor(pos[1] / self.cell_size), ti.i32)
        z = ti.cast(ti.floor(pos[2] / self.cell_size), ti.i32)
        x = ti.max(0, ti.min(hx - 1, x))
        y = ti.max(0, ti.min(hy - 1, y))
        z = ti.max(0, ti.min(hz - 1, z))
        return x, y, z

    @ti.func
    def _flatten(self, x: ti.i32, y: ti.i32, z: ti.i32) -> ti.i32:  # type: ignore
        hx, hy, hz = self.resolution
        return (x * hy + y) * hz + z

    @ti.kernel
    def _count_particles(
        self, particle_pos: ti.template(), num_particles: ti.i32  # type: ignore
    ):
        for h in range(self.num_cells):
            self.cell_count[h] = 0
            self.cell_offset[h] = 0
            self.cell_cursor[h] = 0

        for p in range(num_particles):
            x, y, z = self._coord_from_pos(particle_pos[p])
            h = self._flatten(x, y, z)
            self.particle_cell_id[p] = h
            ti.atomic_add(self.cell_count[h], 1)

    @ti.kernel
    def _build_offsets(self):
        ti.loop_config(serialize=True)
        running = 0
        for h in range(self.num_cells):
            self.cell_offset[h] = running
            self.cell_cursor[h] = running
            running += self.cell_count[h]
        self.cell_offset[self.num_cells] = running

    @ti.kernel
    def _scatter_particles(self, num_particles: ti.i32):  # type: ignore
        for p in range(num_particles):
            h = self.particle_cell_id[p]
            idx = ti.atomic_add(self.cell_cursor[h], 1)
            self.particle_id[idx] = p

    def rebuild(self, particle_pos, num_particles: int) -> None:
        self._count_particles(particle_pos, int(num_particles))
        self._build_offsets()
        self._scatter_particles(int(num_particles))
