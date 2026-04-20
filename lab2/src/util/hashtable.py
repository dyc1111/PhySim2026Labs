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
            int(math.ceil(sx / self.cell_size)),
            int(math.ceil(sy / self.cell_size)),
            int(math.ceil(sz / self.cell_size)),
        )
        hx, hy, hz = self.resolution

        self.max_particles = max_particles
        self.num_cells = hx * hy * hz
        self.scan_size = 1 << (self.num_cells - 1).bit_length()

        self.particle_cell_id = ti.field(dtype=ti.i32, shape=self.max_particles)
        self.cell_count = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.cell_offset = ti.field(dtype=ti.i32, shape=self.num_cells + 1)
        self.cell_cursor = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.particle_id = ti.field(dtype=ti.i32, shape=self.max_particles)
        self.scan_buffer = ti.field(dtype=ti.i32, shape=self.scan_size)
        self.scan_total = ti.field(dtype=ti.i32, shape=())

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
    def flatten(self, x: ti.i32, y: ti.i32, z: ti.i32) -> ti.i32:  # type: ignore
        _, hy, hz = self.resolution
        return (x * hy + y) * hz + z

    @ti.func
    def neighbouring_cell_bounds(self, pos: ti.types.vector(3, ti.f32)):  # type: ignore
        x, y, z = self._coord_from_pos(pos)
        hx, hy, hz = self.resolution
        i0 = ti.max(0, x - 1)
        i1 = ti.min(hx - 1, x + 1)
        j0 = ti.max(0, y - 1)
        j1 = ti.min(hy - 1, y + 1)
        k0 = ti.max(0, z - 1)
        k1 = ti.min(hz - 1, z + 1)
        return i0, i1, j0, j1, k0, k1

    @ti.func
    def cell_begin(self, h: ti.i32) -> ti.i32:  # type: ignore
        return self.cell_offset[h]

    @ti.func
    def cell_end(self, h: ti.i32) -> ti.i32:  # type: ignore
        return self.cell_offset[h + 1]

    @ti.kernel
    def _clear(self):
        for h in range(self.num_cells):
            self.cell_count[h] = 0
            self.cell_offset[h] = 0
            self.cell_cursor[h] = 0
        self.cell_offset[self.num_cells] = 0

    @ti.kernel
    def _count_particles(
        self, particle_pos: ti.template(), num_particles: ti.i32  # type: ignore
    ):
        for p in range(num_particles):
            x, y, z = self._coord_from_pos(particle_pos[p])
            h = self.flatten(x, y, z)
            self.particle_cell_id[p] = h
            ti.atomic_add(self.cell_count[h], 1)

    @ti.kernel
    def _prepare_scan_buffer(self):
        for h in range(self.scan_size):
            if h < self.num_cells:
                self.scan_buffer[h] = self.cell_count[h]
            else:
                self.scan_buffer[h] = 0

    @ti.kernel
    def _scan_upsweep(self, stride: ti.i32):  # type: ignore
        step = stride * 2
        for i in range(self.scan_size // step):
            right = (i + 1) * step - 1
            left = right - stride
            self.scan_buffer[right] += self.scan_buffer[left]

    @ti.kernel
    def _scan_save_total(self):
        self.scan_total[None] = self.scan_buffer[self.scan_size - 1]

    @ti.kernel
    def _scan_set_last_zero(self):
        self.scan_buffer[self.scan_size - 1] = 0

    @ti.kernel
    def _scan_downsweep(self, stride: ti.i32):  # type: ignore
        step = stride * 2
        for i in range(self.scan_size // step):
            right = (i + 1) * step - 1
            left = right - stride
            tmp = self.scan_buffer[left]
            self.scan_buffer[left] = self.scan_buffer[right]
            self.scan_buffer[right] += tmp

    @ti.kernel
    def _write_offsets_from_scan(self):
        for h in range(self.num_cells):
            offset = self.scan_buffer[h]
            self.cell_offset[h] = offset
            self.cell_cursor[h] = offset
        self.cell_offset[self.num_cells] = self.scan_total[None]

    def _build_offsets(self):
        self._prepare_scan_buffer()

        stride = 1
        while stride < self.scan_size:
            self._scan_upsweep(stride)
            stride <<= 1

        self._scan_save_total()
        self._scan_set_last_zero()

        stride = self.scan_size >> 1
        while stride > 0:
            self._scan_downsweep(stride)
            stride >>= 1

        self._write_offsets_from_scan()

    @ti.kernel
    def _scatter_particles(self, num_particles: ti.i32):  # type: ignore
        for p in range(num_particles):
            h = self.particle_cell_id[p]
            idx = ti.atomic_add(self.cell_cursor[h], 1)
            self.particle_id[idx] = p

    def rebuild(self, particle_pos, num_particles):
        self._clear()
        self._count_particles(particle_pos, num_particles)
        self._build_offsets()
        self._scatter_particles(num_particles)
