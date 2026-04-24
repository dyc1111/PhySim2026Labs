from abc import ABC, abstractmethod
import taichi as ti
from scene import Scene
from constants import CellType


class DensityStrategyBase(ABC):
    @abstractmethod
    def handle_density(self):
        """Update per-particle or per-cell density statistics."""
        return NotImplementedError


@ti.data_oriented
class DensityStrategy(DensityStrategyBase):
    def __init__(self, scene: Scene):
        self.scene = scene

    @ti.kernel
    def _calc_density(self):
        for I in ti.grouped(self.scene.grid_density):
            self.scene.grid_density[I] = 0.0

        dx, dy, dz = self.scene.grid_dx, self.scene.grid_dy, self.scene.grid_dz
        nx, ny, nz = self.scene.grid_resolution
        for p in range(self.scene.num_particles):
            pos = self.scene.particle_pos[p]
            xh = pos[0] - dx / 2
            yh = pos[1] - dy / 2
            zh = pos[2] - dz / 2

            ih = ti.cast(ti.floor(xh / dx), ti.i32)
            jh = ti.cast(ti.floor(yh / dy), ti.i32)
            kh = ti.cast(ti.floor(zh / dz), ti.i32)

            fx = xh / dx - ih
            fy = yh / dy - jh
            fz = zh / dz - kh

            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        wx = fx if di else (1 - fx)
                        wy = fy if dj else (1 - fy)
                        wz = fz if dk else (1 - fz)
                        w = wx * wy * wz

                        if (
                            0 <= ih + di < nx
                            and 0 <= jh + dj < ny
                            and 0 <= kh + dk < nz
                        ):
                            ti.atomic_add(
                                self.scene.grid_density[ih + di, jh + dj, kh + dk], w
                            )

    @ti.kernel
    def _init_density(self):
        self.scene.density_sum[None] = 0
        for I in ti.grouped(self.scene.grid_density):
            if self.scene.grid_cell_type[I] == CellType.CELL_WATER.value:
                ti.atomic_add(self.scene.density_sum[None], self.scene.grid_density[I])
        self.scene.avg_density[None] = (
            self.scene.density_sum[None] / self.scene.num_water_grid[None]
        )

    @ti.kernel
    def _update_particle_color(self):
        nx, ny, nz = self.scene.grid_resolution
        for p in range(self.scene.num_particles):
            color = self.scene.particle_color[p]
            fade = 0.01
            color[0] = ti.max(0.0, color[0] - fade)
            color[1] = ti.max(0.0, color[1] - fade)
            color[2] = ti.min(1.0, color[2] + fade)

            x = ti.cast(
                ti.floor(self.scene.particle_pos[p][0] / self.scene.grid_dx), ti.i32
            )
            y = ti.cast(
                ti.floor(self.scene.particle_pos[p][1] / self.scene.grid_dy), ti.i32
            )
            z = ti.cast(
                ti.floor(self.scene.particle_pos[p][2] / self.scene.grid_dz), ti.i32
            )
            x = ti.max(1, ti.min(nx - 2, x))
            y = ti.max(1, ti.min(ny - 2, y))
            z = ti.max(1, ti.min(nz - 2, z))

            d0 = self.scene.avg_density[None]
            if d0 > 0.0:
                rel_density = self.scene.grid_density[x, y, z] / d0
                if rel_density < 0.7:
                    color[0] = 0.6
                    color[1] = 0.6
                    color[2] = 1.0

            self.scene.particle_color[p] = color

    def handle_density(self):
        self._calc_density()
        if self.scene.avg_density[None] == 0:
            self._init_density()
        self._update_particle_color()


class NoOpDensityStrategy(DensityStrategyBase):
    def handle_density(self):
        return
