from abc import ABC, abstractmethod
import taichi as ti
from scene import Scene
from constants import CellType


class DivergenceStrategyBase(ABC):
    @abstractmethod
    def handle_divergence(self, dt):
        """Project grid velocity to a divergence-free field."""
        return NotImplementedError


@ti.data_oriented
class GaussSeidel(DivergenceStrategyBase):
    def __init__(self, scene: Scene, num_iters, over_relaxation, compensate_drift):
        self.scene = scene
        self.num_iters = num_iters
        self.over_relaxation = over_relaxation
        self.compensate_drift = compensate_drift

    @ti.kernel
    def _gauss_seidel_mod2(self, color: ti.i32):  # type: ignore
        nx, ny, nz = self.scene.grid_resolution
        for I in ti.grouped(self.scene.grid_cell_type):
            x, y, z = I
            if (x + y + z) & 1 != color:
                continue
            if self.scene.grid_cell_type[I] != CellType.CELL_WATER.value:
                continue

            divergence = 0.0
            xp = x < nx - 1
            xm = x > 0
            yp = y < ny - 1
            ym = y > 0
            zp = z < nz - 1
            zm = z > 0

            if xp:
                if self.scene.grid_cell_type[x + 1, y, z] == CellType.CELL_SOLID.value:
                    xp = False
                divergence += self.scene.grid_u[x + 1, y, z]
            if xm:
                if self.scene.grid_cell_type[x - 1, y, z] == CellType.CELL_SOLID.value:
                    xm = False
                divergence -= self.scene.grid_u[x, y, z]
            if yp:
                if self.scene.grid_cell_type[x, y + 1, z] == CellType.CELL_SOLID.value:
                    yp = False
                divergence += self.scene.grid_v[x, y + 1, z]
            if ym:
                if self.scene.grid_cell_type[x, y - 1, z] == CellType.CELL_SOLID.value:
                    ym = False
                divergence -= self.scene.grid_v[x, y, z]
            if zp:
                if self.scene.grid_cell_type[x, y, z + 1] == CellType.CELL_SOLID.value:
                    zp = False
                divergence += self.scene.grid_w[x, y, z + 1]
            if zm:
                if self.scene.grid_cell_type[x, y, z - 1] == CellType.CELL_SOLID.value:
                    zm = False
                divergence -= self.scene.grid_w[x, y, z]

            num_cells = (
                ti.cast(xp, ti.i32)
                + ti.cast(xm, ti.i32)
                + ti.cast(yp, ti.i32)
                + ti.cast(ym, ti.i32)
                + ti.cast(zp, ti.i32)
                + ti.cast(zm, ti.i32)
            )
            if num_cells == 0:
                continue
            divergence = self.over_relaxation * divergence
            if self.compensate_drift:
                if self.scene.grid_density[I] > self.scene.avg_density[None]:
                    divergence -= (
                        self.scene.grid_density[I] - self.scene.avg_density[None]
                    )

            delta = divergence / num_cells
            if xp:
                self.scene.grid_u[x + 1, y, z] -= delta
            if xm:
                self.scene.grid_u[x, y, z] += delta
            if yp:
                self.scene.grid_v[x, y + 1, z] -= delta
            if ym:
                self.scene.grid_v[x, y, z] += delta
            if zp:
                self.scene.grid_w[x, y, z + 1] -= delta
            if zm:
                self.scene.grid_w[x, y, z] += delta

    def handle_divergence(self, dt):
        for _ in range(self.num_iters):
            self._gauss_seidel_mod2(0)
            self._gauss_seidel_mod2(1)
