from abc import ABC, abstractmethod
from scene import Scene
import taichi as ti
from constants import CellType


class CollisionStrategyBase(ABC):
    @abstractmethod
    def handle_collision(self):
        """Resolve particle collisions with boundaries and optional obstacles."""
        return NotImplementedError


@ti.data_oriented
class CollisionStrategy(CollisionStrategyBase):
    def __init__(self, scene: Scene):
        self.scene = scene

    @ti.kernel
    def _handle_collision_particles(self):
        dx = self.scene.grid_dx
        dy = self.scene.grid_dy
        dz = self.scene.grid_dz
        x, y, z = self.scene.grid_size

        for p in range(self.scene.num_particles):
            pos = self.scene.particle_pos[p]
            vel = self.scene.particle_vel[p]
            radius = self.scene.particle_radius

            if pos[0] < radius + dx:
                pos[0] = radius + dx
                vel[0] = 0
            if pos[0] > x - dx - radius:
                pos[0] = x - dx - radius
                vel[0] = 0
            if pos[1] < radius + dy:
                pos[1] = radius + dy
                vel[1] = 0
            if pos[1] > y - dy - radius:
                pos[1] = y - dy - radius
                vel[1] = 0
            if pos[2] < radius + dz:
                pos[2] = radius + dz
                vel[2] = 0
            if pos[2] > z - dz - radius:
                pos[2] = z - dz - radius
                vel[2] = 0

            cx = ti.cast(ti.floor(pos[0] / dx), ti.i32)
            cy = ti.cast(ti.floor(pos[1] / dy), ti.i32)
            cz = ti.cast(ti.floor(pos[2] / dz), ti.i32)
            if self.scene.grid_cell_type[cx, cy, cz] == CellType.CELL_SOLID.value:
                vel = self.scene.grid_solid_velocity[cx, cy, cz]

            self.scene.particle_pos[p] = pos
            self.scene.particle_vel[p] = vel

    def handle_collision(self):
        self.scene.update_cell_type()
        self._handle_collision_particles()


class NoOpCollisionStrategy(CollisionStrategyBase):
    def handle_collision(self):
        return
