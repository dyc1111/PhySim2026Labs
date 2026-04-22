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
    def __init__(self, scene: Scene, collision_damping):
        self.scene = scene
        self.collision_damping = collision_damping

    @ti.func
    def _update_vel(self, vel, solid_vel, normal):
        rel_vel = vel - solid_vel
        rel_n = rel_vel.dot(normal)
        if rel_n < 0.0:
            rel_t = rel_vel - rel_n * normal
            vel = solid_vel + (1.0 - self.collision_damping) * rel_t
        return vel

    @ti.func
    def _closest_point_on_aabb(self, p, bmin, bmax):
        return ti.Vector(
            [
                ti.max(bmin[0], ti.min(bmax[0], p[0])),
                ti.max(bmin[1], ti.min(bmax[1], p[1])),
                ti.max(bmin[2], ti.min(bmax[2], p[2])),
            ]
        )

    @ti.kernel
    def handle_collision(self):
        nx, ny, nz = self.scene.grid_resolution
        dx = self.scene.grid_dx
        dy = self.scene.grid_dy
        dz = self.scene.grid_dz
        x, y, z = self.scene.grid_size
        eps = 1e-8

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
            i0 = ti.max(0, cx - 1)
            i1 = ti.min(nx - 1, cx + 1)
            j0 = ti.max(0, cy - 1)
            j1 = ti.min(ny - 1, cy + 1)
            k0 = ti.max(0, cz - 1)
            k1 = ti.min(nz - 1, cz + 1)

            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    for k in range(k0, k1 + 1):
                        if (
                            self.scene.grid_cell_type[i, j, k]
                            != CellType.CELL_SOLID.value
                        ):
                            continue
                        if i == 0 or i == nx - 1:
                            continue
                        if j == 0 or j == ny - 1:
                            continue
                        if k == 0 or k == nz - 1:
                            continue

                        bmin = ti.Vector([i * dx, j * dy, k * dz])
                        bmax = ti.Vector([(i + 1) * dx, (j + 1) * dy, (k + 1) * dz])
                        q = self._closest_point_on_aabb(pos, bmin, bmax)
                        d = pos - q
                        dist2 = d.dot(d)

                        if dist2 >= radius * radius:
                            continue

                        solid_vel = self.scene.grid_solid_velocity[i, j, k]
                        if dist2 > eps:
                            dist = ti.sqrt(dist2)
                            n = d / dist
                            pos += n * (radius - dist)
                            vel = self._update_vel(vel, solid_vel, n)
                        else:
                            dl = pos[0] - bmin[0]
                            dr = bmax[0] - pos[0]
                            db = pos[1] - bmin[1]
                            dt = bmax[1] - pos[1]
                            dk = pos[2] - bmin[2]
                            df = bmax[2] - pos[2]

                            min_face = dl
                            n = ti.Vector([-1.0, 0.0, 0.0])
                            pen = dl + radius

                            if dr < min_face:
                                min_face = dr
                                n = ti.Vector([1.0, 0.0, 0.0])
                                pen = dr + radius
                            if db < min_face:
                                min_face = db
                                n = ti.Vector([0.0, -1.0, 0.0])
                                pen = db + radius
                            if dt < min_face:
                                min_face = dt
                                n = ti.Vector([0.0, 1.0, 0.0])
                                pen = dt + radius
                            if dk < min_face:
                                min_face = dk
                                n = ti.Vector([0.0, 0.0, -1.0])
                                pen = dk + radius
                            if df < min_face:
                                n = ti.Vector([0.0, 0.0, 1.0])
                                pen = df + radius

                            pos += n * pen
                            vel = self._update_vel(vel, solid_vel, n)

            self.scene.particle_pos[p] = pos
            self.scene.particle_vel[p] = vel


class NoOpCollisionStrategy(CollisionStrategyBase):
    def handle_collision(self):
        return
