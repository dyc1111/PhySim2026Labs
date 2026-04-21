from abc import ABC, abstractmethod
import taichi as ti
from scene import Scene


class SeparationStrategyBase(ABC):
    @abstractmethod
    def handle_separation(self):
        """Apply particle separation to improve stability."""
        return NotImplementedError


@ti.data_oriented
class SeparationStrategy(SeparationStrategyBase):
    def __init__(self, scene: Scene, num_iters):
        self.scene = scene
        self.num_iters = num_iters
        self.delta = ti.Vector.field(3, dtype=ti.f32, shape=self.scene.num_particles)

    @ti.kernel
    def _compute_delta(self):
        radius = self.scene.particle_radius
        eps = 1e-8

        for p in range(self.scene.num_particles):
            posp = self.scene.particle_pos[p]
            i0, i1, j0, j1, k0, k1 = self.scene.hashtable.neighbouring_cell_bounds(posp)

            delta_p = ti.Vector([0.0, 0.0, 0.0])
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    for k in range(k0, k1 + 1):
                        h = self.scene.hashtable.flatten(i, j, k)
                        begin = self.scene.hashtable.cell_begin(h)
                        end = self.scene.hashtable.cell_end(h)
                        for idx in range(begin, end):
                            q = self.scene.hashtable.particle_id[idx]
                            if q == p:
                                continue
                            d = posp - self.scene.particle_pos[q]
                            dist = ti.sqrt(d.dot(d))
                            if dist >= 2.0 * radius:
                                continue
                            delta_p += (radius - dist / 2) * d / (dist + eps)
            self.delta[p] = delta_p

    @ti.kernel
    def _apply_delta(self):
        for p in range(self.scene.num_particles):
            self.scene.particle_pos[p] += self.delta[p]

    def handle_separation(self):
        for _ in range(self.num_iters):
            self.scene.hashtable.rebuild(
                self.scene.particle_pos, self.scene.num_particles
            )
            self._compute_delta()
            self._apply_delta()


class NoOpSeparationStrategy(SeparationStrategyBase):
    def handle_separation(self):
        return
