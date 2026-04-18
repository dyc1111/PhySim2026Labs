from abc import ABC, abstractmethod
import taichi as ti
from scene import Scene


class AdvectionStrategyBase(ABC):
    @abstractmethod
    def handle_advection(self, dt):
        """Advect particle positions and velocities."""
        return NotImplementedError


@ti.data_oriented
class EulerIntegration(AdvectionStrategyBase):
    def __init__(self, scene: Scene):
        self.scene = scene

    @ti.kernel
    def handle_advection(self, dt: float):
        for p in range(self.scene.num_particles):
            self.scene.particle_pos[p] += self.scene.particle_vel[p] * dt
            self.scene.particle_vel[p] += self.scene.gravity * dt
