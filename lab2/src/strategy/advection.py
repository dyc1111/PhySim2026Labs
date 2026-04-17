from abc import ABC, abstractmethod
from scene import Scene


class AdvectionStrategyBase(ABC):
    @abstractmethod
    def handle_advection(self, dt):
        """Advect particle positions and velocities."""
        return NotImplementedError


class EulerIntegration(AdvectionStrategyBase):
    def __init__(self, scene: Scene):
        self.scene = scene

    def handle_advection(self, dt):
        pass
