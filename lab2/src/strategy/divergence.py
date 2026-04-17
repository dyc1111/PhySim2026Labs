from abc import ABC, abstractmethod
from scene import Scene


class DivergenceStrategyBase(ABC):
    @abstractmethod
    def handle_divergence(self, dt):
        """Project grid velocity to a divergence-free field."""
        return NotImplementedError


class GaussSeidel(DivergenceStrategyBase):
    def __init__(self, scene: Scene, num_iters, over_relaxation, compensate_drift):
        self.scene = scene
        self.num_iters = num_iters
        self.over_relaxation = over_relaxation
        self.compensate_drift = compensate_drift

    def handle_divergence(self, dt):
        pass
