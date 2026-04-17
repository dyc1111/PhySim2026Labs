from abc import ABC, abstractmethod
from scene import Scene


class SeparationStrategyBase(ABC):
    @abstractmethod
    def handle_separation(self):
        """Apply particle separation to improve stability."""
        return NotImplementedError


class SeparationStrategy(SeparationStrategyBase):
    def __init__(self, scene: Scene, num_iters):
        self.scene = scene
        self.num_iters = num_iters

    def handle_separation(self):
        pass
