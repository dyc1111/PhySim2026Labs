from abc import ABC, abstractmethod
from scene import Scene


class CollisionStrategyBase(ABC):
    @abstractmethod
    def handle_collision(self):
        """Resolve particle collisions with boundaries and optional obstacles."""
        return NotImplementedError


class CollisionStrategy(CollisionStrategyBase):
    def __init__(self, scene: Scene):
        self.scene = scene

    def handle_collision(self):
        pass
