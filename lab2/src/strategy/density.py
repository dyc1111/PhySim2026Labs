from abc import ABC, abstractmethod
from scene import Scene


class DensityStrategyBase(ABC):
    @abstractmethod
    def handle_density(self):
        """Update per-particle or per-cell density statistics."""
        return NotImplementedError


class DensityStrategy(DensityStrategyBase):
    def __init__(self, scene: Scene):
        self.scene = scene

    def handle_density(self):
        self.scene.update_cell_type()
        self.scene.avg_density[None] = float(self.scene.num_particles) / float(
            self.scene.num_water_grid[None]
        )
        print(f"density: {self.scene.avg_density[None]}")


class NoOpDensityStrategy(DensityStrategyBase):
    def handle_density(self):
        return
