from abc import ABC, abstractmethod
import taichi as ti
from scene import Scene


class Energy(ABC):
    def __init__(self, scene: Scene):
        self.scene = scene
        young = scene.young
        poisson = scene.poisson
        self.mu = young / (1 + poisson) / 2
        self.lmbda = young * poisson / (1 + poisson) / (1 - 2 * poisson)

    @abstractmethod
    def apply(self):
        raise NotImplementedError


@ti.data_oriented
class StVk(Energy):
    def __init__(self, scene):
        super().__init__(scene)

    def apply(self):
        pass


@ti.data_oriented
class NeoHookean(Energy):
    def __init__(self, scene):
        super().__init__(scene)

    def apply(self):
        pass


@ti.data_oriented
class Corotated(Energy):
    def __init__(self, scene):
        super().__init__(scene)

    def apply(self):
        pass


ENERGY_MODEL_REGISTER: dict[str, type[Energy]] = {
    "StVK": StVk,
    "Neo-Hookean": NeoHookean,
    "Corotated": Corotated,
}
