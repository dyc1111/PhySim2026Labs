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

    @ti.kernel
    def apply(self):
        for i in range(self.scene.num_elements):
            if self.scene.element_dim == 3:
                sigma = self.scene.S_3[i]
                s1, s2, s3 = sigma[0, 0], sigma[1, 1], sigma[2, 2]
                vol = self.lmbda * (s1 + s2 + s3 - 3) / 2
                grad1 = s1 * (vol + self.mu * (s1 * s1 - 1))
                grad2 = s2 * (vol + self.mu * (s2 * s2 - 1))
                grad3 = s3 * (vol + self.mu * (s3 * s3 - 1))
                self.scene.Sgrad_3[i] = ti.Matrix(
                    [[grad1, 0.0, 0.0], [0.0, grad2, 0.0], [0.0, 0.0, grad3]]
                )
            if self.scene.element_dim == 2:
                sigma = self.scene.S_2[i]
                s1, s2 = sigma[0, 0], sigma[1, 1]
                vol = self.lmbda * (s1 + s2 - 2) / 2
                grad1 = s1 * (vol + self.mu * (s1 * s1 - 1))
                grad2 = s2 * (vol + self.mu * (s2 * s2 - 1))
                self.scene.Sgrad_2[i] = ti.Matrix([[grad1, 0.0], [0.0, grad2]])


@ti.data_oriented
class NeoHookean(Energy):
    def __init__(self, scene):
        super().__init__(scene)

    @ti.kernel
    def apply(self):
        for i in range(self.scene.num_elements):
            if self.scene.element_dim == 3:
                sigma = self.scene.S_3[i]
                s1, s2, s3 = sigma[0, 0], sigma[1, 1], sigma[2, 2]
                s1 = ti.max(sigma[0, 0], 0.1)
                s2 = ti.max(sigma[1, 1], 0.1)
                s3 = ti.max(sigma[2, 2], 0.1)
                det = s1 * s2 * s3
                grad1 = self.mu * (s1 - 1 / s1) + self.lmbda * ti.log(det) / s1
                grad2 = self.mu * (s2 - 1 / s2) + self.lmbda * ti.log(det) / s2
                grad3 = self.mu * (s3 - 1 / s3) + self.lmbda * ti.log(det) / s3
                self.scene.Sgrad_3[i] = ti.Matrix(
                    [[grad1, 0.0, 0.0], [0.0, grad2, 0.0], [0.0, 0.0, grad3]]
                )
            if self.scene.element_dim == 2:
                sigma = self.scene.S_2[i]
                s1, s2 = sigma[0, 0], sigma[1, 1]
                det = s1 * s2
                grad1 = self.mu * (s1 - 1 / s1) + self.lmbda * ti.log(det) / s1
                grad2 = self.mu * (s2 - 1 / s2) + self.lmbda * ti.log(det) / s2
                self.scene.Sgrad_2[i] = ti.Matrix([[grad1, 0.0], [0.0, grad2]])


@ti.data_oriented
class Corotated(Energy):
    def __init__(self, scene):
        super().__init__(scene)

    @ti.kernel
    def apply(self):
        for i in range(self.scene.num_elements):
            if self.scene.element_dim == 3:
                sigma = self.scene.S_3[i]
                s1, s2, s3 = sigma[0, 0], sigma[1, 1], sigma[2, 2]
                s1 = ti.max(sigma[0, 0], 0.1)
                s2 = ti.max(sigma[1, 1], 0.1)
                s3 = ti.max(sigma[2, 2], 0.1)
                det = s1 * s2 * s3
                grad1 = 2 * self.mu * (s1 - 1) + self.lmbda * det * (det - 1) / s1
                grad2 = 2 * self.mu * (s2 - 1) + self.lmbda * det * (det - 1) / s2
                grad3 = 2 * self.mu * (s3 - 1) + self.lmbda * det * (det - 1) / s3
                self.scene.Sgrad_3[i] = ti.Matrix(
                    [[grad1, 0.0, 0.0], [0.0, grad2, 0.0], [0.0, 0.0, grad3]]
                )
            if self.scene.element_dim == 2:
                sigma = self.scene.S_2[i]
                s1, s2 = sigma[0, 0], sigma[1, 1]
                det = s1 * s2
                grad1 = 2 * self.mu * (s1 - 1) + self.lmbda * det * (det - 1) / s1
                grad2 = 2 * self.mu * (s2 - 1) + self.lmbda * det * (det - 1) / s2
                self.scene.Sgrad_2[i] = ti.Matrix([[grad1, 0.0], [0.0, grad2]])


ENERGY_MODEL_REGISTER: dict[str, type[Energy]] = {
    "StVK": StVk,
    "Neo-Hookean": NeoHookean,
    "Corotated": Corotated,
}
