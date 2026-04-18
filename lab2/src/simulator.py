from abc import ABC, abstractmethod
from dataclasses import dataclass
from scene import Scene
from strategy import *


@dataclass
class FluidStrategies:
    advection: AdvectionStrategyBase
    collision: CollisionStrategyBase
    separation: SeparationStrategyBase
    transfer: TransferStrategyBase
    density: DensityStrategyBase
    divergence: DivergenceStrategyBase


class Simulator(ABC):
    """Base class for all simulators (FLIP/PIC/APIC/Eulerian/etc.)."""

    def __init__(self, sim_cfg, scene: Scene):
        self.scene = scene
        self.dt = float(sim_cfg["dt"])
        self.substeps = int(sim_cfg["substeps"])
        self.steps = int(sim_cfg["steps"])
        self.strategies = self._build_strategies()

    def run(self):
        n_steps = self.steps if self.steps >= 0 else float("inf")
        sdt = self.dt / float(self.substeps)
        step = 0
        while step < n_steps:
            for _ in range(self.substeps):
                self._step(sdt)
            step += 1
            if not self.scene.render():
                break

    def _step(self, sdt: float) -> None:
        self.handle_advection(sdt)
        self.handle_collision()
        self.handle_separation()
        self.handle_collision()
        self.handle_transfer(True)
        self.handle_density()
        self.handle_divergence(sdt)
        self.handle_transfer(False)

    @abstractmethod
    def _build_strategies(self) -> FluidStrategies:
        return NotImplementedError

    def handle_advection(self, dt):
        self.strategies.advection.handle_advection(dt)

    def handle_collision(self):
        self.strategies.collision.handle_collision()

    def handle_separation(self):
        self.strategies.separation.handle_separation()

    def handle_transfer(self, is_p2g):
        self.strategies.transfer.handle_transfer(is_p2g)

    def handle_density(self):
        self.strategies.density.handle_density()

    def handle_divergence(self, dt):
        self.strategies.divergence.handle_divergence(dt)


class FlipPicSimulator(Simulator):
    def __init__(self, sim_cfg, scene: Scene):
        self.flip_ratio = float(sim_cfg["flip_ratio"])
        self.collision_damping = float(sim_cfg["collision_damping"])
        self.separate_particles = bool(sim_cfg["separate_particles"])
        self.num_particle_iters = int(sim_cfg["num_particle_iters"])
        self.num_pressure_iters = int(sim_cfg["num_pressure_iters"])
        self.over_relaxation = float(sim_cfg["over_relaxation"])
        self.compensate_drift = bool(sim_cfg["compensate_drift"])
        super().__init__(sim_cfg, scene)

    def _build_strategies(self) -> FluidStrategies:
        if self.separate_particles:
            separate = SeparationStrategy(self.scene, self.num_particle_iters)
        else:
            separate = NoOpSeparationStrategy()
        return FluidStrategies(
            advection=EulerIntegration(self.scene),
            collision=CollisionStrategy(self.scene, self.collision_damping),
            separation=separate,
            transfer=FlipTransferStrategy(self.scene, self.flip_ratio),
            density=DensityStrategy(self.scene),
            divergence=GaussSeidel(
                self.scene,
                self.num_pressure_iters,
                self.over_relaxation,
                self.compensate_drift,
            ),
        )


class APICSimulator(Simulator):
    """Reserved scaffold for APIC implementation."""

    def _build_strategies(self) -> FluidStrategies:
        raise NotImplementedError(
            "APIC simulation scaffold is declared but not implemented"
        )


class EulerianFluidSimulator(Simulator):
    """Reserved scaffold for grid-only (semi-Lagrangian) fluid simulation."""

    def _advance_substep(self, sdt: float) -> None:
        raise NotImplementedError(
            "Eulerian fluid simulation scaffold is declared but not implemented"
        )


def build_simulator(sim_cfg, scene: Scene):
    sim_type = str(sim_cfg.get("type", "flip_pic")).lower()
    if sim_type == "flip_pic":
        return FlipPicSimulator(sim_cfg, scene)
    if sim_type == "apic":
        return APICSimulator(sim_cfg, scene)
    if sim_type == "eulerian":
        return EulerianFluidSimulator(sim_cfg, scene)
    raise NotImplementedError(f"Unsupported simulator type: {sim_type}")
