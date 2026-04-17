from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping
from scene import Scene


class ParticleIntegratorStrategy(ABC):
    @abstractmethod
    def integrate_particles(self, scene: Scene, dt: float) -> None:
        """Advect particle positions and velocities."""


class ParticleCollisionStrategy(ABC):
    @abstractmethod
    def handle_particle_collisions(
        self,
        scene: Scene,
        obstacle_pos: tuple[float, float, float] | None,
        obstacle_level: float,
        obstacle_vel: tuple[float, float, float] | None,
    ) -> None:
        """Resolve particle collisions with boundaries and optional obstacles."""


class ParticleSeparationStrategy(ABC):
    @abstractmethod
    def push_particles_apart(self, scene: Scene, num_iters: int) -> None:
        """Apply particle separation to improve stability."""


class VelocityTransferStrategy(ABC):
    @abstractmethod
    def transfer_velocities(
        self, scene: Scene, is_p2g: bool, flip_ratio: float
    ) -> None:
        """Perform velocity transfer between particles and grid."""


class DensityUpdateStrategy(ABC):
    @abstractmethod
    def update_particle_density(self, scene: Scene) -> None:
        """Update per-particle or per-cell density statistics."""


class IncompressibilitySolveStrategy(ABC):
    @abstractmethod
    def solve_incompressibility(
        self,
        scene: Scene,
        num_pressure_iters: int,
        dt: float,
        over_relaxation: float,
        compensate_drift: bool,
    ) -> None:
        """Project grid velocity to a divergence-free field."""


class RigidBodyCouplingStrategy(ABC):
    @abstractmethod
    def couple_rigidbodies(self, scene: Scene, dt: float) -> None:
        """Optional two-way fluid-rigidbody coupling hook."""


class _UnimplementedParticleIntegrator(ParticleIntegratorStrategy):
    def integrate_particles(self, scene: Scene, dt: float) -> None:
        raise NotImplementedError("Particle integrator strategy is not implemented")


class _UnimplementedCollision(ParticleCollisionStrategy):
    def handle_particle_collisions(
        self,
        scene: Scene,
        obstacle_pos: tuple[float, float, float] | None,
        obstacle_level: float,
        obstacle_vel: tuple[float, float, float] | None,
    ) -> None:
        raise NotImplementedError("Particle collision strategy is not implemented")


class _UnimplementedSeparation(ParticleSeparationStrategy):
    def push_particles_apart(self, scene: Scene, num_iters: int) -> None:
        raise NotImplementedError("Particle separation strategy is not implemented")


class _UnimplementedFlipPicTransfer(VelocityTransferStrategy):
    def transfer_velocities(
        self, scene: Scene, is_p2g: bool, flip_ratio: float
    ) -> None:
        raise NotImplementedError("FLIP/PIC transfer strategy is not implemented")


class _UnimplementedDensityUpdate(DensityUpdateStrategy):
    def update_particle_density(self, scene: Scene) -> None:
        raise NotImplementedError("Density update strategy is not implemented")


class _UnimplementedPressureSolver(IncompressibilitySolveStrategy):
    def solve_incompressibility(
        self,
        scene: Scene,
        num_pressure_iters: int,
        dt: float,
        over_relaxation: float,
        compensate_drift: bool,
    ) -> None:
        raise NotImplementedError(
            "Incompressibility solver strategy is not implemented"
        )


class _NoOpRigidBodyCoupling(RigidBodyCouplingStrategy):
    def couple_rigidbodies(self, scene: Scene, dt: float) -> None:
        return


@dataclass
class FluidStrategies:
    particle_integrator: ParticleIntegratorStrategy
    collision: ParticleCollisionStrategy
    separation: ParticleSeparationStrategy
    transfer: VelocityTransferStrategy
    density_update: DensityUpdateStrategy
    pressure_solver: IncompressibilitySolveStrategy
    rigidbody_coupling: RigidBodyCouplingStrategy


class SimulationBase(ABC):
    """Base class for all simulators (FLIP/PIC/APIC/Eulerian/etc.)."""

    def __init__(self, sim_cfg: Mapping[str, Any], scene: Scene):
        self.scene = scene
        self.sim_cfg = dict(sim_cfg)
        self.dt = float(self.sim_cfg.get("dt", 1.0 / 60.0))
        self.substeps = max(1, int(self.sim_cfg.get("substeps", 1)))
        self.steps = int(self.sim_cfg.get("steps", -1))

    def run(self) -> None:
        n_steps = self.steps if self.steps > 0 else 1
        for _ in range(n_steps):
            self.step()

    def step(self) -> None:
        sdt = self.dt / float(self.substeps)
        for _ in range(self.substeps):
            self._advance_substep(sdt)

    @abstractmethod
    def _advance_substep(self, sdt: float) -> None:
        """Advance one simulation substep."""


class FluidSimulationBase(SimulationBase):
    """Template method class for particle-grid fluid simulation loops."""

    def __init__(self, sim_cfg: Mapping[str, Any], scene: Scene):
        super().__init__(sim_cfg, scene)
        self.flip_ratio = float(self.sim_cfg.get("flip_ratio", 0.95))
        self.separate_particles = bool(self.sim_cfg.get("separate_particles", True))
        self.num_particle_iters = int(self.sim_cfg.get("num_particle_iters", 2))
        self.num_pressure_iters = int(self.sim_cfg.get("num_pressure_iters", 40))
        self.over_relaxation = float(self.sim_cfg.get("over_relaxation", 1.9))
        self.compensate_drift = bool(self.sim_cfg.get("compensate_drift", False))

        self.strategies = self._build_strategies()

    @abstractmethod
    def _build_strategies(self) -> FluidStrategies:
        """Create strategy set for the concrete simulator."""

    def _advance_substep(self, sdt: float) -> None:
        self.integrateParticles(sdt)
        self.handleParticleCollisions(
            obstacle_pos=None, obstacle_level=0.0, obstacle_vel=None
        )

        if self.separate_particles:
            self.pushParticlesApart(self.num_particle_iters)

        self.handleParticleCollisions(
            obstacle_pos=None, obstacle_level=0.0, obstacle_vel=None
        )
        self.transferVelocities(is_p2g=True, flip_ratio=self.flip_ratio)
        self.updateParticleDensity()
        self.solveIncompressibility(
            self.num_pressure_iters,
            sdt,
            self.over_relaxation,
            self.compensate_drift,
        )
        self.transferVelocities(is_p2g=False, flip_ratio=self.flip_ratio)
        self.coupleRigidbodies(sdt)

    # Loop stage APIs (intentionally thin wrappers around strategies)
    def integrateParticles(self, dt: float) -> None:
        self.strategies.particle_integrator.integrate_particles(self.scene, dt)

    def handleParticleCollisions(
        self,
        obstacle_pos: tuple[float, float, float] | None,
        obstacle_level: float,
        obstacle_vel: tuple[float, float, float] | None,
    ) -> None:
        self.strategies.collision.handle_particle_collisions(
            self.scene,
            obstacle_pos,
            obstacle_level,
            obstacle_vel,
        )

    def pushParticlesApart(self, num_particle_iters: int) -> None:
        self.strategies.separation.push_particles_apart(self.scene, num_particle_iters)

    def transferVelocities(self, is_p2g: bool, flip_ratio: float) -> None:
        self.strategies.transfer.transfer_velocities(self.scene, is_p2g, flip_ratio)

    def updateParticleDensity(self) -> None:
        self.strategies.density_update.update_particle_density(self.scene)

    def solveIncompressibility(
        self,
        num_pressure_iters: int,
        dt: float,
        over_relaxation: float,
        compensate_drift: bool,
    ) -> None:
        self.strategies.pressure_solver.solve_incompressibility(
            self.scene,
            num_pressure_iters,
            dt,
            over_relaxation,
            compensate_drift,
        )

    def coupleRigidbodies(self, dt: float) -> None:
        self.strategies.rigidbody_coupling.couple_rigidbodies(self.scene, dt)


class FlipPicSimulation(FluidSimulationBase):
    """PIC+FLIP hybrid simulator scaffold."""

    def _build_strategies(self) -> FluidStrategies:
        return FluidStrategies(
            particle_integrator=_UnimplementedParticleIntegrator(),
            collision=_UnimplementedCollision(),
            separation=_UnimplementedSeparation(),
            transfer=_UnimplementedFlipPicTransfer(),
            density_update=_UnimplementedDensityUpdate(),
            pressure_solver=_UnimplementedPressureSolver(),
            rigidbody_coupling=_NoOpRigidBodyCoupling(),
        )


class APICSimulation(FluidSimulationBase):
    """Reserved scaffold for APIC implementation."""

    def _build_strategies(self) -> FluidStrategies:
        raise NotImplementedError(
            "APIC simulation scaffold is declared but not implemented"
        )


class EulerianFluidSimulation(SimulationBase):
    """Reserved scaffold for grid-only (semi-Lagrangian) fluid simulation."""

    def _advance_substep(self, sdt: float) -> None:
        raise NotImplementedError(
            "Eulerian fluid simulation scaffold is declared but not implemented"
        )


def build_simulator(sim_cfg, scene: Scene):
    sim_type = str(sim_cfg.get("type", "flip_pic")).lower()
    if sim_type == "flip_pic":
        return FlipPicSimulation(sim_cfg, scene)
    if sim_type == "apic":
        return APICSimulation(sim_cfg, scene)
    if sim_type == "eulerian":
        return EulerianFluidSimulation(sim_cfg, scene)
    raise NotImplementedError(f"Unsupported simulator type: {sim_type}")
