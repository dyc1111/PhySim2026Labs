from abc import ABC, abstractmethod
from dataclasses import dataclass
import taichi as ti
import numpy as np
from interaction import InteractionHandler
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
        self.interaction_handler = InteractionHandler(scene)
        self.dt = float(sim_cfg["dt"])
        self.dt_min = 1e-4
        self.dt_max = 5e-2
        self.substeps = int(sim_cfg["substeps"])
        self.steps = int(sim_cfg["steps"])

        self.video = sim_cfg["video"]
        self.video_manager = ti.tools.VideoManager(
            output_dir="./", framerate=30, automatic_build=False
        )
        self.video_manager.clean_frames()
        self.paused = False
        self._space_was_down = False
        self._f_was_down = False
        self._init_renderer()

        self.strategies = self._build_strategies()

    def _init_renderer(self):
        self.window = ti.ui.Window("Fluid Simulation", (1280, 720), vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene_3d = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self._init_camera()

    def _init_camera(self):
        sx, sy, sz = self.scene.grid_size
        center = (0.5 * sx, 0.5 * sy, 0.5 * sz)
        span = max(sx, sy, sz)
        self.camera.position(-0.9 * span, 0.9 * span, -0.9 * span)
        self.camera.lookat(center[0], center[1], center[2])
        self.camera.up(0.0, 1.0, 0.0)

    def _draw_gui(self):
        gui = self.window.get_gui()
        with gui.sub_window("timestep", 0.02, 0.02, 0.28, 0.20):
            self.dt = gui.slider_float("dt", self.dt, self.dt_min, self.dt_max)
            gui.text(f"dt = {self.dt:.6f}")

    def _handle_inputs(self):
        if not self.window.running or self.window.is_pressed(ti.ui.ESCAPE):
            return False

        self._draw_gui()

        space_down = self.window.is_pressed(ti.ui.SPACE)
        if space_down and not self._space_was_down:
            self.paused = not self.paused
        self._space_was_down = space_down

        f_down = self.window.is_pressed("f") or self.window.is_pressed("F")
        if f_down and not self._f_was_down:
            self._init_camera()
            self.scene.reset()
        self._f_was_down = f_down

        return True

    def _render(self):
        if not self.window.running:
            return False

        if not self.window.is_pressed(ti.ui.CTRL):
            self.camera.track_user_inputs(
                self.window, movement_speed=0.03, hold_key=ti.ui.RMB
            )

        self.scene_3d.set_camera(self.camera)
        self.scene_3d.ambient_light((0.6, 0.6, 0.6))
        self.scene_3d.point_light((5, 5, 5), (1.2, 1.2, 1.2))

        if self.scene.num_rigidbodies > 0:
            self.scene.update_rigidbody_mesh_vertices()
            for i in range(self.scene.num_rigidbodies):
                self.scene_3d.mesh(
                    self.scene.mesh_vertices,
                    self.scene.mesh_indices,
                    color=tuple(self.scene.mesh_colors[i]),
                    index_offset=int(self.scene.index_offset[i]),
                    index_count=int(self.scene.index_count[i]),
                )

        self.scene_3d.particles(
            self.scene.particle_pos,
            radius=self.scene.particle_radius * 0.6,
            per_vertex_color=self.scene.particle_color,
        )
        self.scene_3d.lines(
            self.scene.bbox_vertices,
            width=2.0,
            indices=self.scene.bbox_indices,
            color=(0.05, 0.05, 0.05),
        )

        self.canvas.scene(self.scene_3d)
        self.canvas.set_background_color((0.8, 0.8, 0.85))
        self.window.show()

        if self.video:
            pixels_img = self.window.get_image_buffer_as_numpy()
            self.video_manager.write_frame(pixels_img)

        return True

    def run(self):
        n_steps = self.steps if self.steps >= 0 else float("inf")
        step = 0
        try:
            while step < n_steps:
                if not self._handle_inputs():
                    break

                sdt = self.dt / float(self.substeps)
                applied_forces, applied_torques = (
                    self.interaction_handler.process_inputs(self.window, self.camera)
                )

                if not self.paused:
                    for _ in range(self.substeps):
                        self._step(sdt, applied_forces, applied_torques)
                    step += 1

                if not self._render():
                    break
        finally:
            if self.video:
                self.video_manager.make_video(gif=False, mp4=True)
            self.close()

    def _step(self, sdt, applied_forces, applied_torques):
        if self.scene.num_rigidbodies > 0:
            self.scene.pre_solve_kinematics(sdt, applied_forces, applied_torques)
            self.scene.post_solve_kinematics(sdt)
        self.handle_advection(sdt)
        self.handle_collision()
        self.handle_separation()
        self.handle_collision()
        self.handle_transfer(sdt, True)
        self.handle_density()
        self.handle_divergence(sdt)
        self.handle_transfer(sdt, False)

    @abstractmethod
    def _build_strategies(self) -> FluidStrategies:
        return NotImplementedError

    def handle_advection(self, dt):
        self.strategies.advection.handle_advection(dt)

    def handle_collision(self):
        self.strategies.collision.handle_collision()

    def handle_separation(self):
        self.strategies.separation.handle_separation()

    def handle_transfer(self, sdt, is_p2g):
        self.strategies.transfer.handle_transfer(sdt, is_p2g)

    def handle_density(self):
        self.strategies.density.handle_density()

    def handle_divergence(self, dt):
        self.strategies.divergence.handle_divergence(dt)

    def close(self):
        self.strategies.divergence.destroy()


class FlipPicSimulator(Simulator):
    def __init__(self, sim_cfg, scene: Scene):
        self.flip_ratio = float(sim_cfg["flip_ratio"])
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
            advection=GravityIntegration(self.scene),
            collision=CollisionStrategy(self.scene),
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
    """Grid-based semi-Lagrangian simulator (Eulerian fluid)."""

    def __init__(self, sim_cfg, scene: Scene):
        self.extrapolation_iters = sim_cfg["extrapolation_iters"]
        self.num_pressure_iters = sim_cfg["pressure_max_iters"]
        self.pressure_tolerance = sim_cfg["pressure_tolerance"]
        self.rho = sim_cfg["rho"]
        super().__init__(sim_cfg, scene)

    def _build_strategies(self) -> FluidStrategies:
        return FluidStrategies(
            advection=SemiLagrangian(self.scene, self.extrapolation_iters),
            collision=CollisionStrategy(self.scene),
            separation=NoOpSeparationStrategy(),
            transfer=EulerianTransferStrategy(self.scene),
            density=DensityStrategy(self.scene),
            divergence=LinearSystem(
                self.scene, self.num_pressure_iters, self.pressure_tolerance, self.rho
            ),
        )


def build_simulator(sim_cfg, scene: Scene):
    sim_type = sim_cfg["type"]
    if sim_type == "flip_pic":
        return FlipPicSimulator(sim_cfg, scene)
    if sim_type == "apic":
        return APICSimulator(sim_cfg, scene)
    if sim_type == "eulerian":
        return EulerianFluidSimulator(sim_cfg, scene)
    raise NotImplementedError(f"Unsupported simulator type: {sim_type}")
