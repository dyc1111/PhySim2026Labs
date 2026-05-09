from abc import ABC, abstractmethod
import taichi as ti
from interaction import InteractionHandler
from scene import Scene
from energy import ENERGY_MODEL_REGISTER


class Simulator(ABC):
    def __init__(self, sim_cfg, scene: Scene):
        self.scene = scene
        self.dt = sim_cfg["dt"]
        self.substeps = sim_cfg["substeps"]
        self.steps = sim_cfg["steps"]

        model = sim_cfg["model"]
        if model not in ENERGY_MODEL_REGISTER.keys():
            raise NotImplementedError(f"Unsupported energy model: {model}")
        self.model = ENERGY_MODEL_REGISTER[model](scene)

        self.video = sim_cfg["video"]
        self.paused = False
        self._space_was_down = False
        self._f_was_down = False
        self.interaction_handler = InteractionHandler(scene)
        self.video_manager = ti.tools.VideoManager(
            output_dir="./", framerate=30, automatic_build=False
        )
        if self.video:
            self.video_manager.clean_frames()
        self._init_renderer()

    def _init_renderer(self):
        self.window = ti.ui.Window("Softbody Simulation", (1280, 720), vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene_3d = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self._init_camera()

    def _init_camera(self):
        rest = self.scene.rest_pos.to_numpy()
        center = rest.mean(axis=0)
        span = max(float((rest.max(axis=0) - rest.min(axis=0)).max()), 1.0)
        self.camera.position(
            center[0] - 0.9 * span, center[1] + 0.55 * span, center[2] - 1.2 * span
        )
        self.camera.lookat(center[0], center[1], center[2])
        self.camera.up(0.0, 1.0, 0.0)

    def _handle_inputs(self):
        if not self.window.running or self.window.is_pressed(ti.ui.ESCAPE):
            return False, None

        space_down = self.window.is_pressed(ti.ui.SPACE)
        if space_down and not self._space_was_down:
            self.paused = not self.paused
        self._space_was_down = space_down

        f_down = self.window.is_pressed("f") or self.window.is_pressed("F")
        if f_down and not self._f_was_down:
            self.scene.reset()
            self._init_camera()
        self._f_was_down = f_down

        if not self.window.is_pressed(ti.ui.CTRL):
            self.camera.track_user_inputs(
                self.window, movement_speed=0.03, hold_key=ti.ui.RMB
            )

        return True, self.interaction_handler.process_inputs(self.window, self.camera)

    def _render(self):
        self.scene_3d.set_camera(self.camera)
        self.scene_3d.ambient_light((0.6, 0.6, 0.6))
        self.scene_3d.point_light((5, 5, 5), (1.2, 1.2, 1.2))
        self.scene_3d.mesh(
            self.scene.pos,
            self.scene.surface_indices,
            color=self.scene.color,
            two_sided=True,
        )
        self.canvas.scene(self.scene_3d)
        self.canvas.set_background_color((0.8, 0.8, 0.85))
        self.window.show()

        if self.video:
            self.video_manager.write_frame(self.window.get_image_buffer_as_numpy())

    def run(self):
        step = 0
        max_steps = self.steps if self.steps >= 0 else float("inf")
        try:
            while step < max_steps:
                running, applied_forces = self._handle_inputs()
                if not running:
                    break
                if not self.paused:
                    sub_dt = self.dt / float(self.substeps)
                    for _ in range(self.substeps):
                        self._step(sub_dt, applied_forces)
                    step += 1
                self._render()
        finally:
            if self.video:
                self.video_manager.make_video(gif=True, mp4=False)

    @abstractmethod
    def _step(self, dt, applied_forces):
        raise NotImplementedError


class ExplicitEulerSimulator(Simulator):
    def _step(self, dt, applied_forces):
        self.scene.set_external_forces(applied_forces)
        self.scene.calc_grad()
        self.scene.svd()
        self.model.apply()
        self.scene.calc_PK_stress()
        self.scene.calc_internal_forces()
        self.scene.time_integral(dt)


def build_simulator(sim_cfg, scene: Scene):
    sim_type = sim_cfg.get("type", "explicit")
    if sim_type == "explicit":
        return ExplicitEulerSimulator(sim_cfg, scene)
    raise NotImplementedError(f"Unsupported simulator type: {sim_type}")
