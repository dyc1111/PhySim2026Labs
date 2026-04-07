import numpy as np
import taichi as ti
import cvxpy as cp

from scene import Scene
from collision import Collision
from interaction import InteractionHandler


class BaseSimulator:
    def __init__(self, sim_cfg, scene: Scene):
        self.scene = scene
        self.n_bodies = self.scene.num_bodies[None]
        self.interaction_handler = InteractionHandler(self.scene)
        self.dt = sim_cfg["dt"]
        self.steps = sim_cfg["steps"]
        self.substeps = sim_cfg["substeps"]
        self.collision = Collision(sim_cfg["collision"], self.scene)

    def _render(self, window, camera, canvas, scene_3d):
        scene_3d.set_camera(camera)
        scene_3d.ambient_light((0.6, 0.6, 0.6))
        scene_3d.point_light((5, 5, 5), (1.2, 1.2, 1.2))
        for i in range(self.scene.num_bodies[None]):
            scene_3d.mesh(
                self.scene.mesh_vertices,
                self.scene.mesh_indices,
                color=tuple(self.scene.mesh_colors[i]),
                index_offset=self.scene.index_offset[i],
                index_count=self.scene.index_count[i],
            )
        canvas.scene(scene_3d)
        window.get_canvas().set_background_color((0.8, 0.8, 0.85))
        window.show()

    def _step(self, applied_forces=None, applied_torques=None):
        raise NotImplementedError

    def _reset(self, camera):
        camera.position(-2, 1, -2)
        camera.lookat(0, 1, 0)
        camera.up(0, 1, 0)
        self.scene.reset()

    def run(self):
        window = ti.ui.Window("Rigid Body Simulation", (1280, 720), vsync=True)
        canvas = window.get_canvas()
        scene_3d = window.get_scene()
        camera = ti.ui.Camera()
        camera.position(-2, 1, -2)
        camera.lookat(0, 1, 0)
        camera.up(0, 1, 0)

        frame = 0
        while window.running and frame < (
            self.steps if self.steps > 0 else float("inf")
        ):
            if window.is_pressed(ti.ui.ESCAPE):
                break
            elif window.is_pressed(ti.ui.SPACE):
                self._reset(camera)

            if not window.is_pressed(ti.ui.CTRL):
                camera.track_user_inputs(
                    window, movement_speed=0.03, hold_key=ti.ui.LMB
                )

            applied_forces, applied_torques = self.interaction_handler.process_inputs(
                window, camera
            )

            self._step(applied_forces, applied_torques)
            self._render(window, camera, canvas, scene_3d)
            frame += 1


class ImpulseSimulator(BaseSimulator):
    def __init__(self, sim_cfg, scene: Scene):
        super().__init__(sim_cfg, scene)

    def _step(self, applied_forces=None, applied_torques=None):
        dt = self.dt / self.substeps

        if applied_forces is None:
            applied_forces = np.zeros((self.n_bodies, 3), dtype=np.float32)
        if applied_torques is None:
            applied_torques = np.zeros((self.n_bodies, 3), dtype=np.float32)

        for _ in range(self.substeps):
            self.scene.pre_solve_kinematics(dt, applied_forces, applied_torques)
            self.collision.detect_and_resolve(dt)
            self.scene.post_solve_kinematics(dt)

        self.scene.update_mesh_vertices()


class ConstraintSimulator(BaseSimulator):
    def __init__(self, sim_cfg, scene: Scene):
        super().__init__(sim_cfg, scene)

        self.mu = float(sim_cfg["collision"]["mu"])
        self.restitution = float(sim_cfg["collision"]["restitution"])

        self.constraint_cfg = sim_cfg.get("constraint", {})
        self.baumgarte_beta = float(self.constraint_cfg.get("baumgarte_beta", 0.2))
        self.penetration_slop = float(
            self.constraint_cfg.get("penetration_slop", 0.005)
        )

    def _solve_ccp(self, G, g, n_contacts):
        dim = 3 * n_contacts

        # Convert G to a symmetric matrix
        G = 0.5 * (G + G.T)
        # Add a regularizer to make the CCP strongly convex.
        G_reg = G + 1e-6 * np.eye(dim, dtype=np.float32)

        lam = cp.Variable(dim)
        objective = (
            0.5 * cp.quad_form(lam, G_reg.astype(np.float64))
            + g.astype(np.float64) @ lam
        )

        constraints = []
        for i in range(n_contacts):
            lam_n = lam[3 * i]
            lam_t = lam[3 * i + 1 : 3 * i + 3]
            constraints.append(lam_n >= 0.0)
            constraints.append(cp.SOC(self.mu * lam_n, lam_t))

        prob = cp.Problem(cp.Minimize(objective), constraints)

        solver_name = str(self.constraint_cfg.get("solver", "SCS")).upper()
        eps = float(self.constraint_cfg.get("eps", 1e-4))
        max_iters = int(self.constraint_cfg.get("max_iters", 2500))

        solve_kwargs = {"warm_start": True, "verbose": False}
        if solver_name == "ECOS":
            solve_kwargs.update(
                {
                    "solver": cp.ECOS,
                    "abstol": eps,
                    "reltol": eps,
                    "feastol": eps,
                    "max_iters": max_iters,
                }
            )
        else:
            solve_kwargs.update({"solver": cp.SCS, "eps": eps, "max_iters": max_iters})

        try:
            prob.solve(**solve_kwargs)
        except Exception:
            return np.zeros(dim, dtype=np.float32)

        if lam.value is None:
            return np.zeros(dim, dtype=np.float32)

        lam_np = np.asarray(lam.value, dtype=np.float32).reshape(-1)
        lam_np = np.where(np.isfinite(lam_np), lam_np, 0.0)
        return lam_np

    def _step(self, applied_forces=None, applied_torques=None):
        dt = self.dt / self.substeps

        if applied_forces is None:
            applied_forces = np.zeros((self.n_bodies, 3), dtype=np.float32)
        if applied_torques is None:
            applied_torques = np.zeros((self.n_bodies, 3), dtype=np.float32)

        for _ in range(self.substeps):
            # 1) Free dynamics: external forces and torques only.
            self.scene.pre_solve_kinematics(dt, applied_forces, applied_torques)

            # 2) Contact detection and CCP solve.
            contacts = self.collision.detect_contacts(
                max_keep=self.collision.max_contact
            )
            if contacts:
                J = self.scene.calc_jacobian(contacts)
                M_inv = self.scene.calc_mass_inverse_matrix()
                v_free = self.scene.get_generalized_velocity()

                G = J @ M_inv @ J.T
                g = J @ v_free

                for i, c in enumerate(contacts):
                    depth = float(c[4])
                    v_rel_n = g[3 * i]
                    v_target_n = v_rel_n + self.restitution * min(v_rel_n, 0.0)
                    bias = (self.baumgarte_beta / dt) * max(
                        depth - self.penetration_slop, 0.0
                    )
                    g[3 * i] = v_target_n - bias

                lam = self._solve_ccp(G, g, len(contacts))
                delta_v = M_inv @ (J.T @ lam)
                v_next = v_free + delta_v
                self.scene.set_generalized_velocity(v_next)

            # 3) Integrate positions and rotations from resolved velocities.
            self.scene.post_solve_kinematics(dt)

        self.scene.update_mesh_vertices()
