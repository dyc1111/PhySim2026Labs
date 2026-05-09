import numpy as np
import taichi as ti
from mesh import build_mesh


@ti.data_oriented
class Scene:
    def __init__(self, scene_cfg):
        self.scene_cfg = scene_cfg
        mesh = build_mesh(scene_cfg)
        self.gravity = ti.Vector(scene_cfg["gravity"])
        self.young = scene_cfg["young"]
        self.poisson = scene_cfg["poisson"]

        self.num_vertices = mesh.vertices.shape[0]
        self.num_elements = mesh.elements.shape[0]
        self.num_surface_faces = mesh.surface_faces.shape[0]
        self.element_dim = mesh.element_dim
        self.color = tuple(scene_cfg["color"])

        self.rest_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
        self.vel = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
        self.mass = ti.field(dtype=ti.f32, shape=self.num_vertices)
        self.inv_mass = ti.field(dtype=ti.f32, shape=self.num_vertices)
        self.pinned = ti.field(dtype=ti.i32, shape=self.num_vertices)

        self.elements = ti.Vector.field(4, dtype=ti.i32, shape=self.num_elements)
        self.rest_inv_3 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.num_elements)
        self.rest_inv_2 = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.num_elements)
        self.rest_measure = ti.field(dtype=ti.f32, shape=self.num_elements)

        self.surface_indices = ti.field(dtype=ti.i32, shape=3 * self.num_surface_faces)

        self.grad_3 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.num_elements)
        self.grad_2 = ti.Matrix.field(3, 2, dtype=ti.f32, shape=self.num_elements)
        self.U_3 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.num_elements)
        self.U_2 = ti.Matrix.field(3, 2, dtype=ti.f32, shape=self.num_elements)
        self.V_3 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.num_elements)
        self.V_2 = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.num_elements)
        self.S_3 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.num_elements)
        self.S_2 = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.num_elements)
        self.Sgrad_3 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.num_elements)
        self.Sgrad_2 = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.num_elements)
        self.PK_3 = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.num_elements)
        self.PK_2 = ti.Matrix.field(3, 2, dtype=ti.f32, shape=self.num_elements)

        self._load_mesh(mesh)
        self.reset()

    def _load_mesh(self, mesh):
        self.rest_pos.from_numpy(mesh.vertices)
        self.pos.from_numpy(mesh.vertices)
        self.mass.from_numpy(mesh.masses)
        inv_mass = 1.0 / mesh.masses
        self.inv_mass.from_numpy(inv_mass)
        self.elements.from_numpy(mesh.elements)
        self.rest_inv_3.from_numpy(mesh.rest_inv_3)
        self.rest_inv_2.from_numpy(mesh.rest_inv_2)
        self.rest_measure.from_numpy(mesh.rest_measure)
        self.surface_indices.from_numpy(mesh.surface_faces.reshape(-1))

        pinned = np.zeros((self.num_vertices,), dtype=np.int32)
        pin_rule = self.scene_cfg["pin"]
        if pin_rule == "left":
            rest = mesh.vertices
            xmin = np.min(rest[:, 0])
            pinned[np.abs(rest[:, 0] - xmin) <= 1e-8] = 1
        elif pin_rule == "corners" and self.element_dim == 2:
            rest = mesh.vertices
            xmin, xmax = float(np.min(rest[:, 0])), float(np.max(rest[:, 0]))
            zmin, zmax = float(np.min(rest[:, 2])), float(np.max(rest[:, 2]))
            pinned[
                (
                    (np.abs(rest[:, 2] - zmin) <= 1e-8)
                    | (np.abs(rest[:, 2] - zmax) <= 1e-8)
                )
                & (
                    (np.abs(rest[:, 0] - xmin) <= 1e-8)
                    | (np.abs(rest[:, 0] - xmax) <= 1e-8)
                )
            ] = 1
        elif pin_rule == "left_corners" and self.element_dim == 2:
            rest = mesh.vertices
            xmin = float(np.min(rest[:, 0]))
            zmin, zmax = float(np.min(rest[:, 2])), float(np.max(rest[:, 2]))
            pinned[
                (
                    (np.abs(rest[:, 2] - zmin) <= 1e-8)
                    | (np.abs(rest[:, 2] - zmax) <= 1e-8)
                )
                & (np.abs(rest[:, 0] - xmin) <= 1e-8)
            ] = 1

        self.pinned.from_numpy(pinned)

    @ti.kernel
    def reset(self):
        for i in range(self.num_vertices):
            self.pos[i] = self.rest_pos[i]
            self.vel[i] = ti.Vector([0.0, 0.0, 0.0])
            self.force[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def set_external_forces(self, applied_forces: ti.types.ndarray()):  # type: ignore
        for i in range(self.num_vertices):
            self.force[i] = ti.Vector([0.0, 0.0, 0.0])
            if self.pinned[i] == 0:
                self.force[i] += self.mass[i] * self.gravity
                self.force[i] += ti.Vector(
                    [applied_forces[i, 0], applied_forces[i, 1], applied_forces[i, 2]]
                )

    @ti.kernel
    def calc_grad(self):
        for i in range(self.num_elements):
            element = self.elements[i]
            if self.element_dim == 3:
                x0 = self.pos[element[0]]
                x10 = self.pos[element[1]] - x0
                x20 = self.pos[element[2]] - x0
                x30 = self.pos[element[3]] - x0
                self.grad_3[i] = ti.Matrix.cols([x10, x20, x30]) @ self.rest_inv_3[i]
            if self.element_dim == 2:
                x0 = self.pos[element[0]]
                x10 = self.pos[element[1]] - x0
                x20 = self.pos[element[2]] - x0
                self.grad_2[i] = ti.Matrix.cols([x10, x20]) @ self.rest_inv_2[i]

    @ti.kernel
    def svd(self):
        for i in range(self.num_elements):
            if self.element_dim == 3:
                u, s, v = ti.svd(self.grad_3[i], ti.f32)
                self.U_3[i] = u
                self.V_3[i] = v
                self.S_3[i] = s
            if self.element_dim == 2:
                u, s, v = ti.svd(self.grad_2[i].transpose() @ self.grad_2[i], ti.f32)
                self.V_2[i] = v
                s0 = ti.sqrt(ti.max(s[0, 0], 1e-8))
                s1 = ti.sqrt(ti.max(s[1, 1], 1e-8))
                self.S_2[i] = ti.Matrix([[s0, 0.0], [0.0, s1]])
                s_inv = ti.Matrix([[1.0 / s0, 0.0], [0.0, 1.0 / s1]])
                self.U_2[i] = self.grad_2[i] @ v @ s_inv

    @ti.kernel
    def calc_PK_stress(self):
        for i in range(self.num_elements):
            if self.element_dim == 3:
                self.PK_3[i] = self.U_3[i] @ self.Sgrad_3[i] @ self.V_3[i].transpose()
            if self.element_dim == 2:
                self.PK_2[i] = self.U_2[i] @ self.Sgrad_2[i] @ self.V_2[i].transpose()

    @ti.kernel
    def calc_internal_forces(self):
        for i in range(self.num_elements):
            vol = self.rest_measure[i]
            element = self.elements[i]
            if self.element_dim == 3:
                forces = -vol * self.PK_3[i] @ self.rest_inv_3[i].transpose()
                f1 = forces[:, 0]
                f2 = forces[:, 1]
                f3 = forces[:, 2]
                f0 = -f1 - f2 - f3
                ti.atomic_add(self.force[element[0]], f0)
                ti.atomic_add(self.force[element[1]], f1)
                ti.atomic_add(self.force[element[2]], f2)
                ti.atomic_add(self.force[element[3]], f3)
            if self.element_dim == 2:
                forces = -vol * self.PK_2[i] @ self.rest_inv_2[i].transpose()
                f1 = forces[:, 0]
                f2 = forces[:, 1]
                f0 = -f1 - f2
                ti.atomic_add(self.force[element[0]], f0)
                ti.atomic_add(self.force[element[1]], f1)
                ti.atomic_add(self.force[element[2]], f2)

    @ti.kernel
    def time_integral(self, dt: ti.f32):  # type: ignore
        for i in range(self.num_vertices):
            self.vel[i] += self.force[i] * dt / self.mass[i]
            self.pos[i] += self.vel[i] * dt
