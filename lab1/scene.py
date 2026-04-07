import taichi as ti
import numpy as np
from rigidbody import create_rigid_body, is_rigidbody_type
from articulated import create_articulated_body, is_articulated_type
from util import compute_tangent_basis


@ti.data_oriented
class Scene:
    def __init__(self, scene_cfg):
        self.gravity = np.array(scene_cfg["gravity"], dtype=np.float32)
        objects = scene_cfg["objects"]

        self.bodies = []
        self.articulated_bodies = []
        self.joint_constraints = []
        self.parent_child_pairs = set()
        body_to_articulation = []
        for cfg in objects:
            body_type = cfg["type"]
            if is_articulated_type(body_type):
                articulation = create_articulated_body(cfg)
                art_id = len(self.articulated_bodies)
                self.articulated_bodies.append(articulation)

                link_ids = []
                for link_body in articulation.links:
                    link_ids.append(len(self.bodies))
                    self.bodies.append(link_body)
                    body_to_articulation.append(art_id)

                for constraint in articulation.get_joint_constraints(link_ids):
                    self.joint_constraints.append(constraint)
                    pair = tuple(sorted((constraint["parent"], constraint["child"])))
                    self.parent_child_pairs.add(pair)
            elif is_rigidbody_type(body_type):
                body = create_rigid_body(cfg)
                self.bodies.append(body)
                body_to_articulation.append(-1)
            else:
                raise NotImplementedError(f"Unsupported body type: {body_type}")

        self.body_to_articulation = np.array(body_to_articulation, dtype=np.int32)

        self.num_bodies = ti.field(dtype=ti.i32, shape=())
        n_bodies = len(self.bodies)
        self.num_bodies[None] = n_bodies

        self.position = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.rotation = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_bodies)
        self.angular_velocity = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.mass = ti.field(dtype=ti.f32, shape=n_bodies)
        self.inv_mass = ti.field(dtype=ti.f32, shape=n_bodies)
        self.inertia_body = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_bodies)
        self.inv_inertia_body = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_bodies)

        total_vertices = sum(b.vertex_count for b in self.bodies)
        total_indices = sum(b.index_count for b in self.bodies)

        self.mesh_vertices = ti.Vector.field(3, dtype=ti.f32, shape=total_vertices)
        self.mesh_indices = ti.field(dtype=ti.i32, shape=total_indices)
        self.mesh_colors = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.index_offset = ti.field(dtype=ti.i32, shape=n_bodies)
        self.index_count = ti.field(dtype=ti.i32, shape=n_bodies)

        v_offset = 0
        i_offset = 0

        for i, body in enumerate(self.bodies):
            self.position[i] = body.position
            self.velocity[i] = body.velocity
            self.angular_velocity[i] = body.angular_velocity
            self.rotation[i] = body.rotation
            self.mass[i] = body.mass
            self.inv_mass[i] = body.inv_mass

            inertia_diag = body.get_inertia_diag()
            self.inertia_body[i] = np.diag(inertia_diag)

            if body.dyn_type == "freeze":
                self.inv_inertia_body[i] = np.zeros((3, 3), dtype=np.float32)
            else:
                self.inv_inertia_body[i] = np.diag(1.0 / np.maximum(inertia_diag, 1e-8))

            self.mesh_colors[i] = body.color

            self.index_count[i] = body.index_count
            self.index_offset[i] = i_offset

            body.setup_mesh(self.mesh_indices, i_offset, v_offset)

            v_offset += body.vertex_count
            i_offset += body.index_count

        self.init_pos = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.init_vel = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.init_rot = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_bodies)
        self.init_ang_vel = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.init_pos.copy_from(self.position)
        self.init_vel.copy_from(self.velocity)
        self.init_rot.copy_from(self.rotation)
        self.init_ang_vel.copy_from(self.angular_velocity)

        self.update_mesh_vertices()

    @ti.func
    def _get_skew_symmetric(self, v: ti.template()):  # type: ignore
        return ti.Matrix([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

    def calc_jacobian(self, contacts):
        n_bodies = self.num_bodies[None]
        n_contacts = len(contacts)
        if n_contacts == 0:
            return np.zeros((0, 6 * n_bodies), dtype=np.float32)

        pos = self.position.to_numpy()
        J = np.zeros((3 * n_contacts, 6 * n_bodies), dtype=np.float32)

        for i, c in enumerate(contacts):
            a, b, n, p, _ = c
            n = np.array(n, dtype=np.float32)
            n = n / (np.linalg.norm(n) + 1e-8)
            t1, t2 = compute_tangent_basis(n)
            directions = (n, t1, t2)

            p_w = np.array(p, dtype=np.float32)
            r_a = p_w - pos[a]
            r_b = p_w - pos[b]

            for j, direction in enumerate(directions):
                row = 3 * i + j
                J[row, 6 * a : 6 * a + 6] = self.bodies[a].calc_jacobian(
                    r_a, direction, 1.0
                )
                J[row, 6 * b : 6 * b + 6] = self.bodies[b].calc_jacobian(
                    r_b, direction, -1.0
                )

        return J

    def calc_mass_inverse_matrix(self):
        n_bodies = self.num_bodies[None]
        rot = self.rotation.to_numpy()
        inv_mass = self.inv_mass.to_numpy()
        inv_inertia_body = self.inv_inertia_body.to_numpy()

        M_inv = np.zeros((6 * n_bodies, 6 * n_bodies), dtype=np.float32)
        for i in range(n_bodies):
            M_inv[6 * i : 6 * i + 3, 6 * i : 6 * i + 3] = inv_mass[i] * np.eye(
                3, dtype=np.float32
            )
            I_inv_world = rot[i] @ inv_inertia_body[i] @ rot[i].T
            M_inv[6 * i + 3 : 6 * i + 6, 6 * i + 3 : 6 * i + 6] = I_inv_world

        return M_inv

    def calc_joint_jacobian(self, dt, beta):
        n_bodies = self.num_bodies[None]
        n_joint_constraints = len(self.joint_constraints)
        if n_joint_constraints == 0:
            return (
                np.zeros((0, 6 * n_bodies), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        pos = self.position.to_numpy()
        rot = self.rotation.to_numpy()
        axes = np.eye(3, dtype=np.float32)
        stabilization = (float(beta) / float(dt)) if dt > 1e-8 else 0.0

        rows = []
        rhs = []

        for joint in self.joint_constraints:
            a = int(joint["parent"])
            b = int(joint["child"])

            anchor_parent_local = np.array(
                joint["anchor_parent_local"], dtype=np.float32
            )
            anchor_child_local = np.array(joint["anchor_child_local"], dtype=np.float32)

            r_a = rot[a] @ anchor_parent_local
            r_b = rot[b] @ anchor_child_local
            p_a = pos[a] + r_a
            p_b = pos[b] + r_b
            pos_error = p_a - p_b

            for direction in axes:
                row = np.zeros((6 * n_bodies,), dtype=np.float32)
                row[6 * a : 6 * a + 6] = self.bodies[a].calc_jacobian(
                    r_a, direction, 1.0
                )
                row[6 * b : 6 * b + 6] = self.bodies[b].calc_jacobian(
                    r_b, direction, -1.0
                )
                rows.append(row)
                rhs.append(-stabilization * float(np.dot(pos_error, direction)))

            if joint["joint_type"] == "revolute":
                axis_parent = rot[a] @ np.array(
                    joint["axis_parent_local"], dtype=np.float32
                )
                axis_child = rot[b] @ np.array(
                    joint["axis_child_local"], dtype=np.float32
                )
                axis_parent = axis_parent / (np.linalg.norm(axis_parent) + 1e-8)
                axis_child = axis_child / (np.linalg.norm(axis_child) + 1e-8)

                t1, t2 = compute_tangent_basis(axis_parent)
                axis_error = np.cross(axis_parent, axis_child)

                for tangent in (t1, t2):
                    row = np.zeros((6 * n_bodies,), dtype=np.float32)
                    row[6 * a + 3 : 6 * a + 6] = tangent
                    row[6 * b + 3 : 6 * b + 6] = -tangent
                    rows.append(row)
                    rhs.append(-stabilization * float(np.dot(axis_error, tangent)))

        J_joint = np.stack(rows, axis=0).astype(np.float32)
        rhs_joint = np.array(rhs, dtype=np.float32)
        return J_joint, rhs_joint

    def get_generalized_velocity(self):
        vel = self.velocity.to_numpy()
        ang_vel = self.angular_velocity.to_numpy()
        n_bodies = self.num_bodies[None]
        v = np.zeros(6 * n_bodies, dtype=np.float32)
        for i in range(n_bodies):
            v[6 * i : 6 * i + 3] = vel[i]
            v[6 * i + 3 : 6 * i + 6] = ang_vel[i]
        return v

    def set_generalized_velocity(self, v_generalized):
        n_bodies = self.num_bodies[None]
        vel = np.zeros((n_bodies, 3), dtype=np.float32)
        ang_vel = np.zeros((n_bodies, 3), dtype=np.float32)
        for i in range(n_bodies):
            vel[i] = v_generalized[6 * i : 6 * i + 3]
            ang_vel[i] = v_generalized[6 * i + 3 : 6 * i + 6]
        self.velocity.from_numpy(vel)
        self.angular_velocity.from_numpy(ang_vel)

    @ti.kernel
    def pre_solve_kinematics(
        self, dt: ti.f32, forces: ti.types.ndarray(), torques: ti.types.ndarray()  # type: ignore
    ):
        for i in range(self.num_bodies[None]):
            if self.inv_mass[i] > 0.0:
                # linear
                f = ti.Vector([forces[i, 0], forces[i, 1], forces[i, 2]])
                g = ti.Vector([self.gravity[0], self.gravity[1], self.gravity[2]])
                self.velocity[i] += dt * f * self.inv_mass[i]
                self.velocity[i] += dt * g * self.inv_mass[i]

                # angular
                tau = ti.Vector([torques[i, 0], torques[i, 1], torques[i, 2]])
                R = self.rotation[i]
                I_inv = R @ self.inv_inertia_body[i] @ R.transpose()
                I_curr = R @ self.inertia_body[i] @ R.transpose()
                omega = self.angular_velocity[i]

                tau_total = tau - omega.cross(I_curr @ omega)
                self.angular_velocity[i] += dt * (I_inv @ tau_total)

    @ti.kernel
    def post_solve_kinematics(self, dt: ti.f32):  # type: ignore
        for i in range(self.num_bodies[None]):
            if self.inv_mass[i] > 0.0:
                self.position[i] += dt * self.velocity[i]

                omega = self.angular_velocity[i]
                dtheta = omega * dt
                theta = dtheta.norm()
                delta = ti.Matrix.identity(ti.f32, 3)
                if theta < 1e-7:
                    delta += self._get_skew_symmetric(dtheta)
                else:
                    axis = dtheta / theta
                    k = self._get_skew_symmetric(axis)
                    delta += ti.sin(theta) * k + (1.0 - ti.cos(theta)) * (k @ k)
                self.rotation[i] = delta @ self.rotation[i]

    def update_mesh_vertices(self):
        v_offset = 0
        for i, body in enumerate(self.bodies):
            body.update_mesh_vertices(
                self.position, self.rotation, i, self.mesh_vertices, v_offset
            )
            v_offset += body.vertex_count

    def reset(self):
        self.position.copy_from(self.init_pos)
        self.velocity.copy_from(self.init_vel)
        self.rotation.copy_from(self.init_rot)
        self.angular_velocity.copy_from(self.init_ang_vel)

    def get_state(self):
        pos = self.position.to_numpy()
        vel = self.velocity.to_numpy()
        rot = self.rotation.to_numpy()
        ang_vel = self.angular_velocity.to_numpy()
        return pos, vel, rot, ang_vel
