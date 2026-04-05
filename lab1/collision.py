import numpy as np
import fcl
import taichi as ti


@ti.data_oriented
class Collision:
    def __init__(self, c_cfg, scene):
        self.scene = scene
        self.restitution = c_cfg["restitution"]
        self.mu = c_cfg["mu"]
        self.max_contact = c_cfg["max_c"]
        self.fcl_objs = [body.to_fcl() for body in self.scene.bodies]

        n_bodies = self.scene.num_bodies[None]
        self.max_possible_contacts = max(1, n_bodies * (n_bodies - 1) // 2)

        self.contact_a = ti.field(dtype=ti.i32, shape=self.max_possible_contacts)
        self.contact_b = ti.field(dtype=ti.i32, shape=self.max_possible_contacts)
        self.contact_n = ti.Vector.field(
            3, dtype=ti.f32, shape=self.max_possible_contacts
        )
        self.contact_p = ti.Vector.field(
            3, dtype=ti.f32, shape=self.max_possible_contacts
        )
        self.contact_d = ti.field(dtype=ti.f32, shape=self.max_possible_contacts)
        self.num_contacts = ti.field(dtype=ti.i32, shape=())

        self.delta_vel = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)
        self.delta_ang_vel = ti.Vector.field(3, dtype=ti.f32, shape=n_bodies)

        # Force JIT compilation of the collision kernel to prevent stutter on first impact
        self.num_contacts[None] = 0
        self._resolve_contacts(0.016)

    def get_contacts(self, idx_a, idx_b, pos, rot):
        obj_a = self.fcl_objs[idx_a]
        obj_b = self.fcl_objs[idx_b]

        # Update transforms
        obj_a.setTransform(fcl.Transform(rot[idx_a], pos[idx_a]))
        obj_b.setTransform(fcl.Transform(rot[idx_b], pos[idx_b]))

        # Perform collision check
        request = fcl.CollisionRequest(
            num_max_contacts=self.max_contact, enable_contact=True
        )
        result = fcl.CollisionResult()

        fcl.collide(obj_a, obj_b, request, result)

        contacts = []
        if result.is_collision:
            for c in result.contacts:
                n = c.normal

                # We need normal to point from B to A for consistency with our solver
                if np.dot(n, pos[idx_a] - pos[idx_b]) < 0:
                    n = -n

                # FCL already provides correct point and depth.
                contacts.append((idx_a, idx_b, n, c.pos, c.penetration_depth))

        return contacts

    def detect_and_resolve(self, dt):
        n_bodies = self.scene.num_bodies[None]

        pos = self.scene.position.to_numpy()
        rot = self.scene.rotation.to_numpy()

        # 1. Collision Detection
        contact_groups = {}
        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                c_list = self.get_contacts(i, j, pos, rot)
                if c_list:
                    contact_groups[(i, j)] = c_list

        num_c = len(contact_groups)
        if num_c == 0:
            return

        a_arr = np.zeros(num_c, dtype=np.int32)
        b_arr = np.zeros(num_c, dtype=np.int32)
        n_arr = np.zeros((num_c, 3), dtype=np.float32)
        p_arr = np.zeros((num_c, 3), dtype=np.float32)
        d_arr = np.zeros(num_c, dtype=np.float32)

        idx = 0
        for (a, b), c_list in contact_groups.items():
            n_avg = np.mean([np.array(c[2]) for c in c_list], axis=0)
            n_len = np.linalg.norm(n_avg)
            if n_len < 1e-6:
                n_avg = np.array(c_list[0][2])
            else:
                n_avg /= n_len

            p_avg = np.mean([np.array(c[3]) for c in c_list], axis=0)
            d_max = np.max([c[4] for c in c_list])

            a_arr[idx] = a
            b_arr[idx] = b
            n_arr[idx] = n_avg
            p_arr[idx] = p_avg
            d_arr[idx] = d_max
            idx += 1

        self.num_contacts[None] = num_c

        pad_size = self.max_possible_contacts - num_c
        if pad_size > 0:
            a_arr = np.pad(a_arr, (0, pad_size))
            b_arr = np.pad(b_arr, (0, pad_size))
            n_arr = np.pad(n_arr, ((0, pad_size), (0, 0)))
            p_arr = np.pad(p_arr, ((0, pad_size), (0, 0)))
            d_arr = np.pad(d_arr, (0, pad_size))

        self.contact_a.from_numpy(a_arr)
        self.contact_b.from_numpy(b_arr)
        self.contact_n.from_numpy(n_arr)
        self.contact_p.from_numpy(p_arr)
        self.contact_d.from_numpy(d_arr)

        self._resolve_contacts(dt)

    @ti.kernel
    def _resolve_contacts(self, dt: ti.f32):  # type: ignore
        for i in range(self.scene.num_bodies[None]):
            self.delta_vel[i] = ti.Vector.zero(ti.f32, 3)
            self.delta_ang_vel[i] = ti.Vector.zero(ti.f32, 3)

        for i in range(self.num_contacts[None]):
            a = self.contact_a[i]
            b = self.contact_b[i]
            n = self.contact_n[i]
            p = self.contact_p[i]
            d = self.contact_d[i]

            x_a = p - self.scene.position[a]
            x_b = p - self.scene.position[b]

            # Velocity of collision point on body A and B
            v_p_a = self.scene.velocity[a] + self.scene.angular_velocity[a].cross(x_a)
            v_p_b = self.scene.velocity[b] + self.scene.angular_velocity[b].cross(x_b)
            v_rel = v_p_a - v_p_b

            v_rel_n = v_rel.dot(n)
            if v_rel_n < 0:
                v_rel_t = v_rel - v_rel_n * n
                v_rel_t_norm = v_rel_t.norm()
                t = ti.Vector.zero(ti.f32, 3)
                if v_rel_t_norm > 1e-8:
                    t = v_rel_t / v_rel_t_norm

                inv_mass_a = self.scene.inv_mass[a]
                inv_mass_b = self.scene.inv_mass[b]

                rot_a = self.scene.rotation[a]
                rot_b = self.scene.rotation[b]
                I_inv_a = rot_a @ self.scene.inv_inertia_body[a] @ rot_a.transpose()
                I_inv_b = rot_b @ self.scene.inv_inertia_body[b] @ rot_b.transpose()

                w_a_part = (I_inv_a @ (x_a.cross(n))).cross(x_a)
                w_b_part = (I_inv_b @ (x_b.cross(n))).cross(x_b)

                denom = inv_mass_a + inv_mass_b + (w_a_part + w_b_part).dot(n)

                bias = (0.2 / dt) * ti.max(d - 0.005, 0.0)

                Jn = (-(1.0 + self.restitution) * v_rel_n + bias) / denom
                Jt_target = -(v_rel_t_norm / denom)
                Jt = ti.max(-self.mu * Jn, ti.min(self.mu * Jn, Jt_target))

                J = Jn * n + Jt * t

                self.delta_vel[a] += J * inv_mass_a
                self.delta_vel[b] -= J * inv_mass_b

                self.delta_ang_vel[a] += I_inv_a @ (x_a.cross(J))
                self.delta_ang_vel[b] -= I_inv_b @ (x_b.cross(J))

        for i in range(self.scene.num_bodies[None]):
            self.scene.velocity[i] += self.delta_vel[i]
            self.scene.angular_velocity[i] += self.delta_ang_vel[i]
