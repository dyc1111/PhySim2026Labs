import numpy as np
import fcl


class Collision:
    def __init__(self, c_cfg, scene):
        self.scene = scene
        self.restitution = c_cfg["restitution"]
        self.mu = c_cfg["mu"]
        self.max_contact = c_cfg["max_c"]
        self.fcl_objs = [body.to_fcl() for body in self.scene.bodies]

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

    def detect_and_resolve(self, pos, vel, rot, ang_vel, dt):
        n_bodies = self.scene.num_bodies[None]
        inv_masses = self.scene.inv_mass.to_numpy()
        inv_inertia_body = self.scene.inv_inertia_body.to_numpy()

        # 1. Collision Detection
        contact_groups = {}
        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                c_list = self.get_contacts(i, j, pos, rot)
                if c_list:
                    contact_groups[(i, j)] = c_list

        if not contact_groups:
            return vel, ang_vel

        contacts = []
        for (a, b), c_list in contact_groups.items():
            n_avg = np.mean([np.array(c[2]) for c in c_list], axis=0)
            n_len = np.linalg.norm(n_avg)
            if n_len < 1e-6:
                n_avg = np.array(c_list[0][2])
            else:
                n_avg /= n_len

            p_avg = np.mean([np.array(c[3]) for c in c_list], axis=0)
            d_max = np.max([c[4] for c in c_list])
            contacts.append((a, b, n_avg, p_avg, d_max))

        delta_vel = np.zeros_like(vel)
        delta_ang_vel = np.zeros_like(ang_vel)

        for a, b, n, p, d in contacts:
            x_a = p - pos[a]
            x_b = p - pos[b]

            # Velocity of collision point on body A and B
            v_p_a = vel[a] + np.cross(ang_vel[a], x_a)
            v_p_b = vel[b] + np.cross(ang_vel[b], x_b)
            v_rel = v_p_a - v_p_b
            # print(v_rel, n, num_c, d)

            # Check if the two bodies are already separating
            v_rel_n = np.dot(v_rel, n)
            if v_rel_n >= 0:
                continue
            v_rel_t = v_rel - v_rel_n * n
            t = v_rel_t / (np.linalg.norm(v_rel_t) + 1e-8)
            v_rel_t = np.linalg.norm(v_rel_t)

            I_inv_a = rot[a] @ inv_inertia_body[a] @ rot[a].T
            I_inv_b = rot[b] @ inv_inertia_body[b] @ rot[b].T

            # Compute angular component
            w_a_part = np.cross(I_inv_a @ np.cross(x_a, n), x_a)
            w_b_part = np.cross(I_inv_b @ np.cross(x_b, n), x_b)

            denom = inv_masses[a] + inv_masses[b] + np.dot(w_a_part + w_b_part, n)

            # Using Baumgarte Stabilization
            bias = (0.2 / dt) * max(d - 0.005, 0.0)

            # Compute impulse and scale by num_c to avoid over-correction
            Jn = (-(1.0 + self.restitution) * v_rel_n + bias) / denom
            Jt_target = -v_rel_t / denom
            Jt = np.clip(Jt_target, -self.mu * Jn, self.mu * Jn)
            J = Jn * n + Jt * t

            # Apply linear impulse
            delta_vel[a] += J * inv_masses[a]
            delta_vel[b] -= J * inv_masses[b]

            # Apply angular impulse
            delta_ang_vel[a] += I_inv_a @ np.cross(x_a, J)
            delta_ang_vel[b] -= I_inv_b @ np.cross(x_b, J)

        vel += delta_vel
        ang_vel += delta_ang_vel

        return vel, ang_vel
