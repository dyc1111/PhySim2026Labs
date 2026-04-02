import numpy as np
import fcl


class Collision:
    def __init__(self, scene):
        self.scene = scene
        self.restitution = 1  # Coefficient of restitution (c)
        self.fcl_objs = [body.to_fcl() for body in self.scene.bodies]

    def get_contacts(self, idx_a, idx_b, pos, rot):
        obj_a = self.fcl_objs[idx_a]
        obj_b = self.fcl_objs[idx_b]

        # Update transforms
        obj_a.setTransform(fcl.Transform(rot[idx_a], pos[idx_a]))
        obj_b.setTransform(fcl.Transform(rot[idx_b], pos[idx_b]))

        # Perform collision check
        request = fcl.CollisionRequest(num_max_contacts=10, enable_contact=True)
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
        contacts = []
        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                c_list = self.get_contacts(i, j, pos, rot)
                contacts.extend(c_list)

        if not contacts:
            return vel, ang_vel

        # Precompute denominators and set up accumulated impulses
        solver_contacts = []
        for a, b, n, p, d in contacts:
            x_a = p - pos[a]
            x_b = p - pos[b]

            I_inv_a = rot[a] @ inv_inertia_body[a] @ rot[a].T
            I_inv_b = rot[b] @ inv_inertia_body[b] @ rot[b].T

            # Compute angular component
            w_a_part = np.cross(I_inv_a @ np.cross(x_a, n), x_a)
            w_b_part = np.cross(I_inv_b @ np.cross(x_b, n), x_b)

            denom = inv_masses[a] + inv_masses[b] + np.dot(w_a_part + w_b_part, n)

            # Using Baumgarte Stabilization scaled down for multi-iterations
            bias = (0.2 / dt) * max(d - 0.005, 0.0)

            solver_contacts.append(
                {
                    "a": a,
                    "b": b,
                    "n": n,
                    "p": p,
                    "x_a": x_a,
                    "x_b": x_b,
                    "I_inv_a": I_inv_a,
                    "I_inv_b": I_inv_b,
                    "denom": denom,
                    "bias": bias,
                    "acc_J": 0.0,
                }
            )

        for c in solver_contacts:
            a = c["a"]
            b = c["b"]
            n = c["n"]

            # Velocity of collision point on body A and B
            v_p_a = vel[a] + np.cross(ang_vel[a], c["x_a"])
            v_p_b = vel[b] + np.cross(ang_vel[b], c["x_b"])
            v_rel = v_p_a - v_p_b

            v_rel_n = np.dot(v_rel, n)

            # Compute raw impulse
            J = (-(1.0 + self.restitution) * v_rel_n + c["bias"]) / c["denom"]

            # Apply linear impulse
            vel[a] += (J * inv_masses[a]) * n
            vel[b] -= (J * inv_masses[b]) * n

            # Apply angular impulse
            ang_vel[a] += c["I_inv_a"] @ np.cross(c["x_a"], J * n)
            ang_vel[b] -= c["I_inv_b"] @ np.cross(c["x_b"], J * n)

        return vel, ang_vel
