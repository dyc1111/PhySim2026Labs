import numpy as np


class Collision:
    def __init__(self, scene):
        self.scene = scene
        self.restitution = 0  # Coefficient of restitution (c)

    def _sat_obb_obb(self, pos_a, rot_a, ext_a, pos_b, rot_b, ext_b):
        # print("sat_obb_obb")
        # print(rot_a, rot_b)
        axes_a = [rot_a[:, 0], rot_a[:, 1], rot_a[:, 2]]
        axes_b = [rot_b[:, 0], rot_b[:, 1], rot_b[:, 2]]

        axes = axes_a + axes_b
        for i in range(3):
            for j in range(3):
                cross = np.cross(axes_a[i], axes_b[j])
                mag = np.linalg.norm(cross)
                if mag > 1e-6:
                    axes.append(cross / mag)

        min_depth = np.inf
        best_n = None

        for axis in axes:
            # print(f"axis: {axis}")
            proj_a = sum(ext_a[i] * abs(np.dot(axis, axes_a[i])) for i in range(3))
            proj_b = sum(ext_b[i] * abs(np.dot(axis, axes_b[i])) for i in range(3))
            # print(proj_a, proj_b)
            dist = abs(np.dot(pos_a - pos_b, axis))

            depth = proj_a + proj_b - dist
            if depth <= 0:
                return False, 0.0, None  # Found a separating axis

            if depth < min_depth:
                min_depth = depth
                best_n = axis

        # Normal should ALWAYS point from B to A for consistency
        # (J n / M_a applied to A, -J n / M_b applied to B)
        if np.dot(best_n, pos_a - pos_b) < 0:
            best_n = -best_n

        return True, min_depth, best_n

    def _get_deepest_vertex(
        self, pos_test, rot_test, ext_test, pos_ref, rot_ref, ext_ref
    ):
        vertices = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    v = pos_test + rot_test @ np.array(
                        [x * ext_test[0], y * ext_test[1], z * ext_test[2]],
                        dtype=np.float32,
                    )
                    vertices.append(v)

        min_sdf = np.inf
        best_v = None
        for v in vertices:
            local_p = rot_ref.T @ (v - pos_ref)
            dx = abs(local_p[0]) - ext_ref[0]
            dy = abs(local_p[1]) - ext_ref[1]
            dz = abs(local_p[2]) - ext_ref[2]
            sdf = max(dx, dy, dz)
            if sdf < min_sdf:
                min_sdf = sdf
                best_v = v

        return best_v, min_sdf

    def get_contact(self, idx_a, idx_b, pos, rot):
        pos_a = pos[idx_a]
        rot_a = rot[idx_a]
        ext_a = self.scene.bodies[idx_a].half_extent

        pos_b = pos[idx_b]
        rot_b = rot[idx_b]
        ext_b = self.scene.bodies[idx_b].half_extent

        is_col, depth, n = self._sat_obb_obb(pos_a, rot_a, ext_a, pos_b, rot_b, ext_b)
        if not is_col:
            return None

        v_a_in_b, sdf_a = self._get_deepest_vertex(
            pos_a, rot_a, ext_a, pos_b, rot_b, ext_b
        )
        v_b_in_a, sdf_b = self._get_deepest_vertex(
            pos_b, rot_b, ext_b, pos_a, rot_a, ext_a
        )

        # Select the vertex that penetrates the deepest as the single contact point approximation
        if sdf_a < sdf_b:
            p_col = v_a_in_b
        else:
            p_col = v_b_in_a

        return {"a": idx_a, "b": idx_b, "normal": n, "point": p_col, "depth": depth}

    def detect_and_resolve(self, pos, vel, rot, ang_vel, dt):
        n_bodies = self.scene.num_bodies[None]
        inv_masses = self.scene.inv_mass.to_numpy()
        inv_inertia_body = self.scene.inv_inertia_body.to_numpy()

        # 1. Collision Detection
        contacts = []
        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                c = self.get_contact(i, j, pos, rot)
                if c is not None:
                    contacts.append(c)

        # 2. Impulse Based Resolution
        for c in contacts:
            a = c["a"]
            b = c["b"]
            n = c["normal"]
            p = c["point"]

            x_a = p - pos[a]
            x_b = p - pos[b]

            v_a = vel[a]
            w_a = ang_vel[a]
            v_b = vel[b]
            w_b = ang_vel[b]

            # Velocity of collision point on body A and B
            v_p_a = v_a + np.cross(w_a, x_a)
            v_p_b = v_b + np.cross(w_b, x_b)
            v_rel = v_p_a - v_p_b

            # Check if bodies are already separating
            v_rel_n = np.dot(v_rel, n)
            if v_rel_n >= 0:
                continue

            inv_M_a = inv_masses[a]
            inv_M_b = inv_masses[b]

            I_inv_a = rot[a] @ inv_inertia_body[a] @ rot[a].T
            I_inv_b = rot[b] @ inv_inertia_body[b] @ rot[b].T

            # Compute angular component
            w_a_part = np.cross(I_inv_a @ np.cross(x_a, n), x_a)
            w_b_part = np.cross(I_inv_b @ np.cross(x_b, n), x_b)

            denom = inv_M_a + inv_M_b + np.dot(w_a_part + w_b_part, n)

            # Compute impulse magnitude J
            # Using Baumgarte Stabilization to counter drift sinking without position projection
            # beta = 0.2, slop = 0.001 (tuning parameters for pushout)
            bias = 0.2 / dt * max(c["depth"] - 0.001, 0.0)

            # Apply restitution only for high velocity impacts to prevent micro-bouncing jitter
            restitution = self.restitution if v_rel_n < -0.1 else 0.0
            J = (-(1.0 + restitution) * v_rel_n + bias) / denom

            # Apply linear impulse
            vel[a] += (J * inv_M_a) * n
            vel[b] -= (J * inv_M_b) * n

            # Apply angular impulse
            ang_vel[a] += I_inv_a @ np.cross(x_a, J * n)
            ang_vel[b] -= I_inv_b @ np.cross(x_b, J * n)

        return pos, vel, rot, ang_vel
