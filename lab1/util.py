import numpy as np


def euler_angle_to_matrix(rot_deg):
    rx, ry, rz = np.radians(np.array(rot_deg, dtype=np.float32))

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz_m @ ry_m @ rx_m


def compute_tangent_basis(n):
    n = np.array(n, dtype=np.float32)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        n = n / n_norm

    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(n[0]) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    t1 = np.cross(n, ref)
    t1_norm = np.linalg.norm(t1)
    if t1_norm < 1e-8:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        t1 = np.cross(n, ref)
        t1_norm = np.linalg.norm(t1)
    t1 = t1 / (t1_norm + 1e-8)
    t2 = np.cross(n, t1)
    t2 = t2 / (np.linalg.norm(t2) + 1e-8)
    return t1.astype(np.float32), t2.astype(np.float32)


def get_camera_ray_dir(u, v, cam_pos, cam_lookat, cam_up, fov_deg=45.0, aspect=1.0):
    F = np.array(cam_lookat, dtype=np.float32) - np.array(cam_pos, dtype=np.float32)
    F /= np.linalg.norm(F) + 1e-8
    U0 = np.array(cam_up, dtype=np.float32)
    U0 /= np.linalg.norm(U0) + 1e-8
    R = np.cross(F, U0)
    R /= np.linalg.norm(R) + 1e-8
    U = np.cross(R, F)
    U /= np.linalg.norm(U) + 1e-8

    fov_rad = np.radians(fov_deg)
    tan_half_fov = np.tan(fov_rad / 2.0)

    x_ndc = 2.0 * u - 1.0
    y_ndc = 2.0 * v - 1.0

    dir = F + R * (x_ndc * tan_half_fov * aspect) + U * (y_ndc * tan_half_fov)
    return dir / np.linalg.norm(dir)


def ray_aabb_intersect(orig, dir, half_ext):
    t_min = -np.inf
    t_max = np.inf
    hit_normal = np.zeros(3, dtype=np.float32)

    for i in range(3):
        if np.abs(dir[i]) < 1e-8:
            if orig[i] < -half_ext[i] or orig[i] > half_ext[i]:
                return False, 0.0, hit_normal
        else:
            inv_d = 1.0 / dir[i]
            t0 = (-half_ext[i] - orig[i]) * inv_d
            t1 = (half_ext[i] - orig[i]) * inv_d
            n0 = np.zeros(3, dtype=np.float32)
            n0[i] = -1.0
            n1 = np.zeros(3, dtype=np.float32)
            n1[i] = 1.0

            if t0 > t1:
                t0, t1 = t1, t0
                n0, n1 = n1, n0

            if t0 > t_min:
                t_min = t0
                hit_normal = n0

            if t1 < t_max:
                t_max = t1

            if t_max < t_min:
                return False, 0.0, np.zeros(3)

    if t_max < 0.0:
        return False, 0.0, np.zeros(3)

    return True, t_min if t_min > 0 else t_max, hit_normal


def ray_sphere_intersect(orig, dir, radius):
    a = np.dot(dir, dir)
    b = 2.0 * np.dot(orig, dir)
    c = np.dot(orig, orig) - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False, 0.0, np.zeros(3)

    sqrt_disc = np.sqrt(discriminant)
    t0 = (-b - sqrt_disc) / (2.0 * a)
    t1 = (-b + sqrt_disc) / (2.0 * a)

    if t0 > t1:
        t0, t1 = t1, t0

    if t1 < 0.0:
        return False, 0.0, np.zeros(3)

    t = t0 if t0 > 0 else t1
    hit_pos = orig + t * dir
    hit_normal = hit_pos / radius
    return True, t, hit_normal


def ray_cylinder_intersect(orig, dir, radius, height):
    a = dir[0] ** 2 + dir[1] ** 2
    b = 2.0 * (orig[0] * dir[0] + orig[1] * dir[1])
    c = orig[0] ** 2 + orig[1] ** 2 - radius**2

    t_min = np.inf
    hit_normal = np.zeros(3)
    hit = False

    if a > 1e-8:
        discriminant = b**2 - 4 * a * c
        if discriminant >= 0:
            sqrt_disc = np.sqrt(discriminant)
            t0 = (-b - sqrt_disc) / (2.0 * a)
            t1 = (-b + sqrt_disc) / (2.0 * a)

            for t_cyl in [t0, t1]:
                if t_cyl > 0:
                    z = orig[2] + t_cyl * dir[2]
                    if -height / 2 <= z <= height / 2 and t_cyl < t_min:
                        t_min = t_cyl
                        hit = True
                        n = np.array(
                            [orig[0] + t_min * dir[0], orig[1] + t_min * dir[1], 0.0]
                        )
                        hit_normal = n / (np.linalg.norm(n) + 1e-8)

    if abs(dir[2]) > 1e-8:
        for z_cap, nz in [(height / 2, 1.0), (-height / 2, -1.0)]:
            t_cap = (z_cap - orig[2]) / dir[2]
            if t_cap > 0 and t_cap < t_min:
                x = orig[0] + t_cap * dir[0]
                y = orig[1] + t_cap * dir[1]
                if x**2 + y**2 <= radius**2:
                    t_min = t_cap
                    hit = True
                    hit_normal = np.array([0.0, 0.0, nz])

    return hit, t_min if hit else 0.0, hit_normal
