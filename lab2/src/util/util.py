import numpy as np
import taichi as ti


def euler_angle_to_matrix(rot_deg):
    rx, ry, rz = np.radians(np.array(rot_deg, dtype=np.float32))

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz_m @ ry_m @ rx_m


@ti.func
def skew_symmetric(v: ti.template()):  # type: ignore
    return ti.Matrix([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def get_camera_ray_dir(u, v, cam_pos, cam_lookat, cam_up, fov_deg=45.0, aspect=1.0):
    fwd = np.array(cam_lookat, dtype=np.float32) - np.array(cam_pos, dtype=np.float32)
    fwd /= np.linalg.norm(fwd) + 1e-8
    up0 = np.array(cam_up, dtype=np.float32)
    up0 /= np.linalg.norm(up0) + 1e-8
    right = np.cross(fwd, up0)
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up) + 1e-8

    fov_rad = np.radians(fov_deg)
    tan_half_fov = np.tan(fov_rad / 2.0)

    x_ndc = 2.0 * u - 1.0
    y_ndc = 2.0 * v - 1.0
    direction = (
        fwd + right * (x_ndc * tan_half_fov * aspect) + up * (y_ndc * tan_half_fov)
    )
    return direction / (np.linalg.norm(direction) + 1e-8)


def ray_aabb_intersect(orig, direction, half_extent):
    t_min = -np.inf
    t_max = np.inf
    hit_normal = np.zeros(3, dtype=np.float32)

    for axis in range(3):
        if np.abs(direction[axis]) < 1e-8:
            if orig[axis] < -half_extent[axis] or orig[axis] > half_extent[axis]:
                return False, 0.0, hit_normal
        else:
            inv_dir = 1.0 / direction[axis]
            t0 = (-half_extent[axis] - orig[axis]) * inv_dir
            t1 = (half_extent[axis] - orig[axis]) * inv_dir
            n0 = np.zeros(3, dtype=np.float32)
            n1 = np.zeros(3, dtype=np.float32)
            n0[axis] = -1.0
            n1[axis] = 1.0

            if t0 > t1:
                t0, t1 = t1, t0
                n0, n1 = n1, n0

            if t0 > t_min:
                t_min = t0
                hit_normal = n0
            if t1 < t_max:
                t_max = t1

            if t_max < t_min:
                return False, 0.0, np.zeros(3, dtype=np.float32)

    if t_max < 0.0:
        return False, 0.0, np.zeros(3, dtype=np.float32)
    return True, t_min if t_min > 0.0 else t_max, hit_normal


def ray_sphere_intersect(orig, direction, radius):
    a = np.dot(direction, direction)
    b = 2.0 * np.dot(orig, direction)
    c = np.dot(orig, orig) - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False, 0.0, np.zeros(3, dtype=np.float32)

    sqrt_disc = np.sqrt(disc)
    t0 = (-b - sqrt_disc) / (2.0 * a)
    t1 = (-b + sqrt_disc) / (2.0 * a)
    if t0 > t1:
        t0, t1 = t1, t0
    if t1 < 0.0:
        return False, 0.0, np.zeros(3, dtype=np.float32)

    t = t0 if t0 > 0.0 else t1
    hit_pos = orig + t * direction
    hit_normal = hit_pos / (np.linalg.norm(hit_pos) + 1e-8)
    return True, t, hit_normal.astype(np.float32)


def ray_cylinder_intersect(orig, direction, radius, height):
    a = direction[0] ** 2 + direction[1] ** 2
    b = 2.0 * (orig[0] * direction[0] + orig[1] * direction[1])
    c = orig[0] ** 2 + orig[1] ** 2 - radius**2

    t_min = np.inf
    hit_normal = np.zeros(3, dtype=np.float32)
    hit = False

    if a > 1e-8:
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            sqrt_disc = np.sqrt(disc)
            t0 = (-b - sqrt_disc) / (2.0 * a)
            t1 = (-b + sqrt_disc) / (2.0 * a)
            for t_cyl in (t0, t1):
                if t_cyl > 0.0:
                    z = orig[2] + t_cyl * direction[2]
                    if -0.5 * height <= z <= 0.5 * height and t_cyl < t_min:
                        t_min = t_cyl
                        hit = True
                        n = np.array(
                            [
                                orig[0] + t_cyl * direction[0],
                                orig[1] + t_cyl * direction[1],
                                0.0,
                            ],
                            dtype=np.float32,
                        )
                        hit_normal = n / (np.linalg.norm(n) + 1e-8)

    if abs(direction[2]) > 1e-8:
        for z_cap, nz in ((0.5 * height, 1.0), (-0.5 * height, -1.0)):
            t_cap = (z_cap - orig[2]) / direction[2]
            if t_cap > 0.0 and t_cap < t_min:
                x = orig[0] + t_cap * direction[0]
                y = orig[1] + t_cap * direction[1]
                if x * x + y * y <= radius * radius:
                    t_min = t_cap
                    hit = True
                    hit_normal = np.array([0.0, 0.0, nz], dtype=np.float32)

    return hit, t_min if hit else 0.0, hit_normal
