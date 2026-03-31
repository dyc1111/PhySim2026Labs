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


def skew_symmetric(v):
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float32,
    )


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
