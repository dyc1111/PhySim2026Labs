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


def cuboid_inertia_diag(m, size):
    lx, ly, lz = size
    ixx = (m / 12.0) * (ly * ly + lz * lz)
    iyy = (m / 12.0) * (lx * lx + lz * lz)
    izz = (m / 12.0) * (lx * lx + ly * ly)
    return np.array([ixx, iyy, izz], dtype=np.float32)


def skew_symmetric(v):
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float32,
    )


def integrate_rotation(r, omega, dt):
    dtheta = omega * dt
    theta = np.linalg.norm(dtheta)
    if theta < 1e-7:
        delta = np.eye(3, dtype=np.float32) + skew_symmetric(dtheta)
    else:
        axis = dtheta / theta
        k = skew_symmetric(axis)
        delta = (
            np.eye(3, dtype=np.float32)
            + np.sin(theta) * k
            + (1.0 - np.cos(theta)) * (k @ k)
        )
    return delta @ r
