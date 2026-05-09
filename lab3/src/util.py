import numpy as np


def get_camera_ray_dir(
    u, v, cam_pos, cam_lookat, cam_up, fov_deg=45.0, aspect=1280.0 / 720.0
):
    fwd = np.asarray(cam_lookat, dtype=np.float32) - np.asarray(
        cam_pos, dtype=np.float32
    )
    fwd /= np.linalg.norm(fwd) + 1e-8
    up0 = np.asarray(cam_up, dtype=np.float32)
    up0 /= np.linalg.norm(up0) + 1e-8
    right = np.cross(fwd, up0)
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up) + 1e-8

    tan_half = np.tan(np.radians(fov_deg) * 0.5)
    x_ndc = 2.0 * u - 1.0
    y_ndc = 2.0 * v - 1.0
    ray = fwd + right * (x_ndc * tan_half * aspect) + up * (y_ndc * tan_half)
    return ray / (np.linalg.norm(ray) + 1e-8)
