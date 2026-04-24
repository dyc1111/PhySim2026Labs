from enum import Enum
import numpy as np


class CellType(Enum):
    CELL_AIR = 0
    CELL_WATER = 1
    CELL_SOLID = 2


bbox_verts = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)

bbox_indices = np.array(
    [
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        0,
        4,
        5,
        5,
        6,
        6,
        7,
        7,
        4,
        0,
        4,
        1,
        5,
        2,
        6,
        3,
        7,
    ],
    dtype=np.int32,
)

CUBE_LOCAL_VERTS_NP = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)

CUBE_INDICES_NP = np.array(
    [
        0,
        1,
        2,
        0,
        2,
        3,
        4,
        6,
        5,
        4,
        7,
        6,
        0,
        4,
        5,
        0,
        5,
        1,
        1,
        5,
        6,
        1,
        6,
        2,
        2,
        6,
        7,
        2,
        7,
        3,
        3,
        7,
        4,
        3,
        4,
        0,
    ],
    dtype=np.int32,
)


def _generate_sphere_mesh(lat_segments=16, lon_segments=32):
    verts = [[0.0, 1.0, 0.0]]
    indices = []
    for i in range(1, lat_segments):
        phi = (i / lat_segments) * np.pi
        for j in range(lon_segments):
            theta = (j / lon_segments) * 2.0 * np.pi
            x = np.cos(theta) * np.sin(phi)
            y = np.cos(phi)
            z = np.sin(theta) * np.sin(phi)
            verts.append([x, y, z])
    verts.append([0.0, -1.0, 0.0])

    num_verts = len(verts)
    for j in range(lon_segments):
        next_j = (j + 1) % lon_segments
        indices.extend([0, 1 + next_j, 1 + j])

    for i in range(1, lat_segments - 1):
        row_start = 1 + (i - 1) * lon_segments
        next_row_start = 1 + i * lon_segments
        for j in range(lon_segments):
            next_j = (j + 1) % lon_segments
            p00 = row_start + j
            p01 = row_start + next_j
            p10 = next_row_start + j
            p11 = next_row_start + next_j
            indices.extend([p00, p01, p10])
            indices.extend([p01, p11, p10])

    bottom = num_verts - 1
    last_row_start = 1 + (lat_segments - 2) * lon_segments
    for j in range(lon_segments):
        next_j = (j + 1) % lon_segments
        indices.extend([bottom, last_row_start + j, last_row_start + next_j])

    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.int32)


def _generate_cylinder_mesh(segments=32):
    verts = [[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]]
    indices = []

    for i in range(segments):
        theta = (i / segments) * 2.0 * np.pi
        x = np.cos(theta)
        y = np.sin(theta)
        verts.append([x, y, -0.5])
        verts.append([x, y, 0.5])

    for i in range(segments):
        nxt = (i + 1) % segments
        indices.extend([0, 2 + 2 * nxt, 2 + 2 * i])
        indices.extend([1, 3 + 2 * i, 3 + 2 * nxt])
        v1 = 2 + 2 * i
        v2 = 3 + 2 * i
        v3 = 2 + 2 * nxt
        v4 = 3 + 2 * nxt
        indices.extend([v1, v3, v2])
        indices.extend([v2, v3, v4])

    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.int32)


SPHERE_LOCAL_VERTS_NP, SPHERE_INDICES_NP = _generate_sphere_mesh()
CYLINDER_LOCAL_VERTS_NP, CYLINDER_INDICES_NP = _generate_cylinder_mesh()
