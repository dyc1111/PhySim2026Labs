import numpy as np

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
    verts = []
    indices = []
    # Vertices
    verts.append([0.0, 1.0, 0.0])  # Top pole
    for i in range(1, lat_segments):
        phi = (i / lat_segments) * np.pi
        for j in range(lon_segments):
            theta = (j / lon_segments) * 2 * np.pi
            x = np.cos(theta) * np.sin(phi)
            y = np.cos(phi)
            z = np.sin(theta) * np.sin(phi)
            verts.append([x, y, z])
    verts.append([0.0, -1.0, 0.0])  # Bottom pole

    lenv = len(verts)

    # Indices
    # Top cap
    for j in range(lon_segments):
        next_j = (j + 1) % lon_segments
        indices.extend([0, 1 + next_j, 1 + j])  # CCW

    # Body
    for i in range(1, lat_segments - 1):
        row_start = 1 + (i - 1) * lon_segments
        next_row_start = 1 + i * lon_segments
        for j in range(lon_segments):
            next_j = (j + 1) % lon_segments

            p00 = row_start + j
            p01 = row_start + next_j
            p10 = next_row_start + j
            p11 = next_row_start + next_j

            # Winding: top-left, top-right, bottom-left
            indices.extend([p00, p01, p10])
            indices.extend([p01, p11, p10])

    # Bottom cap
    bottom_pole = lenv - 1
    last_row_start = 1 + (lat_segments - 2) * lon_segments
    for j in range(lon_segments):
        next_j = (j + 1) % lon_segments
        indices.extend(
            [bottom_pole, last_row_start + j, last_row_start + next_j]
        )  # CCW

    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.int32)


SPHERE_LOCAL_VERTS_NP, SPHERE_INDICES_NP = _generate_sphere_mesh()


def _generate_cylinder_mesh(segments=32):
    verts = []
    indices = []

    # Verts: bottom center and top center
    verts.append([0.0, 0.0, -0.5])
    verts.append([0.0, 0.0, 0.5])

    # Verts: side
    for i in range(segments):
        theta = (i / segments) * 2 * np.pi
        x = np.cos(theta)
        y = np.sin(theta)
        verts.append([x, y, -0.5])
        verts.append([x, y, 0.5])

    # Indices
    for i in range(segments):
        next_i = (i + 1) % segments

        # Bottom cap (z=-0.5, face down)
        indices.extend([0, 2 + next_i * 2, 2 + i * 2])
        # Top cap (z=0.5, face up)
        indices.extend([1, 3 + i * 2, 3 + next_i * 2])

        # Sides
        v1 = 2 + i * 2
        v2 = 3 + i * 2
        v3 = 2 + next_i * 2
        v4 = 3 + next_i * 2
        indices.extend([v1, v3, v2])
        indices.extend([v2, v3, v4])

    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.int32)


CYLINDER_LOCAL_VERTS_NP, CYLINDER_INDICES_NP = _generate_cylinder_mesh()
