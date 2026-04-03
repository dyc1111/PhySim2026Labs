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
    for i in range(lat_segments + 1):
        v = i / lat_segments
        phi = v * np.pi
        for j in range(lon_segments):
            u = j / lon_segments
            theta = u * 2 * np.pi
            x = np.cos(theta) * np.sin(phi)
            y = np.cos(phi)
            z = np.sin(theta) * np.sin(phi)
            verts.append([x, y, z])
            
    # Indices
    for i in range(lat_segments):
        for j in range(lon_segments):
            first = (i * lon_segments) + j
            second = first + lon_segments
            next_j = (j + 1) % lon_segments
            
            indices.extend([first, second, first + next_j - j])
            indices.extend([second, second + next_j - j, first + next_j - j])
            
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
        indices.extend([v1, v2, v3])
        indices.extend([v2, v4, v3])
        
    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.int32)

CYLINDER_LOCAL_VERTS_NP, CYLINDER_INDICES_NP = _generate_cylinder_mesh()
