from dataclasses import dataclass
import numpy as np


@dataclass
class MeshData:
    vertices: np.ndarray
    elements: np.ndarray
    surface_faces: np.ndarray
    rest_inv_3: np.ndarray
    rest_inv_2: np.ndarray
    rest_measure: np.ndarray
    masses: np.ndarray
    element_dim: int


def _orient_tet_positive(vertices, tet):
    tet = list(tet)
    x0, x1, x2, x3 = vertices[tet]
    det = np.linalg.det(np.column_stack((x1 - x0, x2 - x0, x3 - x0)))
    if det < 0.0:
        tet[2], tet[3] = tet[3], tet[2]
    return tet


def _oriented_face_away_from_vertex(vertices, face, opposite):
    face = list(face)
    x0, x1, x2 = vertices[face]
    normal = np.cross(x1 - x0, x2 - x0)
    if np.dot(normal, vertices[opposite] - x0) > 0.0:
        face[1], face[2] = face[2], face[1]
    return tuple(face)


def extract_tet_surface_faces(vertices, elements):
    faces = {}
    for tet in elements:
        a, b, c, d = [int(x) for x in tet[:4]]
        candidates = (
            ((b, c, d), a),
            ((a, d, c), b),
            ((a, b, d), c),
            ((a, c, b), d),
        )
        for face, opposite in candidates:
            key = tuple(sorted(face))
            oriented = _oriented_face_away_from_vertex(vertices, face, opposite)
            if key in faces:
                faces[key] = None
            else:
                faces[key] = oriented

    exposed = [face for face in faces.values() if face is not None]
    return np.asarray(exposed, dtype=np.int32)


def build_cuboid_mesh(cfg):
    origin = np.asarray(cfg["origin"], dtype=np.float32)
    size = np.asarray(cfg["size"], dtype=np.float32)
    resolution = np.asarray(cfg["resolution"], dtype=np.int32)
    density = cfg["density"]

    nx, ny, nz = [int(x) for x in resolution]
    vertices = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                uvw = np.array([i / nx, j / ny, k / nz], dtype=np.float32)
                vertices.append(origin + uvw * size)
    vertices = np.asarray(vertices, dtype=np.float32)

    def _grid_id(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    elements = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                ids = {
                    "000": _grid_id(i, j, k),
                    "001": _grid_id(i, j, k + 1),
                    "010": _grid_id(i, j + 1, k),
                    "011": _grid_id(i, j + 1, k + 1),
                    "100": _grid_id(i + 1, j, k),
                    "101": _grid_id(i + 1, j, k + 1),
                    "110": _grid_id(i + 1, j + 1, k),
                    "111": _grid_id(i + 1, j + 1, k + 1),
                }
                local = (
                    (ids["000"], ids["001"], ids["011"], ids["111"]),
                    (ids["000"], ids["010"], ids["011"], ids["111"]),
                    (ids["000"], ids["001"], ids["101"], ids["111"]),
                    (ids["000"], ids["100"], ids["101"], ids["111"]),
                    (ids["000"], ids["010"], ids["110"], ids["111"]),
                    (ids["000"], ids["100"], ids["110"], ids["111"]),
                )
                elements.extend(_orient_tet_positive(vertices, tet) for tet in local)

    elements = np.asarray(elements, dtype=np.int32)
    rest_inv_3 = np.zeros((len(elements), 3, 3), dtype=np.float32)
    rest_inv_2 = np.zeros((len(elements), 2, 2), dtype=np.float32)
    rest_measure = np.zeros((len(elements),), dtype=np.float32)
    masses = np.zeros((len(vertices),), dtype=np.float32)

    for i, element in enumerate(elements):
        x0, x1, x2, x3 = vertices[element]
        E = np.column_stack((x1 - x0, x2 - x0, x3 - x0)).astype(np.float32)
        det = float(np.linalg.det(E))
        volume = abs(det) / 6.0
        rest_inv_3[i] = np.linalg.inv(E).astype(np.float32)
        rest_measure[i] = volume
        for vid in element:
            masses[int(vid)] += density * volume / 4.0

    return MeshData(
        vertices=vertices,
        elements=elements,
        surface_faces=extract_tet_surface_faces(vertices, elements),
        rest_inv_3=rest_inv_3,
        rest_inv_2=rest_inv_2,
        rest_measure=rest_measure,
        masses=masses,
        element_dim=3,
    )


def build_cloth_mesh(cfg):
    origin = np.asarray(cfg["origin"], dtype=np.float32)
    size = np.asarray(cfg["size"], dtype=np.float32)
    resolution = np.asarray(cfg["resolution"], dtype=np.int32)
    density = cfg["density"]

    nx, nz = [int(x) for x in resolution]
    vertices = []
    rest_uv = []
    for i in range(nx + 1):
        for k in range(nz + 1):
            u = i / nx
            v = k / nz
            vertices.append(
                [origin[0] + u * size[0], origin[1], origin[2] + v * size[1]]
            )
            rest_uv.append([u * size[0], v * size[1]])
    vertices = np.asarray(vertices, dtype=np.float32)
    rest_uv = np.asarray(rest_uv, dtype=np.float32)

    def _grid_id(i, k):
        return i * (nz + 1) + k

    triangles = []
    for i in range(nx):
        for k in range(nz):
            v00 = _grid_id(i, k)
            v10 = _grid_id(i + 1, k)
            v01 = _grid_id(i, k + 1)
            v11 = _grid_id(i + 1, k + 1)
            triangles.append((v00, v11, v10))
            triangles.append((v00, v01, v11))

    triangles = np.asarray(triangles, dtype=np.int32)
    elements = np.full((len(triangles), 4), -1, dtype=np.int32)
    elements[:, :3] = triangles
    rest_inv_3 = np.zeros((len(elements), 3, 3), dtype=np.float32)
    rest_inv_2 = np.zeros((len(elements), 2, 2), dtype=np.float32)
    rest_measure = np.zeros((len(elements),), dtype=np.float32)
    masses = np.zeros((len(vertices),), dtype=np.float32)

    for i, triangle in enumerate(triangles):
        uv0, uv1, uv2 = rest_uv[triangle]
        E = np.column_stack((uv1 - uv0, uv2 - uv0)).astype(np.float32)
        area = abs(float(np.linalg.det(E))) / 2.0
        rest_inv_2[i] = np.linalg.inv(E).astype(np.float32)
        rest_measure[i] = area
        for vidx in triangle:
            masses[int(vidx)] += density * area / 3.0

    return MeshData(
        vertices=vertices,
        elements=elements,
        surface_faces=triangles,
        rest_inv_3=rest_inv_3,
        rest_inv_2=rest_inv_2,
        rest_measure=rest_measure,
        masses=masses,
        element_dim=2,
    )


def build_mesh(scene_cfg):
    scene_type = scene_cfg["type"]
    if scene_type == "cuboid":
        return build_cuboid_mesh(scene_cfg)
    if scene_type == "cloth":
        return build_cloth_mesh(scene_cfg)
    raise NotImplementedError(f"Unsupported scene type: {scene_type}")
