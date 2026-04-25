import numpy as np
from util import (
    DEFAULT_SDF_RESOLUTION,
    euler_angle_to_matrix,
    load_custom_mesh,
    load_sdf_cache,
    ray_aabb_intersect,
    ray_cylinder_intersect,
    ray_sphere_intersect,
)
from constants import (
    CUBE_LOCAL_VERTS_NP,
    CUBE_INDICES_NP,
    SPHERE_LOCAL_VERTS_NP,
    SPHERE_INDICES_NP,
    CYLINDER_LOCAL_VERTS_NP,
    CYLINDER_INDICES_NP,
)


class RigidBody:
    def __init__(self, cfg):
        self.dyn_type = cfg.get("dyn_type", "freeze")
        self.color = np.array(cfg.get("color", [0.55, 0.45, 0.35]), dtype=np.float32)
        self.position = np.array(cfg.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.rotation = euler_angle_to_matrix(cfg.get("rotation_deg", [0.0, 0.0, 0.0]))
        if self.dyn_type == "freeze":
            self.mass = np.inf
            self.inv_mass = 0.0
            self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif self.dyn_type == "free":
            self.mass = float(cfg.get("mass", 1.0))
            self.inv_mass = 1.0 / self.mass
            self.velocity = np.array(
                cfg.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32
            )
            self.angular_velocity = np.array(
                cfg.get("angular_velocity", [0.0, 0.0, 0.0]), dtype=np.float32
            )
        else:
            raise NotImplementedError(
                f"Unsupported rigidbody dyn_type: {self.dyn_type}"
            )

    def get_inertia_diag(self):
        raise NotImplementedError

    def ray_intersect(self, orig_l, dir_l):
        raise NotImplementedError

    def get_local_mesh(self):
        raise NotImplementedError

    def intersects_grid_cells(self, cell_centers_world, position, rotation, inflate):
        raise NotImplementedError

    def sample_solid_velocity(
        self, points_world, position, linear_velocity, angular_velocity
    ):
        if points_world.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        rel = points_world - position.reshape(1, 3)
        omega = np.broadcast_to(angular_velocity.reshape(1, 3), rel.shape)
        vel = linear_velocity.reshape(1, 3) + np.cross(omega, rel)
        return vel.astype(np.float32)


class Cuboid(RigidBody):
    type_name = "cuboid"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.size = np.array(cfg["size"], dtype=np.float32)
        self.half_extent = 0.5 * self.size

    def get_inertia_diag(self):
        if self.dyn_type == "freeze":
            return np.zeros(3, dtype=np.float32)
        lx, ly, lz = self.size
        ixx = (self.mass / 12.0) * (ly * ly + lz * lz)
        iyy = (self.mass / 12.0) * (lx * lx + lz * lz)
        izz = (self.mass / 12.0) * (lx * lx + ly * ly)
        return np.array([ixx, iyy, izz], dtype=np.float32)

    def ray_intersect(self, orig_l, dir_l):
        if self.dyn_type == "freeze":
            return False, 0.0, np.zeros(3, dtype=np.float32)
        return ray_aabb_intersect(orig_l, dir_l, self.half_extent)

    def get_local_mesh(self):
        return (CUBE_LOCAL_VERTS_NP * self.half_extent.reshape(1, 3), CUBE_INDICES_NP)

    def intersects_grid_cells(self, cell_centers_world, position, rotation, inflate):
        rel = cell_centers_world - position.reshape(1, 3)
        local = (rotation.T @ rel.T).T
        q = np.abs(local) - self.half_extent.reshape(1, 3)
        outside = np.maximum(q, 0.0)
        outside_dist = np.linalg.norm(outside, axis=1)
        inside_dist = np.minimum(np.maximum(q[:, 0], np.maximum(q[:, 1], q[:, 2])), 0.0)
        return outside_dist + inside_dist <= float(inflate)


class Sphere(RigidBody):
    type_name = "sphere"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.radius = float(cfg["size"])

    def get_inertia_diag(self):
        if self.dyn_type == "freeze":
            return np.zeros(3, dtype=np.float32)
        i_val = (2.0 / 5.0) * self.mass * self.radius * self.radius
        return np.array([i_val, i_val, i_val], dtype=np.float32)

    def ray_intersect(self, orig_l, dir_l):
        if self.dyn_type == "freeze":
            return False, 0.0, np.zeros(3, dtype=np.float32)
        return ray_sphere_intersect(orig_l, dir_l, self.radius)

    def get_local_mesh(self):
        return (SPHERE_LOCAL_VERTS_NP * self.radius, SPHERE_INDICES_NP)

    def intersects_grid_cells(self, cell_centers_world, position, rotation, inflate):
        rel = cell_centers_world - position.reshape(1, 3)
        dist = np.linalg.norm(rel, axis=1)
        return dist <= float(self.radius + inflate)


class Cylinder(RigidBody):
    type_name = "cylinder"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.radius = float(cfg["size"][0])
        self.height = float(cfg["size"][1])

    def get_inertia_diag(self):
        if self.dyn_type == "freeze":
            return np.zeros(3, dtype=np.float32)
        ixx = iyy = (self.mass / 12.0) * (3.0 * self.radius**2 + self.height**2)
        izz = (self.mass / 2.0) * self.radius**2
        return np.array([ixx, iyy, izz], dtype=np.float32)

    def ray_intersect(self, orig_l, dir_l):
        if self.dyn_type == "freeze":
            return False, 0.0, np.zeros(3, dtype=np.float32)
        return ray_cylinder_intersect(orig_l, dir_l, self.radius, self.height)

    def get_local_mesh(self):
        scale = np.array([self.radius, self.radius, self.height], dtype=np.float32)
        return (CYLINDER_LOCAL_VERTS_NP * scale.reshape(1, 3), CYLINDER_INDICES_NP)

    def intersects_grid_cells(self, cell_centers_world, position, rotation, inflate):
        rel = cell_centers_world - position.reshape(1, 3)
        local = (rotation.T @ rel.T).T
        radial = np.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2)
        d0 = radial - self.radius
        d1 = np.abs(local[:, 2]) - 0.5 * self.height
        outside = np.stack([np.maximum(d0, 0.0), np.maximum(d1, 0.0)], axis=1)
        outside_dist = np.linalg.norm(outside, axis=1)
        inside_dist = np.minimum(np.maximum(d0, d1), 0.0)
        return outside_dist + inside_dist <= float(inflate)


class Custom(RigidBody):
    type_name = "custom"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.file_path = cfg["file_path"]
        self.convexify = bool(cfg["convexify"])
        self.sdf_resolution = int(cfg.get("sdf_resolution", DEFAULT_SDF_RESOLUTION))

        mesh, self.scale_vec, _ = load_custom_mesh(
            self.file_path, cfg["size"], self.convexify
        )
        self.mesh_data = mesh

        self.half_extent = np.max(np.abs(mesh.vertices), axis=0).astype(np.float32)
        self._local_mesh_vertices = mesh.vertices.astype(np.float32)
        self._local_mesh_indices = mesh.faces.flatten().astype(np.int32)

        self.base_inertia = mesh.moment_inertia.copy()
        self.base_mass = max(float(mesh.mass), 1e-8)

        cache = load_sdf_cache(
            self.mesh_data,
            self.file_path,
            self.scale_vec,
            self.convexify,
            self.sdf_resolution,
        )

        self.sdf = np.asarray(cache["sdf"], dtype=np.float32)
        self.sdf_bbox_min = np.asarray(cache["bbox_min"], dtype=np.float32)
        self.sdf_bbox_max = np.asarray(cache["bbox_max"], dtype=np.float32)
        self.sdf_res = int(cache["resolution"])
        self.sdf_inv_dx = (self.sdf_res - 1.0) / np.maximum(
            self.sdf_bbox_max - self.sdf_bbox_min, 1e-8
        )

    def get_inertia_diag(self):
        if self.dyn_type == "freeze":
            return np.zeros(3, dtype=np.float32)
        inertia = self.base_inertia * (self.mass / self.base_mass)
        return np.array([inertia[0, 0], inertia[1, 1], inertia[2, 2]], dtype=np.float32)

    def ray_intersect(self, orig_l, dir_l):
        if self.dyn_type == "freeze":
            return False, 0.0, np.zeros(3, dtype=np.float32)
        return ray_aabb_intersect(orig_l, dir_l, self.half_extent)

    def get_local_mesh(self):
        return self._local_mesh_vertices, self._local_mesh_indices

    def _sample_sdf_trilinear(self, points_local):
        points_local = np.asarray(points_local, dtype=np.float32)
        coords = (
            points_local - self.sdf_bbox_min.reshape(1, 3)
        ) * self.sdf_inv_dx.reshape(1, 3)
        coords = np.clip(coords, 0.0, self.sdf_res - 1.0)

        idx0 = np.floor(coords).astype(np.int32)
        idx1 = np.minimum(idx0 + 1, self.sdf_res - 1)
        frac = (coords - idx0.astype(np.float32)).astype(np.float32)

        x0, y0, z0 = idx0[:, 0], idx0[:, 1], idx0[:, 2]
        x1, y1, z1 = idx1[:, 0], idx1[:, 1], idx1[:, 2]

        c000 = self.sdf[x0, y0, z0]
        c100 = self.sdf[x1, y0, z0]
        c010 = self.sdf[x0, y1, z0]
        c110 = self.sdf[x1, y1, z0]
        c001 = self.sdf[x0, y0, z1]
        c101 = self.sdf[x1, y0, z1]
        c011 = self.sdf[x0, y1, z1]
        c111 = self.sdf[x1, y1, z1]

        tx = frac[:, 0]
        ty = frac[:, 1]
        tz = frac[:, 2]

        c00 = c000 * (1.0 - tx) + c100 * tx
        c10 = c010 * (1.0 - tx) + c110 * tx
        c01 = c001 * (1.0 - tx) + c101 * tx
        c11 = c011 * (1.0 - tx) + c111 * tx
        c0 = c00 * (1.0 - ty) + c10 * ty
        c1 = c01 * (1.0 - ty) + c11 * ty
        return c0 * (1.0 - tz) + c1 * tz

    def intersects_grid_cells(self, cell_centers_world, position, rotation, inflate):
        rel = cell_centers_world - position.reshape(1, 3)
        local = (rotation.T @ rel.T).T

        lower = self.sdf_bbox_min.reshape(1, 3) - float(inflate)
        upper = self.sdf_bbox_max.reshape(1, 3) + float(inflate)
        inside_aabb = np.all((local >= lower) & (local <= upper), axis=1)
        if not np.any(inside_aabb):
            return np.zeros((cell_centers_world.shape[0],), dtype=bool)

        phi = np.full((cell_centers_world.shape[0],), np.inf, dtype=np.float32)
        phi[inside_aabb] = self._sample_sdf_trilinear(local[inside_aabb])
        return phi <= float(inflate)


RIGIDBODY_TYPE_TO_CLASS = {
    Cuboid.type_name: Cuboid,
    Sphere.type_name: Sphere,
    Cylinder.type_name: Cylinder,
    Custom.type_name: Custom,
}


def _is_rigidbody_type(body_type):
    return body_type in RIGIDBODY_TYPE_TO_CLASS


def create_rigid_body(cfg):
    body_type = cfg["type"]
    if not _is_rigidbody_type(body_type):
        raise NotImplementedError(f"Unsupported body type: {body_type}")
    return RIGIDBODY_TYPE_TO_CLASS[body_type](cfg)
