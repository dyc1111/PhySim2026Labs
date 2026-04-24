import numpy as np
from util import (
    euler_angle_to_matrix,
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

    def intersects_grid_cells(
        self,
        cell_centers_world: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
        inflate: float,
    ) -> np.ndarray:
        raise NotImplementedError

    def sample_solid_velocity(
        self,
        points_world: np.ndarray,
        position: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> np.ndarray:
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

    def intersects_grid_cells(
        self,
        cell_centers_world: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
        inflate: float,
    ) -> np.ndarray:
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

    def intersects_grid_cells(
        self,
        cell_centers_world: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
        inflate: float,
    ) -> np.ndarray:
        del rotation
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

    def intersects_grid_cells(
        self,
        cell_centers_world: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
        inflate: float,
    ) -> np.ndarray:
        rel = cell_centers_world - position.reshape(1, 3)
        local = (rotation.T @ rel.T).T
        radial = np.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2)
        d0 = radial - self.radius
        d1 = np.abs(local[:, 2]) - 0.5 * self.height
        outside = np.stack([np.maximum(d0, 0.0), np.maximum(d1, 0.0)], axis=1)
        outside_dist = np.linalg.norm(outside, axis=1)
        inside_dist = np.minimum(np.maximum(d0, d1), 0.0)
        return outside_dist + inside_dist <= float(inflate)


RIGIDBODY_TYPE_TO_CLASS = {
    Cuboid.type_name: Cuboid,
    Sphere.type_name: Sphere,
    Cylinder.type_name: Cylinder,
}


def _is_rigidbody_type(body_type):
    return body_type in RIGIDBODY_TYPE_TO_CLASS


def create_rigid_body(cfg):
    body_type = cfg["type"]
    if not _is_rigidbody_type(body_type):
        raise NotImplementedError(f"Unsupported body type: {body_type}")
    return RIGIDBODY_TYPE_TO_CLASS[body_type](cfg)
