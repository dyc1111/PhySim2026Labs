import numpy as np
import taichi as ti
import fcl
import trimesh
from constants import *
from util import (
    euler_angle_to_matrix,
    ray_aabb_intersect,
    ray_sphere_intersect,
    ray_cylinder_intersect,
)


class RigidBody:
    def __init__(self, cfg):
        self.color = np.array(cfg["color"], dtype=np.float32)
        self.dyn_type = cfg["dyn_type"]
        self.position = np.array(cfg["position"], dtype=np.float32)
        self.rotation = euler_angle_to_matrix(cfg["rotation_deg"])
        if self.dyn_type == "freeze":
            self.mass = np.inf
            self.inv_mass = 0.0
            self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif self.dyn_type == "free":
            self.mass = float(cfg["mass"])
            self.inv_mass = 1.0 / self.mass
            self.velocity = np.array(cfg["velocity"], dtype=np.float32)
            self.angular_velocity = np.array(cfg["angular_velocity"], dtype=np.float32)

    @property
    def vertex_count(self):
        raise NotImplementedError

    @property
    def index_count(self):
        raise NotImplementedError

    def get_inertia_diag(self):
        raise NotImplementedError

    def to_fcl(self):
        raise NotImplementedError

    def setup_mesh(self, indices_field, i_offset, v_offset):
        raise NotImplementedError

    def update_mesh_vertices(
        self, pos_field, rot_field, body_id, vertices_field, v_offset
    ):
        raise NotImplementedError

    def ray_intersect(self, orig_l, dir_l):
        raise NotImplementedError


@ti.data_oriented
class Cuboid(RigidBody):
    vertex_count = 8
    index_count = 36

    local_verts = ti.Vector.field(3, dtype=ti.f32, shape=8)
    indices = ti.field(dtype=ti.i32, shape=36)
    _is_initialized = False

    @classmethod
    def initialize(cls):
        if not cls._is_initialized:
            cls.local_verts.from_numpy(CUBE_LOCAL_VERTS_NP)
            cls.indices.from_numpy(CUBE_INDICES_NP)
            cls._is_initialized = True

    def __init__(self, cfg):
        super().__init__(cfg)
        Cuboid.initialize()
        self.size = np.array(cfg["size"], dtype=np.float32)
        self.half_extent = 0.5 * self.size

    def get_inertia_diag(self):
        lx, ly, lz = self.size
        ixx = (self.mass / 12.0) * (ly * ly + lz * lz)
        iyy = (self.mass / 12.0) * (lx * lx + lz * lz)
        izz = (self.mass / 12.0) * (lx * lx + ly * ly)
        return np.array([ixx, iyy, izz], dtype=np.float32)

    def to_fcl(self):
        geom = fcl.Box(self.size[0], self.size[1], self.size[2])
        # The transform will be updated every frame, so initialize with identity
        tf = fcl.Transform()
        return fcl.CollisionObject(geom, tf)

    def ray_intersect(self, orig_l, dir_l):
        if self.dyn_type == "freeze":
            return False, None, None
        return ray_aabb_intersect(orig_l, dir_l, self.half_extent)

    def setup_mesh(self, indices_field, i_offset, v_offset):
        self._setup_mesh(indices_field, i_offset, v_offset)

    @classmethod
    @ti.kernel
    def _setup_mesh(
        cls, indices_field: ti.template(), i_offset: ti.i32, v_offset: ti.i32  # type: ignore
    ):
        for i in range(cls.index_count):
            indices_field[i_offset + i] = v_offset + cls.indices[i]

    def update_mesh_vertices(
        self, pos_field, rot_field, body_id, vertices_field, v_offset
    ):
        self._update_mesh_vertices(
            pos_field,
            rot_field,
            body_id,
            ti.Vector(self.half_extent),
            vertices_field,
            v_offset,
        )

    @classmethod
    @ti.kernel
    def _update_mesh_vertices(
        cls,
        pos_field: ti.template(),  # type: ignore
        rot_field: ti.template(),  # type: ignore
        body_id: ti.i32,  # type: ignore
        half_ext: ti.types.vector(3, ti.f32),  # type: ignore
        vertices_field: ti.template(),  # type: ignore
        v_offset: ti.i32,  # type: ignore
    ):
        pos = pos_field[body_id]
        rot = rot_field[body_id]
        for k in range(cls.vertex_count):
            lv = cls.local_verts[k]
            local = ti.Vector(
                [lv[0] * half_ext[0], lv[1] * half_ext[1], lv[2] * half_ext[2]]
            )
            world = rot @ local + pos
            vertices_field[v_offset + k] = world


@ti.data_oriented
class Sphere(RigidBody):
    _is_initialized = False

    @classmethod
    def initialize(cls):
        if not cls._is_initialized:
            cls.vertex_count = SPHERE_LOCAL_VERTS_NP.shape[0]
            cls.index_count = SPHERE_INDICES_NP.shape[0]
            cls.local_verts = ti.Vector.field(3, dtype=ti.f32, shape=cls.vertex_count)
            cls.indices = ti.field(dtype=ti.i32, shape=cls.index_count)
            cls.local_verts.from_numpy(SPHERE_LOCAL_VERTS_NP)
            cls.indices.from_numpy(SPHERE_INDICES_NP)
            cls._is_initialized = True

    def __init__(self, cfg):
        super().__init__(cfg)
        Sphere.initialize()
        self.radius = float(cfg["size"])

    @property
    def vertex_count(self):
        return Sphere.vertex_count

    @property
    def index_count(self):
        return Sphere.index_count

    def get_inertia_diag(self):
        i_val = (2.0 / 5.0) * self.mass * self.radius * self.radius
        return np.array([i_val, i_val, i_val], dtype=np.float32)

    def to_fcl(self):
        geom = fcl.Sphere(self.radius)
        tf = fcl.Transform()
        return fcl.CollisionObject(geom, tf)

    def ray_intersect(self, orig_l, dir_l):
        if self.dyn_type == "freeze":
            return False, None, None
        return ray_sphere_intersect(orig_l, dir_l, self.radius)

    def setup_mesh(self, indices_field, i_offset, v_offset):
        self._setup_mesh(indices_field, i_offset, v_offset, self.index_count)

    @classmethod
    @ti.kernel
    def _setup_mesh(
        cls, indices_field: ti.template(), i_offset: ti.i32, v_offset: ti.i32, n_idx: ti.i32  # type: ignore
    ):
        for i in range(n_idx):
            indices_field[i_offset + i] = v_offset + cls.indices[i]

    def update_mesh_vertices(
        self, pos_field, rot_field, body_id, vertices_field, v_offset
    ):
        self._update_mesh_vertices(
            pos_field,
            rot_field,
            body_id,
            self.radius,
            vertices_field,
            v_offset,
            self.vertex_count,
        )

    @classmethod
    @ti.kernel
    def _update_mesh_vertices(
        cls,
        pos_field: ti.template(),  # type: ignore
        rot_field: ti.template(),  # type: ignore
        body_id: ti.i32,  # type: ignore
        radius: ti.f32,  # type: ignore
        vertices_field: ti.template(),  # type: ignore
        v_offset: ti.i32,  # type: ignore
        n_vtx: ti.i32,  # type: ignore
    ):
        pos = pos_field[body_id]
        rot = rot_field[body_id]
        for k in range(n_vtx):
            lv = cls.local_verts[k]
            # uniform scaling
            local = ti.Vector([lv[0] * radius, lv[1] * radius, lv[2] * radius])
            world = rot @ local + pos
            vertices_field[v_offset + k] = world


@ti.data_oriented
class Cylinder(RigidBody):
    _is_initialized = False

    @classmethod
    def initialize(cls):
        if not cls._is_initialized:
            cls.vertex_count = CYLINDER_LOCAL_VERTS_NP.shape[0]
            cls.index_count = CYLINDER_INDICES_NP.shape[0]
            cls.local_verts = ti.Vector.field(3, dtype=ti.f32, shape=cls.vertex_count)
            cls.indices = ti.field(dtype=ti.i32, shape=cls.index_count)
            cls.local_verts.from_numpy(CYLINDER_LOCAL_VERTS_NP)
            cls.indices.from_numpy(CYLINDER_INDICES_NP)
            cls._is_initialized = True

    def __init__(self, cfg):
        super().__init__(cfg)
        Cylinder.initialize()
        self.radius = float(cfg["size"][0])
        self.height = float(cfg["size"][1])

    @property
    def vertex_count(self):
        return Cylinder.vertex_count

    @property
    def index_count(self):
        return Cylinder.index_count

    def get_inertia_diag(self):
        ixx = iyy = (self.mass / 12.0) * (3.0 * self.radius**2 + self.height**2)
        izz = (self.mass / 2.0) * self.radius**2
        return np.array([ixx, iyy, izz], dtype=np.float32)

    def to_fcl(self):
        geom = fcl.Cylinder(self.radius, self.height)
        tf = fcl.Transform()
        return fcl.CollisionObject(geom, tf)

    def ray_intersect(self, orig_l, dir_l):
        if self.dyn_type == "freeze":
            return False, None, None
        return ray_cylinder_intersect(orig_l, dir_l, self.radius, self.height)

    def setup_mesh(self, indices_field, i_offset, v_offset):
        self._setup_mesh(indices_field, i_offset, v_offset, self.index_count)

    @classmethod
    @ti.kernel
    def _setup_mesh(
        cls,
        indices_field: ti.template(),  # type: ignore
        i_offset: ti.i32,  # type: ignore
        v_offset: ti.i32,  # type: ignore
        n_idx: ti.i32,  # type: ignore
    ):
        for i in range(n_idx):
            indices_field[i_offset + i] = v_offset + cls.indices[i]

    def update_mesh_vertices(
        self, pos_field, rot_field, body_id, vertices_field, v_offset
    ):
        self._update_mesh_vertices(
            pos_field,
            rot_field,
            body_id,
            self.radius,
            self.height,
            vertices_field,
            v_offset,
            self.vertex_count,
        )

    @classmethod
    @ti.kernel
    def _update_mesh_vertices(
        cls,
        pos_field: ti.template(),  # type: ignore
        rot_field: ti.template(),  # type: ignore
        body_id: ti.i32,  # type: ignore
        radius: ti.f32,  # type: ignore
        height: ti.f32,  # type: ignore
        vertices_field: ti.template(),  # type: ignore
        v_offset: ti.i32,  # type: ignore
        n_vtx: ti.i32,  # type: ignore
    ):
        pos = pos_field[body_id]
        rot = rot_field[body_id]
        for k in range(n_vtx):
            lv = cls.local_verts[k]
            # scale X/Y by radius, Z by height
            local = ti.Vector([lv[0] * radius, lv[1] * radius, lv[2] * height])
            world = rot @ local + pos
            vertices_field[v_offset + k] = world


@ti.data_oriented
class Custom(RigidBody):
    def __init__(self, cfg):
        super().__init__(cfg)
        file_path = cfg["file_path"]
        scale = cfg["size"]
        convexify = cfg["convexify"]

        mesh = trimesh.load(file_path, force="mesh")
        mesh.apply_scale(scale)

        if convexify:
            mesh = mesh.convex_hull

        # Center the mesh at its center of mass
        mesh.vertices -= mesh.center_mass

        self.mesh_data = mesh
        self._vertex_count = len(mesh.vertices)
        self._index_count = len(mesh.faces) * 3

        self.local_verts = ti.Vector.field(3, dtype=ti.f32, shape=self._vertex_count)
        self.indices = ti.field(dtype=ti.i32, shape=self._index_count)

        self.local_verts.from_numpy(mesh.vertices.astype(np.float32))
        self.indices.from_numpy(mesh.faces.flatten().astype(np.int32))

        self.half_extent = np.max(np.abs(mesh.vertices), axis=0).astype(np.float32)

        if mesh.is_volume:
            self.base_inertia = mesh.moment_inertia.copy()
            self.base_mass = max(mesh.mass, 1e-8)
        else:
            # Fallback to an AABB box inertia for unclosed/thin wrapper meshes
            lx, ly, lz = self.half_extent * 2.0
            ixx = (1.0 / 12.0) * (ly * ly + lz * lz)
            iyy = (1.0 / 12.0) * (lx * lx + lz * lz)
            izz = (1.0 / 12.0) * (lx * lx + ly * ly)
            self.base_inertia = np.diag([ixx, iyy, izz])
            self.base_mass = 1.0

    @property
    def vertex_count(self):
        return self._vertex_count

    @property
    def index_count(self):
        return self._index_count

    def get_inertia_diag(self):
        if self.dyn_type == "freeze":
            return np.zeros(3, dtype=np.float32)
        # Base trimesh inertia relies on mesh.mass (which depends on density=1.0)
        # We scale it to match the configured mass.
        I = self.base_inertia * (self.mass / self.base_mass)
        return np.array([I[0, 0], I[1, 1], I[2, 2]], dtype=np.float32)

    def to_fcl(self):
        verts = self.mesh_data.vertices
        faces = self.mesh_data.faces
        bvh = fcl.BVHModel()
        bvh.beginModel(len(verts), len(faces))
        bvh.addSubModel(verts, faces)
        bvh.endModel()
        tf = fcl.Transform()
        return fcl.CollisionObject(bvh, tf)

    def ray_intersect(self, orig_l, dir_l):
        if self.dyn_type == "freeze":
            return False, 0.0, np.zeros(3)
        # Use fast AABB intersection on CPU for real-time mouse picking raycasts
        return ray_aabb_intersect(orig_l, dir_l, self.half_extent)

    def setup_mesh(self, indices_field, i_offset, v_offset):
        self._setup_mesh(indices_field, i_offset, v_offset, self.index_count)

    @ti.kernel
    def _setup_mesh(
        self,
        indices_field: ti.template(),  # type: ignore
        i_offset: ti.i32,  # type: ignore
        v_offset: ti.i32,  # type: ignore
        n_idx: ti.i32,  # type: ignore
    ):
        for i in range(n_idx):
            indices_field[i_offset + i] = self.indices[i] + v_offset

    def update_mesh_vertices(
        self, pos_field, rot_field, body_id, vertices_field, v_offset
    ):
        self._update_mesh_vertices(
            pos_field, rot_field, body_id, vertices_field, v_offset, self.vertex_count
        )

    @ti.kernel
    def _update_mesh_vertices(
        self,
        pos_field: ti.template(),  # type: ignore
        rot_field: ti.template(),  # type: ignore
        body_id: ti.i32,  # type: ignore
        vertices_field: ti.template(),  # type: ignore
        v_offset: ti.i32,  # type: ignore
        n_verts: ti.i32,  # type: ignore
    ):
        pos = pos_field[body_id]
        rot = rot_field[body_id]
        for k in range(n_verts):
            vertices_field[v_offset + k] = rot @ self.local_verts[k] + pos
