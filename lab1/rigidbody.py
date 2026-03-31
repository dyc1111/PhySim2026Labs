import numpy as np
import taichi as ti
from constants import *
from util import euler_angle_to_matrix, ray_aabb_intersect


class RigidBody:
    def __init__(self, cfg):
        self.mass = float(cfg["mass"])
        self.position = np.array(cfg["position"], dtype=np.float32)
        self.velocity = np.array(cfg["velocity"], dtype=np.float32)
        self.angular_velocity = np.array(cfg["angular_velocity"], dtype=np.float32)
        self.rotation = euler_angle_to_matrix(cfg["rotation_deg"])
        self.color = np.array(cfg["color"], dtype=np.float32)

    @property
    def vertex_count(self):
        raise NotImplementedError

    @property
    def index_count(self):
        raise NotImplementedError

    def get_inertia_diag(self):
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

    def ray_intersect(self, orig_l, dir_l):
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
