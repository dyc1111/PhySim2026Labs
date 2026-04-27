from abc import ABC, abstractmethod
import taichi as ti

from constants import CellType
from scene import Scene


class AdvectionStrategyBase(ABC):
    @abstractmethod
    def handle_advection(self, dt):
        """Advect simulation state."""
        return NotImplementedError


@ti.data_oriented
class GravityIntegration(AdvectionStrategyBase):
    def __init__(self, scene: Scene):
        self.scene = scene

    @ti.kernel
    def handle_advection(self, dt: float):
        for p in range(self.scene.num_particles):
            self.scene.particle_vel[p] += self.scene.gravity * dt
            self.scene.particle_pos[p] += self.scene.particle_vel[p] * dt


@ti.data_oriented
class SemiLagrangian(AdvectionStrategyBase):
    def __init__(self, scene: Scene, extrapolation_iters: int):
        self.scene = scene
        self.extrapolation_iters = max(1, int(extrapolation_iters))

        self.u_src = ti.field(dtype=ti.f32, shape=scene.grid_u.shape)
        self.v_src = ti.field(dtype=ti.f32, shape=scene.grid_v.shape)
        self.w_src = ti.field(dtype=ti.f32, shape=scene.grid_w.shape)

        self.u_ext = ti.field(dtype=ti.f32, shape=scene.grid_u.shape)
        self.v_ext = ti.field(dtype=ti.f32, shape=scene.grid_v.shape)
        self.w_ext = ti.field(dtype=ti.f32, shape=scene.grid_w.shape)
        self.u_ext_next = ti.field(dtype=ti.f32, shape=scene.grid_u.shape)
        self.v_ext_next = ti.field(dtype=ti.f32, shape=scene.grid_v.shape)
        self.w_ext_next = ti.field(dtype=ti.f32, shape=scene.grid_w.shape)

        self.u_mask = ti.field(dtype=ti.i32, shape=scene.grid_u.shape)
        self.v_mask = ti.field(dtype=ti.i32, shape=scene.grid_v.shape)
        self.w_mask = ti.field(dtype=ti.i32, shape=scene.grid_w.shape)
        self.u_mask_next = ti.field(dtype=ti.i32, shape=scene.grid_u.shape)
        self.v_mask_next = ti.field(dtype=ti.i32, shape=scene.grid_v.shape)
        self.w_mask_next = ti.field(dtype=ti.i32, shape=scene.grid_w.shape)

        self.u_solid = ti.field(dtype=ti.i32, shape=scene.grid_u.shape)
        self.v_solid = ti.field(dtype=ti.i32, shape=scene.grid_v.shape)
        self.w_solid = ti.field(dtype=ti.i32, shape=scene.grid_w.shape)
        self.u_solid_vel = ti.field(dtype=ti.f32, shape=scene.grid_u.shape)
        self.v_solid_val = ti.field(dtype=ti.f32, shape=scene.grid_v.shape)
        self.w_solid_val = ti.field(dtype=ti.f32, shape=scene.grid_w.shape)

        self.u_adv = ti.field(dtype=ti.f32, shape=scene.grid_u.shape)
        self.v_adv = ti.field(dtype=ti.f32, shape=scene.grid_v.shape)
        self.w_adv = ti.field(dtype=ti.f32, shape=scene.grid_w.shape)

    @ti.func
    def _cell_type(self, i, j, k, nx, ny, nz):
        t = CellType.CELL_SOLID.value
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            t = self.scene.grid_cell_type[i, j, k]
        return t

    @ti.func
    def _trilerp(self, field, gx, gy, gz, sx, sy, sz):
        i0 = ti.cast(ti.floor(gx), ti.i32)
        j0 = ti.cast(ti.floor(gy), ti.i32)
        k0 = ti.cast(ti.floor(gz), ti.i32)

        fx = gx - i0
        fy = gy - j0
        fz = gz - k0

        num, denom = 0.0, 1e-8
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    wx = fx if di else (1 - fx)
                    wy = fy if dj else (1 - fy)
                    wz = fz if dk else (1 - fz)
                    w = wx * wy * wz
                    i = i0 + di
                    j = j0 + dj
                    k = k0 + dk
                    if 0 <= i < sx and 0 <= j < sy and 0 <= k < sz:
                        num += w * field[i, j, k]
                        denom += w
        return num / denom

    @ti.func
    def _interpolate_u(self, field, pos):
        nx, ny, nz = self.scene.grid_resolution
        gx = pos[0] / self.scene.grid_dx
        gy = pos[1] / self.scene.grid_dy - 0.5
        gz = pos[2] / self.scene.grid_dz - 0.5
        return self._trilerp(field, gx, gy, gz, nx + 1, ny, nz)

    @ti.func
    def _interpolate_v(self, field, pos):
        nx, ny, nz = self.scene.grid_resolution
        gx = pos[0] / self.scene.grid_dx - 0.5
        gy = pos[1] / self.scene.grid_dy
        gz = pos[2] / self.scene.grid_dz - 0.5
        return self._trilerp(field, gx, gy, gz, nx, ny + 1, nz)

    @ti.func
    def _interpolate_w(self, field, pos):
        nx, ny, nz = self.scene.grid_resolution
        gx = pos[0] / self.scene.grid_dx - 0.5
        gy = pos[1] / self.scene.grid_dy - 0.5
        gz = pos[2] / self.scene.grid_dz
        return self._trilerp(field, gx, gy, gz, nx, ny, nz + 1)

    @ti.func
    def _interpolate_vel(self, pos):
        return ti.Vector(
            [
                self._interpolate_u(self.u_ext, pos),
                self._interpolate_v(self.v_ext, pos),
                self._interpolate_w(self.w_ext, pos),
            ]
        )

    @ti.kernel
    def _read(self):
        for I in ti.grouped(self.scene.grid_u):
            self.u_src[I] = self.scene.grid_u[I]
        for I in ti.grouped(self.scene.grid_v):
            self.v_src[I] = self.scene.grid_v[I]
        for I in ti.grouped(self.scene.grid_w):
            self.w_src[I] = self.scene.grid_w[I]

    @ti.kernel
    def _build_face_meta(self):
        nx, ny, nz = self.scene.grid_resolution

        for I in ti.grouped(self.u_src):
            i, j, k = I
            left = self._cell_type(i - 1, j, k, nx, ny, nz)
            right = self._cell_type(i, j, k, nx, ny, nz)
            is_solid = ti.cast(
                left == CellType.CELL_SOLID.value or right == CellType.CELL_SOLID.value,
                ti.i32,
            )
            self.u_solid[I] = is_solid
            solid_vel = 0.0
            count = 0.0
            if i > 0 and left == CellType.CELL_SOLID.value:
                solid_vel += self.scene.grid_solid_velocity[i - 1, j, k][0]
                count += 1.0
            if i < nx and right == CellType.CELL_SOLID.value:
                solid_vel += self.scene.grid_solid_velocity[i, j, k][0]
                count += 1.0
            if count > 0.0:
                solid_vel /= count
            self.u_solid_vel[I] = solid_vel

            fluid_known = ti.cast(
                left == CellType.CELL_WATER.value or right == CellType.CELL_WATER.value,
                ti.i32,
            )
            known = ti.cast(is_solid == 1 or fluid_known == 1, ti.i32)
            self.u_mask[I] = known
            if is_solid == 1:
                self.u_ext[I] = solid_vel
            else:
                self.u_ext[I] = self.u_src[I]

        for I in ti.grouped(self.v_src):
            i, j, k = I
            down = self._cell_type(i, j - 1, k, nx, ny, nz)
            up = self._cell_type(i, j, k, nx, ny, nz)
            is_solid = ti.cast(
                down == CellType.CELL_SOLID.value or up == CellType.CELL_SOLID.value,
                ti.i32,
            )
            self.v_solid[I] = is_solid
            solid_vel = 0.0
            count = 0.0
            if j > 0 and down == CellType.CELL_SOLID.value:
                solid_vel += self.scene.grid_solid_velocity[i, j - 1, k][1]
                count += 1.0
            if j < ny and up == CellType.CELL_SOLID.value:
                solid_vel += self.scene.grid_solid_velocity[i, j, k][1]
                count += 1.0
            if count > 0.0:
                solid_vel /= count
            self.v_solid_val[I] = solid_vel

            fluid_known = ti.cast(
                down == CellType.CELL_WATER.value or up == CellType.CELL_WATER.value,
                ti.i32,
            )
            known = ti.cast(is_solid == 1 or fluid_known == 1, ti.i32)
            self.v_mask[I] = known
            if is_solid == 1:
                self.v_ext[I] = solid_vel
            else:
                self.v_ext[I] = self.v_src[I]

        for I in ti.grouped(self.w_src):
            i, j, k = I
            back = self._cell_type(i, j, k - 1, nx, ny, nz)
            front = self._cell_type(i, j, k, nx, ny, nz)
            is_solid = ti.cast(
                back == CellType.CELL_SOLID.value or front == CellType.CELL_SOLID.value,
                ti.i32,
            )
            self.w_solid[I] = is_solid
            solid_vel = 0.0
            count = 0.0
            if k > 0 and back == CellType.CELL_SOLID.value:
                solid_vel += self.scene.grid_solid_velocity[i, j, k - 1][2]
                count += 1.0
            if k < nz and front == CellType.CELL_SOLID.value:
                solid_vel += self.scene.grid_solid_velocity[i, j, k][2]
                count += 1.0
            if count > 0.0:
                solid_vel /= count
            self.w_solid_val[I] = solid_vel

            fluid_known = ti.cast(
                back == CellType.CELL_WATER.value or front == CellType.CELL_WATER.value,
                ti.i32,
            )
            known = ti.cast(is_solid == 1 or fluid_known == 1, ti.i32)
            self.w_mask[I] = known
            if is_solid == 1:
                self.w_ext[I] = solid_vel
            else:
                self.w_ext[I] = self.w_src[I]

    @ti.kernel
    def _extrapolate_u(
        self,
        src: ti.template(),  # type: ignore
        src_mask: ti.template(),  # type: ignore
        dst: ti.template(),  # type: ignore
        dst_mask: ti.template(),  # type: ignore
    ):
        sx, sy, sz = self.scene.grid_u.shape
        for I in ti.grouped(src):
            i, j, k = I
            if self.u_solid[I] == 1:
                dst[I] = self.u_solid_vel[I]
                dst_mask[I] = 1
            elif src_mask[I] == 1:
                dst[I] = src[I]
                dst_mask[I] = 1
            else:
                total = 0.0
                count = 0
                if i > 0 and src_mask[i - 1, j, k] == 1:
                    total += src[i - 1, j, k]
                    count += 1
                if i + 1 < sx and src_mask[i + 1, j, k] == 1:
                    total += src[i + 1, j, k]
                    count += 1
                if j > 0 and src_mask[i, j - 1, k] == 1:
                    total += src[i, j - 1, k]
                    count += 1
                if j + 1 < sy and src_mask[i, j + 1, k] == 1:
                    total += src[i, j + 1, k]
                    count += 1
                if k > 0 and src_mask[i, j, k - 1] == 1:
                    total += src[i, j, k - 1]
                    count += 1
                if k + 1 < sz and src_mask[i, j, k + 1] == 1:
                    total += src[i, j, k + 1]
                    count += 1
                if count > 0:
                    dst[I] = total / ti.cast(count, ti.f32)
                    dst_mask[I] = 1
                else:
                    dst[I] = src[I]
                    dst_mask[I] = 0

    @ti.kernel
    def _extrapolate_v(
        self,
        src: ti.template(),  # type: ignore
        src_mask: ti.template(),  # type: ignore
        dst: ti.template(),  # type: ignore
        dst_mask: ti.template(),  # type: ignore
    ):
        sx, sy, sz = self.scene.grid_v.shape
        for I in ti.grouped(src):
            i, j, k = I
            if self.v_solid[I] == 1:
                dst[I] = self.v_solid_val[I]
                dst_mask[I] = 1
            elif src_mask[I] == 1:
                dst[I] = src[I]
                dst_mask[I] = 1
            else:
                total = 0.0
                count = 0
                if i > 0 and src_mask[i - 1, j, k] == 1:
                    total += src[i - 1, j, k]
                    count += 1
                if i + 1 < sx and src_mask[i + 1, j, k] == 1:
                    total += src[i + 1, j, k]
                    count += 1
                if j > 0 and src_mask[i, j - 1, k] == 1:
                    total += src[i, j - 1, k]
                    count += 1
                if j + 1 < sy and src_mask[i, j + 1, k] == 1:
                    total += src[i, j + 1, k]
                    count += 1
                if k > 0 and src_mask[i, j, k - 1] == 1:
                    total += src[i, j, k - 1]
                    count += 1
                if k + 1 < sz and src_mask[i, j, k + 1] == 1:
                    total += src[i, j, k + 1]
                    count += 1
                if count > 0:
                    dst[I] = total / ti.cast(count, ti.f32)
                    dst_mask[I] = 1
                else:
                    dst[I] = src[I]
                    dst_mask[I] = 0

    @ti.kernel
    def _extrapolate_w(
        self,
        src: ti.template(),  # type: ignore
        src_mask: ti.template(),  # type: ignore
        dst: ti.template(),  # type: ignore
        dst_mask: ti.template(),  # type: ignore
    ):
        sx, sy, sz = self.scene.grid_w.shape
        for I in ti.grouped(src):
            i, j, k = I
            if self.w_solid[I] == 1:
                dst[I] = self.w_solid_val[I]
                dst_mask[I] = 1
            elif src_mask[I] == 1:
                dst[I] = src[I]
                dst_mask[I] = 1
            else:
                total = 0.0
                count = 0
                if i > 0 and src_mask[i - 1, j, k] == 1:
                    total += src[i - 1, j, k]
                    count += 1
                if i + 1 < sx and src_mask[i + 1, j, k] == 1:
                    total += src[i + 1, j, k]
                    count += 1
                if j > 0 and src_mask[i, j - 1, k] == 1:
                    total += src[i, j - 1, k]
                    count += 1
                if j + 1 < sy and src_mask[i, j + 1, k] == 1:
                    total += src[i, j + 1, k]
                    count += 1
                if k > 0 and src_mask[i, j, k - 1] == 1:
                    total += src[i, j, k - 1]
                    count += 1
                if k + 1 < sz and src_mask[i, j, k + 1] == 1:
                    total += src[i, j, k + 1]
                    count += 1
                if count > 0:
                    dst[I] = total / ti.cast(count, ti.f32)
                    dst_mask[I] = 1
                else:
                    dst[I] = src[I]
                    dst_mask[I] = 0

    @ti.kernel
    def _advect_u(self, dt: ti.f32):  # type: ignore
        for I in ti.grouped(self.u_adv):
            i, j, k = I
            if self.u_solid[I] == 1:
                self.u_adv[I] = self.u_solid_vel[I]
            else:
                x0 = ti.Vector(
                    [
                        ti.cast(i, ti.f32) * self.scene.grid_dx,
                        (ti.cast(j, ti.f32) + 0.5) * self.scene.grid_dy,
                        (ti.cast(k, ti.f32) + 0.5) * self.scene.grid_dz,
                    ]
                )
                v0 = self._interpolate_vel(x0)
                xmid = x0 - 0.5 * dt * v0
                vmid = self._interpolate_vel(xmid)
                xback = x0 - dt * vmid
                self.u_adv[I] = (
                    self._interpolate_u(self.u_ext, xback) + self.scene.gravity[0] * dt
                )

    @ti.kernel
    def _advect_v(self, dt: ti.f32):  # type: ignore
        for I in ti.grouped(self.v_adv):
            i, j, k = I
            if self.v_solid[I] == 1:
                self.v_adv[I] = self.v_solid_val[I]
            else:
                x0 = ti.Vector(
                    [
                        (ti.cast(i, ti.f32) + 0.5) * self.scene.grid_dx,
                        ti.cast(j, ti.f32) * self.scene.grid_dy,
                        (ti.cast(k, ti.f32) + 0.5) * self.scene.grid_dz,
                    ]
                )
                v0 = self._interpolate_vel(x0)
                xmid = x0 - 0.5 * dt * v0
                vmid = self._interpolate_vel(xmid)
                xback = x0 - dt * vmid
                self.v_adv[I] = (
                    self._interpolate_v(self.v_ext, xback) + self.scene.gravity[1] * dt
                )

    @ti.kernel
    def _advect_w(self, dt: ti.f32):  # type: ignore
        for I in ti.grouped(self.w_adv):
            i, j, k = I
            if self.w_solid[I] == 1:
                self.w_adv[I] = self.w_solid_val[I]
            else:
                x0 = ti.Vector(
                    [
                        (ti.cast(i, ti.f32) + 0.5) * self.scene.grid_dx,
                        (ti.cast(j, ti.f32) + 0.5) * self.scene.grid_dy,
                        ti.cast(k, ti.f32) * self.scene.grid_dz,
                    ]
                )
                v0 = self._interpolate_vel(x0)
                xmid = x0 - 0.5 * dt * v0
                vmid = self._interpolate_vel(xmid)
                xback = x0 - dt * vmid
                self.w_adv[I] = (
                    self._interpolate_w(self.w_ext, xback) + self.scene.gravity[2] * dt
                )

    @ti.kernel
    def _write(self):
        for I in ti.grouped(self.scene.grid_u):
            self.scene.grid_u[I] = self.u_adv[I]
        for I in ti.grouped(self.scene.grid_v):
            self.scene.grid_v[I] = self.v_adv[I]
        for I in ti.grouped(self.scene.grid_w):
            self.scene.grid_w[I] = self.w_adv[I]

    def _extrapolate_faces(self):
        for _ in range(self.extrapolation_iters):
            self._extrapolate_u(
                self.u_ext, self.u_mask, self.u_ext_next, self.u_mask_next
            )
            self.u_ext.copy_from(self.u_ext_next)
            self.u_mask.copy_from(self.u_mask_next)

            self._extrapolate_v(
                self.v_ext, self.v_mask, self.v_ext_next, self.v_mask_next
            )
            self.v_ext.copy_from(self.v_ext_next)
            self.v_mask.copy_from(self.v_mask_next)

            self._extrapolate_w(
                self.w_ext, self.w_mask, self.w_ext_next, self.w_mask_next
            )
            self.w_ext.copy_from(self.w_ext_next)
            self.w_mask.copy_from(self.w_mask_next)

    def handle_advection(self, dt: float):
        self._read()
        self._build_face_meta()
        self._extrapolate_faces()
        self._advect_u(dt)
        self._advect_v(dt)
        self._advect_w(dt)
        self._write()
