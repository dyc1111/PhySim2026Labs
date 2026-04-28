from abc import ABC, abstractmethod
import taichi as ti
from constants import CellType
from scene import Scene
from util import bspline


class TransferStrategyBase(ABC):
    @abstractmethod
    def handle_transfer(self, sdt, is_p2g):
        """Perform velocity transfer between particles and grid."""
        return NotImplementedError


@ti.data_oriented
class FpicTransferStragety(TransferStrategyBase):
    def __init__(self, scene: Scene, flip_ratio):
        self.scene = scene
        self.flip_ratio = flip_ratio
        self.pic_vel = ti.Vector.field(3, dtype=ti.f32, shape=self.scene.num_particles)
        self.flip_delta_vel = ti.Vector.field(
            3, dtype=ti.f32, shape=self.scene.num_particles
        )

    @ti.kernel
    def _p2g_transfer(self):
        eps = 1e-8
        for I in ti.grouped(self.scene.grid_u_num):
            self.scene.grid_u_num[I] = 0.0
            self.scene.grid_u_denom[I] = eps
            self.scene.grid_u_prev[I] = self.scene.grid_u[I]
        for I in ti.grouped(self.scene.grid_v_num):
            self.scene.grid_v_num[I] = 0.0
            self.scene.grid_v_denom[I] = eps
            self.scene.grid_v_prev[I] = self.scene.grid_v[I]
        for I in ti.grouped(self.scene.grid_w_num):
            self.scene.grid_w_num[I] = 0.0
            self.scene.grid_w_denom[I] = eps
            self.scene.grid_w_prev[I] = self.scene.grid_w[I]

        dx, dy, dz = self.scene.grid_dx, self.scene.grid_dy, self.scene.grid_dz
        nx, ny, nz = self.scene.grid_resolution

        for p in range(self.scene.num_particles):
            pos = self.scene.particle_pos[p]
            vel = self.scene.particle_vel[p]

            x = pos[0]
            y = pos[1]
            z = pos[2]
            xh = pos[0] - dx / 2
            yh = pos[1] - dy / 2
            zh = pos[2] - dz / 2

            i = ti.cast(ti.floor(x / dx), ti.i32)
            j = ti.cast(ti.floor(y / dy), ti.i32)
            k = ti.cast(ti.floor(z / dz), ti.i32)
            ih = ti.cast(ti.floor(xh / dx), ti.i32)
            jh = ti.cast(ti.floor(yh / dy), ti.i32)
            kh = ti.cast(ti.floor(zh / dz), ti.i32)

            fx = x / dx - i
            fy = y / dy - j
            fz = z / dz - k
            fxh = xh / dx - ih
            fyh = yh / dy - jh
            fzh = zh / dz - kh

            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        wx = fx if di else (1 - fx)
                        wy = fy if dj else (1 - fy)
                        wz = fz if dk else (1 - fz)
                        wxh = fxh if di else (1 - fxh)
                        wyh = fyh if dj else (1 - fyh)
                        wzh = fzh if dk else (1 - fzh)

                        wu = wx * wyh * wzh
                        wv = wxh * wy * wzh
                        ww = wxh * wyh * wz

                        if (
                            0 <= i + di < nx + 1
                            and 0 <= jh + dj < ny
                            and 0 <= kh + dk < nz
                        ):
                            ti.atomic_add(
                                self.scene.grid_u_num[i + di, jh + dj, kh + dk],
                                wu * vel[0],
                            )
                            ti.atomic_add(
                                self.scene.grid_u_denom[i + di, jh + dj, kh + dk], wu
                            )

                        if (
                            0 <= ih + di < nx
                            and 0 <= j + dj < ny + 1
                            and 0 <= kh + dk < nz
                        ):
                            ti.atomic_add(
                                self.scene.grid_v_num[ih + di, j + dj, kh + dk],
                                wv * vel[1],
                            )
                            ti.atomic_add(
                                self.scene.grid_v_denom[ih + di, j + dj, kh + dk], wv
                            )

                        if (
                            0 <= ih + di < nx
                            and 0 <= jh + dj < ny
                            and 0 <= k + dk < nz + 1
                        ):
                            ti.atomic_add(
                                self.scene.grid_w_num[ih + di, jh + dj, k + dk],
                                ww * vel[2],
                            )
                            ti.atomic_add(
                                self.scene.grid_w_denom[ih + di, jh + dj, k + dk], ww
                            )

        for I in ti.grouped(self.scene.grid_u_num):
            self.scene.grid_u[I] = self.scene.grid_u_num[I] / self.scene.grid_u_denom[I]
        for I in ti.grouped(self.scene.grid_v_num):
            self.scene.grid_v[I] = self.scene.grid_v_num[I] / self.scene.grid_v_denom[I]
        for I in ti.grouped(self.scene.grid_w_num):
            self.scene.grid_w[I] = self.scene.grid_w_num[I] / self.scene.grid_w_denom[I]

        for I in ti.grouped(self.scene.grid_cell_type):
            i, j, k = I
            if self.scene.grid_cell_type[I] != CellType.CELL_SOLID.value:
                continue
            if (
                i > 0
                and self.scene.grid_cell_type[i - 1, j, k] == CellType.CELL_SOLID.value
            ):
                self.scene.grid_u[I] = self.scene.grid_u_prev[I]
            if (
                j > 0
                and self.scene.grid_cell_type[i, j - 1, k] == CellType.CELL_SOLID.value
            ):
                self.scene.grid_v[I] = self.scene.grid_v_prev[I]
            if (
                k > 0
                and self.scene.grid_cell_type[i, j, k - 1] == CellType.CELL_SOLID.value
            ):
                self.scene.grid_w[I] = self.scene.grid_w_prev[I]

        for I in ti.grouped(self.scene.grid_u):
            self.scene.grid_u_prev[I] = self.scene.grid_u[I]
        for I in ti.grouped(self.scene.grid_v):
            self.scene.grid_v_prev[I] = self.scene.grid_v[I]
        for I in ti.grouped(self.scene.grid_w):
            self.scene.grid_w_prev[I] = self.scene.grid_w[I]

    @ti.kernel
    def _g2p_transfer(
        self, pic_vel: ti.template(), flip_delta_vel: ti.template()  # type: ignore
    ):
        dx, dy, dz = self.scene.grid_dx, self.scene.grid_dy, self.scene.grid_dz
        nx, ny, nz = self.scene.grid_resolution

        for p in range(self.scene.num_particles):
            pos = self.scene.particle_pos[p]

            x = pos[0]
            y = pos[1]
            z = pos[2]
            xh = pos[0] - dx / 2
            yh = pos[1] - dy / 2
            zh = pos[2] - dz / 2

            i = ti.cast(ti.floor(x / dx), ti.i32)
            j = ti.cast(ti.floor(y / dy), ti.i32)
            k = ti.cast(ti.floor(z / dz), ti.i32)
            ih = ti.cast(ti.floor(xh / dx), ti.i32)
            jh = ti.cast(ti.floor(yh / dy), ti.i32)
            kh = ti.cast(ti.floor(zh / dz), ti.i32)

            fx = x / dx - i
            fy = y / dy - j
            fz = z / dz - k
            fxh = xh / dx - ih
            fyh = yh / dy - jh
            fzh = zh / dz - kh

            u_num_pic, v_num_pic, w_num_pic = 0.0, 0.0, 0.0
            u_num_flip, v_num_flip, w_num_flip = 0.0, 0.0, 0.0
            u_denom, v_denom, w_denom = 1e-8, 1e-8, 1e-8

            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        wx = fx if di else (1 - fx)
                        wy = fy if dj else (1 - fy)
                        wz = fz if dk else (1 - fz)
                        wxh = fxh if di else (1 - fxh)
                        wyh = fyh if dj else (1 - fyh)
                        wzh = fzh if dk else (1 - fzh)

                        wu = wx * wyh * wzh
                        wv = wxh * wy * wzh
                        ww = wxh * wyh * wz

                        if (
                            0 <= i + di < nx + 1
                            and 0 <= jh + dj < ny
                            and 0 <= kh + dk < nz
                        ):
                            u_num_pic += (
                                wu * self.scene.grid_u[i + di, jh + dj, kh + dk]
                            )
                            u_num_flip += wu * (
                                self.scene.grid_u[i + di, jh + dj, kh + dk]
                                - self.scene.grid_u_prev[i + di, jh + dj, kh + dk]
                            )
                            u_denom += wu

                        if (
                            0 <= ih + di < nx
                            and 0 <= j + dj < ny + 1
                            and 0 <= kh + dk < nz
                        ):
                            v_num_pic += (
                                wv * self.scene.grid_v[ih + di, j + dj, kh + dk]
                            )
                            v_num_flip += wv * (
                                self.scene.grid_v[ih + di, j + dj, kh + dk]
                                - self.scene.grid_v_prev[ih + di, j + dj, kh + dk]
                            )
                            v_denom += wv

                        if (
                            0 <= ih + di < nx
                            and 0 <= jh + dj < ny
                            and 0 <= k + dk < nz + 1
                        ):
                            w_num_pic += (
                                ww * self.scene.grid_w[ih + di, jh + dj, k + dk]
                            )
                            w_num_flip += ww * (
                                self.scene.grid_w[ih + di, jh + dj, k + dk]
                                - self.scene.grid_w_prev[ih + di, jh + dj, k + dk]
                            )
                            w_denom += ww

            pic_vel[p][0] = u_num_pic / u_denom
            pic_vel[p][1] = v_num_pic / v_denom
            pic_vel[p][2] = w_num_pic / w_denom
            flip_delta_vel[p][0] = u_num_flip / u_denom
            flip_delta_vel[p][1] = v_num_flip / v_denom
            flip_delta_vel[p][2] = w_num_flip / w_denom

        for p in range(self.scene.num_particles):
            self.scene.particle_vel[p] = (
                self.flip_ratio * (self.scene.particle_vel[p] + flip_delta_vel[p])
                + (1 - self.flip_ratio) * pic_vel[p]
            )

    def handle_transfer(self, sdt, is_p2g):
        if is_p2g:
            self._p2g_transfer()
        else:
            self._g2p_transfer(self.pic_vel, self.flip_delta_vel)


@ti.data_oriented
class EulerianTransferStrategy(TransferStrategyBase):
    def __init__(self, scene: Scene):
        self.scene = scene

    @ti.kernel
    def _g2p_transfer(self, dt: ti.f32):  # type: ignore
        dx, dy, dz = self.scene.grid_dx, self.scene.grid_dy, self.scene.grid_dz
        nx, ny, nz = self.scene.grid_resolution
        sx, sy, sz = self.scene.grid_size
        radius = self.scene.particle_radius

        for p in range(self.scene.num_particles):
            pos = self.scene.particle_pos[p]

            x = pos[0]
            y = pos[1]
            z = pos[2]
            xh = x - dx / 2
            yh = y - dy / 2
            zh = z - dz / 2

            i = ti.cast(ti.floor(x / dx), ti.i32)
            j = ti.cast(ti.floor(y / dy), ti.i32)
            k = ti.cast(ti.floor(z / dz), ti.i32)
            ih = ti.cast(ti.floor(xh / dx), ti.i32)
            jh = ti.cast(ti.floor(yh / dy), ti.i32)
            kh = ti.cast(ti.floor(zh / dz), ti.i32)

            fx = x / dx - i
            fy = y / dy - j
            fz = z / dz - k
            fxh = xh / dx - ih
            fyh = yh / dy - jh
            fzh = zh / dz - kh

            u_num, v_num, w_num = 0.0, 0.0, 0.0
            u_denom, v_denom, w_denom = 1e-8, 1e-8, 1e-8

            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        wx = fx if di else (1 - fx)
                        wy = fy if dj else (1 - fy)
                        wz = fz if dk else (1 - fz)
                        wxh = fxh if di else (1 - fxh)
                        wyh = fyh if dj else (1 - fyh)
                        wzh = fzh if dk else (1 - fzh)

                        wu = wx * wyh * wzh
                        wv = wxh * wy * wzh
                        ww = wxh * wyh * wz

                        if (
                            0 <= i + di < nx + 1
                            and 0 <= jh + dj < ny
                            and 0 <= kh + dk < nz
                        ):
                            u_num += wu * self.scene.grid_u[i + di, jh + dj, kh + dk]
                            u_denom += wu

                        if (
                            0 <= ih + di < nx
                            and 0 <= j + dj < ny + 1
                            and 0 <= kh + dk < nz
                        ):
                            v_num += wv * self.scene.grid_v[ih + di, j + dj, kh + dk]
                            v_denom += wv

                        if (
                            0 <= ih + di < nx
                            and 0 <= jh + dj < ny
                            and 0 <= k + dk < nz + 1
                        ):
                            w_num += ww * self.scene.grid_w[ih + di, jh + dj, k + dk]
                            w_denom += ww

            vel = ti.Vector([u_num / u_denom, v_num / v_denom, w_num / w_denom])
            pos += dt * vel

            min_x = radius + dx
            max_x = sx - dx - radius
            min_y = radius + dy
            max_y = sy - dy - radius
            min_z = radius + dz
            max_z = sz - dz - radius
            pos[0] = ti.max(min_x, ti.min(max_x, pos[0]))
            pos[1] = ti.max(min_y, ti.min(max_y, pos[1]))
            pos[2] = ti.max(min_z, ti.min(max_z, pos[2]))

            cx = ti.cast(ti.floor(pos[0] / dx), ti.i32)
            cy = ti.cast(ti.floor(pos[1] / dy), ti.i32)
            cz = ti.cast(ti.floor(pos[2] / dz), ti.i32)
            cx = ti.max(0, ti.min(nx - 1, cx))
            cy = ti.max(0, ti.min(ny - 1, cy))
            cz = ti.max(0, ti.min(nz - 1, cz))
            if self.scene.grid_cell_type[cx, cy, cz] == CellType.CELL_SOLID.value:
                vel = self.scene.grid_solid_velocity[cx, cy, cz]

            self.scene.particle_vel[p] = vel
            self.scene.particle_pos[p] = pos

    def handle_transfer(self, sdt, is_p2g):
        if is_p2g:
            return
        self._g2p_transfer(sdt)


@ti.data_oriented
class ApicTransferStrategy(TransferStrategyBase):
    def __init__(self, scene: Scene):
        self.scene = scene
        self.new_vel = ti.Vector.field(3, dtype=ti.f32, shape=self.scene.num_particles)
        self.new_mat = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=self.scene.num_particles
        )

    @ti.kernel
    def _p2g_transfer(self):
        eps = 1e-8
        for I in ti.grouped(self.scene.grid_u_num):
            self.scene.grid_u_num[I] = 0.0
            self.scene.grid_u_denom[I] = eps
            self.scene.grid_u_prev[I] = self.scene.grid_u[I]
        for I in ti.grouped(self.scene.grid_v_num):
            self.scene.grid_v_num[I] = 0.0
            self.scene.grid_v_denom[I] = eps
            self.scene.grid_v_prev[I] = self.scene.grid_v[I]
        for I in ti.grouped(self.scene.grid_w_num):
            self.scene.grid_w_num[I] = 0.0
            self.scene.grid_w_denom[I] = eps
            self.scene.grid_w_prev[I] = self.scene.grid_w[I]

        dx, dy, dz = self.scene.grid_dx, self.scene.grid_dy, self.scene.grid_dz
        nx, ny, nz = self.scene.grid_resolution

        for p in range(self.scene.num_particles):
            pos = self.scene.particle_pos[p]
            vel = self.scene.particle_vel[p]
            mat = self.scene.particle_mat[p]

            x = pos[0]
            y = pos[1]
            z = pos[2]
            xh = pos[0] - dx / 2
            yh = pos[1] - dy / 2
            zh = pos[2] - dz / 2

            i0 = ti.cast(ti.floor(x / dx), ti.i32)
            j0 = ti.cast(ti.floor(y / dy), ti.i32)
            k0 = ti.cast(ti.floor(z / dz), ti.i32)
            ih0 = ti.cast(ti.floor(xh / dx), ti.i32)
            jh0 = ti.cast(ti.floor(yh / dy), ti.i32)
            kh0 = ti.cast(ti.floor(zh / dz), ti.i32)

            for di in range(-1, 3):
                for dj in range(-1, 3):
                    for dk in range(-1, 3):
                        i = i0 + di
                        j = j0 + dj
                        k = k0 + dk
                        ih = ih0 + di
                        jh = jh0 + dj
                        kh = kh0 + dk

                        faceu = ti.Vector([i * dx, (jh + 0.5) * dy, (kh + 0.5) * dz])
                        facev = ti.Vector([(ih + 0.5) * dx, j * dy, (kh + 0.5) * dz])
                        facew = ti.Vector([(ih + 0.5) * dx, (jh + 0.5) * dy, k * dz])

                        wx = bspline(x / dx - i)
                        wy = bspline(y / dy - j)
                        wz = bspline(z / dz - k)
                        wxh = bspline(x / dx - ih - 0.5)
                        wyh = bspline(y / dy - jh - 0.5)
                        wzh = bspline(z / dz - kh - 0.5)

                        wu = wx * wyh * wzh
                        wv = wxh * wy * wzh
                        ww = wxh * wyh * wz

                        velu = vel[0] + (mat @ (faceu - pos))[0]
                        velv = vel[1] + (mat @ (facev - pos))[1]
                        velw = vel[2] + (mat @ (facew - pos))[2]

                        if 0 <= i < nx + 1 and 0 <= jh < ny and 0 <= kh < nz:
                            ti.atomic_add(self.scene.grid_u_num[i, jh, kh], wu * velu)
                            ti.atomic_add(self.scene.grid_u_denom[i, jh, kh], wu)

                        if 0 <= ih < nx and 0 <= j < ny + 1 and 0 <= kh < nz:
                            ti.atomic_add(self.scene.grid_v_num[ih, j, kh], wv * velv)
                            ti.atomic_add(self.scene.grid_v_denom[ih, j, kh], wv)

                        if 0 <= ih < nx and 0 <= jh < ny and 0 <= k < nz + 1:
                            ti.atomic_add(self.scene.grid_w_num[ih, jh, k], ww * velw)
                            ti.atomic_add(self.scene.grid_w_denom[ih, jh, k], ww)

        for I in ti.grouped(self.scene.grid_u_num):
            self.scene.grid_u[I] = self.scene.grid_u_num[I] / self.scene.grid_u_denom[I]
        for I in ti.grouped(self.scene.grid_v_num):
            self.scene.grid_v[I] = self.scene.grid_v_num[I] / self.scene.grid_v_denom[I]
        for I in ti.grouped(self.scene.grid_w_num):
            self.scene.grid_w[I] = self.scene.grid_w_num[I] / self.scene.grid_w_denom[I]

        for I in ti.grouped(self.scene.grid_cell_type):
            i, j, k = I
            if self.scene.grid_cell_type[I] != CellType.CELL_SOLID.value:
                continue
            if (
                i > 0
                and self.scene.grid_cell_type[i - 1, j, k] == CellType.CELL_SOLID.value
            ):
                self.scene.grid_u[I] = self.scene.grid_u_prev[I]
            if (
                j > 0
                and self.scene.grid_cell_type[i, j - 1, k] == CellType.CELL_SOLID.value
            ):
                self.scene.grid_v[I] = self.scene.grid_v_prev[I]
            if (
                k > 0
                and self.scene.grid_cell_type[i, j, k - 1] == CellType.CELL_SOLID.value
            ):
                self.scene.grid_w[I] = self.scene.grid_w_prev[I]

        for I in ti.grouped(self.scene.grid_u):
            self.scene.grid_u_prev[I] = self.scene.grid_u[I]
        for I in ti.grouped(self.scene.grid_v):
            self.scene.grid_v_prev[I] = self.scene.grid_v[I]
        for I in ti.grouped(self.scene.grid_w):
            self.scene.grid_w_prev[I] = self.scene.grid_w[I]

    @ti.kernel
    def _g2p_transfer(self, new_vel: ti.template(), new_mat: ti.template()):  # type: ignore
        dx, dy, dz = self.scene.grid_dx, self.scene.grid_dy, self.scene.grid_dz
        nx, ny, nz = self.scene.grid_resolution

        for p in range(self.scene.num_particles):
            pos = self.scene.particle_pos[p]

            x = pos[0]
            y = pos[1]
            z = pos[2]
            xh = pos[0] - dx / 2
            yh = pos[1] - dy / 2
            zh = pos[2] - dz / 2

            i0 = ti.cast(ti.floor(x / dx), ti.i32)
            j0 = ti.cast(ti.floor(y / dy), ti.i32)
            k0 = ti.cast(ti.floor(z / dz), ti.i32)
            ih0 = ti.cast(ti.floor(xh / dx), ti.i32)
            jh0 = ti.cast(ti.floor(yh / dy), ti.i32)
            kh0 = ti.cast(ti.floor(zh / dz), ti.i32)

            u_num, v_num, w_num = 0.0, 0.0, 0.0
            u_denom, v_denom, w_denom = 1e-8, 1e-8, 1e-8
            mat11, mat12, mat13 = 0.0, 0.0, 0.0
            mat21, mat22, mat23 = 0.0, 0.0, 0.0
            mat31, mat32, mat33 = 0.0, 0.0, 0.0

            for di in range(-1, 3):
                for dj in range(-1, 3):
                    for dk in range(-1, 3):
                        i = i0 + di
                        j = j0 + dj
                        k = k0 + dk
                        ih = ih0 + di
                        jh = jh0 + dj
                        kh = kh0 + dk

                        faceu = ti.Vector([i * dx, (jh + 0.5) * dy, (kh + 0.5) * dz])
                        facev = ti.Vector([(ih + 0.5) * dx, j * dy, (kh + 0.5) * dz])
                        facew = ti.Vector([(ih + 0.5) * dx, (jh + 0.5) * dy, k * dz])

                        wx = bspline(x / dx - i)
                        wy = bspline(y / dy - j)
                        wz = bspline(z / dz - k)
                        wxh = bspline(x / dx - ih - 0.5)
                        wyh = bspline(y / dy - jh - 0.5)
                        wzh = bspline(z / dz - kh - 0.5)

                        wu = wx * wyh * wzh
                        wv = wxh * wy * wzh
                        ww = wxh * wyh * wz

                        if 0 <= i < nx + 1 and 0 <= jh < ny and 0 <= kh < nz:
                            delta_u = wu * self.scene.grid_u[i, jh, kh]
                            u_num += delta_u
                            u_denom += wu
                            rel_pos = faceu - pos
                            mat11 += delta_u * rel_pos[0]
                            mat12 += delta_u * rel_pos[1]
                            mat13 += delta_u * rel_pos[2]

                        if 0 <= ih < nx and 0 <= j < ny + 1 and 0 <= kh < nz:
                            delta_v = wv * self.scene.grid_v[ih, j, kh]
                            v_num += delta_v
                            v_denom += wv
                            rel_pos = facev - pos
                            mat21 += delta_v * rel_pos[0]
                            mat22 += delta_v * rel_pos[1]
                            mat23 += delta_v * rel_pos[2]

                        if 0 <= ih < nx and 0 <= jh < ny and 0 <= k < nz + 1:
                            delta_w = ww * self.scene.grid_w[ih, jh, k]
                            w_num += delta_w
                            w_denom += ww
                            rel_pos = facew - pos
                            mat31 += delta_w * rel_pos[0]
                            mat32 += delta_w * rel_pos[1]
                            mat33 += delta_w * rel_pos[2]

            new_vel[p][0] = u_num / u_denom
            new_vel[p][1] = v_num / v_denom
            new_vel[p][2] = w_num / w_denom

            new_mat[p][0, 0] = mat11 * 4 / u_denom / dx / dx
            new_mat[p][0, 1] = mat12 * 4 / u_denom / dx / dx
            new_mat[p][0, 2] = mat13 * 4 / u_denom / dx / dx
            new_mat[p][1, 0] = mat21 * 4 / v_denom / dx / dx
            new_mat[p][1, 1] = mat22 * 4 / v_denom / dx / dx
            new_mat[p][1, 2] = mat23 * 4 / v_denom / dx / dx
            new_mat[p][2, 0] = mat31 * 4 / w_denom / dx / dx
            new_mat[p][2, 1] = mat32 * 4 / w_denom / dx / dx
            new_mat[p][2, 2] = mat33 * 4 / w_denom / dx / dx

        for p in range(self.scene.num_particles):
            self.scene.particle_vel[p] = new_vel[p]
            self.scene.particle_mat[p] = new_mat[p]

    def handle_transfer(self, sdt, is_p2g):
        if is_p2g:
            self._p2g_transfer()
        else:
            self._g2p_transfer(self.new_vel, self.new_mat)
