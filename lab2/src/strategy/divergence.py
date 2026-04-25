from abc import ABC, abstractmethod
import atexit
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import taichi as ti

from constants import CellType
from scene import Scene


class DivergenceStrategyBase(ABC):
    @abstractmethod
    def handle_divergence(self, dt):
        """Project grid velocity to a divergence-free field."""
        return NotImplementedError


@ti.data_oriented
class GaussSeidel(DivergenceStrategyBase):
    def __init__(self, scene: Scene, num_iters, over_relaxation, compensate_drift):
        self.scene = scene
        self.num_iters = num_iters
        self.over_relaxation = over_relaxation
        self.compensate_drift = compensate_drift

    @ti.kernel
    def _gauss_seidel_mod2(self, color: ti.i32):  # type: ignore
        nx, ny, nz = self.scene.grid_resolution
        for I in ti.grouped(self.scene.grid_cell_type):
            x, y, z = I
            if (x + y + z) & 1 != color:
                continue
            if self.scene.grid_cell_type[I] != CellType.CELL_WATER.value:
                continue

            divergence = 0.0
            xp = x < nx - 1
            xm = x > 0
            yp = y < ny - 1
            ym = y > 0
            zp = z < nz - 1
            zm = z > 0

            if xp:
                if self.scene.grid_cell_type[x + 1, y, z] == CellType.CELL_SOLID.value:
                    xp = False
                divergence += self.scene.grid_u[x + 1, y, z]
            if xm:
                if self.scene.grid_cell_type[x - 1, y, z] == CellType.CELL_SOLID.value:
                    xm = False
                divergence -= self.scene.grid_u[x, y, z]
            if yp:
                if self.scene.grid_cell_type[x, y + 1, z] == CellType.CELL_SOLID.value:
                    yp = False
                divergence += self.scene.grid_v[x, y + 1, z]
            if ym:
                if self.scene.grid_cell_type[x, y - 1, z] == CellType.CELL_SOLID.value:
                    ym = False
                divergence -= self.scene.grid_v[x, y, z]
            if zp:
                if self.scene.grid_cell_type[x, y, z + 1] == CellType.CELL_SOLID.value:
                    zp = False
                divergence += self.scene.grid_w[x, y, z + 1]
            if zm:
                if self.scene.grid_cell_type[x, y, z - 1] == CellType.CELL_SOLID.value:
                    zm = False
                divergence -= self.scene.grid_w[x, y, z]

            num_cells = (
                ti.cast(xp, ti.i32)
                + ti.cast(xm, ti.i32)
                + ti.cast(yp, ti.i32)
                + ti.cast(ym, ti.i32)
                + ti.cast(zp, ti.i32)
                + ti.cast(zm, ti.i32)
            )
            if num_cells == 0:
                continue

            divergence = self.over_relaxation * divergence
            if self.compensate_drift:
                if self.scene.grid_density[I] > self.scene.avg_density[None]:
                    divergence -= (
                        self.scene.grid_density[I] - self.scene.avg_density[None]
                    )

            delta = divergence / num_cells
            if xp:
                self.scene.grid_u[x + 1, y, z] -= delta
            if xm:
                self.scene.grid_u[x, y, z] += delta
            if yp:
                self.scene.grid_v[x, y + 1, z] -= delta
            if ym:
                self.scene.grid_v[x, y, z] += delta
            if zp:
                self.scene.grid_w[x, y, z + 1] -= delta
            if zm:
                self.scene.grid_w[x, y, z] += delta

    def handle_divergence(self, dt):
        for _ in range(self.num_iters):
            self._gauss_seidel_mod2(0)
            self._gauss_seidel_mod2(1)


class _AmgxSolver:
    _initialized = False

    @classmethod
    def _initialize_runtime(cls, pyamgx_mod):
        if not cls._initialized:
            pyamgx_mod.initialize()
            atexit.register(pyamgx_mod.finalize)
            cls._initialized = True

    def __init__(self, max_iters: int, tolerance: float):
        import pyamgx

        self.pyamgx = pyamgx
        _AmgxSolver._initialize_runtime(self.pyamgx)
        cfg_dict = {
            "config_version": 2,
            "exception_handling": 1,
            "solver": {
                "scope": "main",
                "solver": "PCG",
                "preconditioner": {"scope": "amg", "solver": "NOSOLVER"},
                "max_iters": int(max_iters),
                "tolerance": float(tolerance),
                "monitor_residual": 0,
                "norm": "L2",
            },
        }
        self.cfg = pyamgx.Config().create_from_dict(cfg_dict)
        self.rsrc = pyamgx.Resources().create_simple(self.cfg)
        self.A = pyamgx.Matrix().create(self.rsrc)
        self.b = pyamgx.Vector().create(self.rsrc)
        self.x = pyamgx.Vector().create(self.rsrc)
        self.solver = pyamgx.Solver().create(self.rsrc, self.cfg)

    def solve(self, A_csr: sp.csr_matrix, b_np: np.ndarray) -> np.ndarray:
        A_csr = A_csr.astype(np.float64)
        b_np = b_np.astype(np.float64)
        self.A.upload_CSR(A_csr)
        self.b.upload(b_np)
        self.x.upload(np.zeros_like(b_np))
        self.solver.setup(self.A)
        self.solver.solve(self.b, self.x)
        out = np.zeros_like(b_np)
        self.x.download(out)
        return out

    def destroy(self):
        self.solver.destroy()
        self.x.destroy()
        self.b.destroy()
        self.A.destroy()
        self.rsrc.destroy()
        self.cfg.destroy()


class EulerianPressureProjection(DivergenceStrategyBase):
    def __init__(self, scene: Scene, max_iters, tolerance, rho=1.0, use_pyamgx=True):
        self.scene = scene
        self.max_iters = int(max_iters)
        self.tolerance = float(tolerance)
        self.rho = float(rho)
        self._amgx = None
        self._amgx_enabled = bool(use_pyamgx)
        if self._amgx_enabled:
            try:
                self._amgx = _AmgxSolver(self.max_iters, self.tolerance)
            except Exception:
                self._amgx_enabled = False

    def _cell_type(self, cell_type: np.ndarray, i: int, j: int, k: int) -> int:
        nx, ny, nz = cell_type.shape
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            return int(cell_type[i, j, k])
        return CellType.CELL_SOLID.value

    def _u_face_solid_vel(
        self, cell_type: np.ndarray, solid_vel: np.ndarray, i: int, j: int, k: int
    ) -> float:
        nx = cell_type.shape[0]
        val = 0.0
        count = 0.0
        if i > 0 and cell_type[i - 1, j, k] == CellType.CELL_SOLID.value:
            val += float(solid_vel[i - 1, j, k, 0])
            count += 1.0
        if i < nx and cell_type[i, j, k] == CellType.CELL_SOLID.value:
            val += float(solid_vel[i, j, k, 0])
            count += 1.0
        return val / count if count > 0.0 else 0.0

    def _v_face_solid_vel(
        self, cell_type: np.ndarray, solid_vel: np.ndarray, i: int, j: int, k: int
    ) -> float:
        ny = cell_type.shape[1]
        val = 0.0
        count = 0.0
        if j > 0 and cell_type[i, j - 1, k] == CellType.CELL_SOLID.value:
            val += float(solid_vel[i, j - 1, k, 1])
            count += 1.0
        if j < ny and cell_type[i, j, k] == CellType.CELL_SOLID.value:
            val += float(solid_vel[i, j, k, 1])
            count += 1.0
        return val / count if count > 0.0 else 0.0

    def _w_face_solid_vel(
        self, cell_type: np.ndarray, solid_vel: np.ndarray, i: int, j: int, k: int
    ) -> float:
        nz = cell_type.shape[2]
        val = 0.0
        count = 0.0
        if k > 0 and cell_type[i, j, k - 1] == CellType.CELL_SOLID.value:
            val += float(solid_vel[i, j, k - 1, 2])
            count += 1.0
        if k < nz and cell_type[i, j, k] == CellType.CELL_SOLID.value:
            val += float(solid_vel[i, j, k, 2])
            count += 1.0
        return val / count if count > 0.0 else 0.0

    def _build_linear_system(
        self,
        dt: float,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        cell_type: np.ndarray,
        solid_vel: np.ndarray,
    ):
        nx, ny, nz = self.scene.grid_resolution
        h = float(self.scene.grid_dx)
        inv_h2 = 1.0 / (h * h)

        fluid = cell_type == CellType.CELL_WATER.value
        index = -np.ones((nx, ny, nz), dtype=np.int32)
        fluid_coords = np.argwhere(fluid)
        for row_id, (i, j, k) in enumerate(fluid_coords):
            index[i, j, k] = row_id

        n = int(fluid_coords.shape[0])
        if n == 0:
            return None, None, index

        rows = []
        cols = []
        vals = []
        b = np.zeros((n,), dtype=np.float64)

        for i, j, k in fluid_coords:
            row = int(index[i, j, k])
            diag = 0.0

            t_xm = self._cell_type(cell_type, i - 1, j, k)
            t_xp = self._cell_type(cell_type, i + 1, j, k)
            t_ym = self._cell_type(cell_type, i, j - 1, k)
            t_yp = self._cell_type(cell_type, i, j + 1, k)
            t_zm = self._cell_type(cell_type, i, j, k - 1)
            t_zp = self._cell_type(cell_type, i, j, k + 1)

            u_left = (
                self._u_face_solid_vel(cell_type, solid_vel, i, j, k)
                if t_xm == CellType.CELL_SOLID.value
                else float(u[i, j, k])
            )
            u_right = (
                self._u_face_solid_vel(cell_type, solid_vel, i + 1, j, k)
                if t_xp == CellType.CELL_SOLID.value
                else float(u[i + 1, j, k])
            )
            v_down = (
                self._v_face_solid_vel(cell_type, solid_vel, i, j, k)
                if t_ym == CellType.CELL_SOLID.value
                else float(v[i, j, k])
            )
            v_up = (
                self._v_face_solid_vel(cell_type, solid_vel, i, j + 1, k)
                if t_yp == CellType.CELL_SOLID.value
                else float(v[i, j + 1, k])
            )
            w_back = (
                self._w_face_solid_vel(cell_type, solid_vel, i, j, k)
                if t_zm == CellType.CELL_SOLID.value
                else float(w[i, j, k])
            )
            w_front = (
                self._w_face_solid_vel(cell_type, solid_vel, i, j, k + 1)
                if t_zp == CellType.CELL_SOLID.value
                else float(w[i, j, k + 1])
            )

            div = (u_right - u_left + v_up - v_down + w_front - w_back) / h
            b[row] = (self.rho / dt) * div

            for ni, nj, nk, nt in (
                (i - 1, j, k, t_xm),
                (i + 1, j, k, t_xp),
                (i, j - 1, k, t_ym),
                (i, j + 1, k, t_yp),
                (i, j, k - 1, t_zm),
                (i, j, k + 1, t_zp),
            ):
                if nt == CellType.CELL_WATER.value:
                    diag += inv_h2
                    rows.append(row)
                    cols.append(int(index[ni, nj, nk]))
                    vals.append(-inv_h2)
                elif nt == CellType.CELL_AIR.value:
                    diag += inv_h2

            if diag <= 0.0:
                diag = 1.0
                b[row] = 0.0
            rows.append(row)
            cols.append(row)
            vals.append(diag)

        A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
        return A, b, index

    def _solve_pressure(self, A: sp.csr_matrix, b: np.ndarray) -> np.ndarray:
        if self._amgx_enabled and self._amgx is not None:
            try:
                return self._amgx.solve(A, b)
            except Exception:
                self._amgx_enabled = False
                self._amgx = None

        x, info = spla.cg(A, b, maxiter=self.max_iters, atol=0.0, rtol=self.tolerance)
        if info != 0:
            x = spla.spsolve(A, b)
        return np.asarray(x, dtype=np.float64)

    def _project_velocity(
        self,
        dt: float,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        cell_type: np.ndarray,
        solid_vel: np.ndarray,
        pressure: np.ndarray,
    ):
        nx, ny, nz = self.scene.grid_resolution
        h = float(self.scene.grid_dx)
        scale = dt / self.rho

        u_new = u.copy()
        v_new = v.copy()
        w_new = w.copy()

        for i in range(nx + 1):
            for j in range(ny):
                for k in range(nz):
                    left_t = self._cell_type(cell_type, i - 1, j, k)
                    right_t = self._cell_type(cell_type, i, j, k)
                    solid = (
                        left_t == CellType.CELL_SOLID.value
                        or right_t == CellType.CELL_SOLID.value
                    )
                    if solid:
                        u_new[i, j, k] = self._u_face_solid_vel(
                            cell_type, solid_vel, i, j, k
                        )
                    else:
                        p_left = (
                            pressure[i - 1, j, k]
                            if left_t == CellType.CELL_WATER.value
                            else 0.0
                        )
                        p_right = (
                            pressure[i, j, k]
                            if right_t == CellType.CELL_WATER.value
                            else 0.0
                        )
                        u_new[i, j, k] -= scale * (p_right - p_left) / h

        for i in range(nx):
            for j in range(ny + 1):
                for k in range(nz):
                    down_t = self._cell_type(cell_type, i, j - 1, k)
                    up_t = self._cell_type(cell_type, i, j, k)
                    solid = (
                        down_t == CellType.CELL_SOLID.value
                        or up_t == CellType.CELL_SOLID.value
                    )
                    if solid:
                        v_new[i, j, k] = self._v_face_solid_vel(
                            cell_type, solid_vel, i, j, k
                        )
                    else:
                        p_down = (
                            pressure[i, j - 1, k]
                            if down_t == CellType.CELL_WATER.value
                            else 0.0
                        )
                        p_up = (
                            pressure[i, j, k]
                            if up_t == CellType.CELL_WATER.value
                            else 0.0
                        )
                        v_new[i, j, k] -= scale * (p_up - p_down) / h

        for i in range(nx):
            for j in range(ny):
                for k in range(nz + 1):
                    back_t = self._cell_type(cell_type, i, j, k - 1)
                    front_t = self._cell_type(cell_type, i, j, k)
                    solid = (
                        back_t == CellType.CELL_SOLID.value
                        or front_t == CellType.CELL_SOLID.value
                    )
                    if solid:
                        w_new[i, j, k] = self._w_face_solid_vel(
                            cell_type, solid_vel, i, j, k
                        )
                    else:
                        p_back = (
                            pressure[i, j, k - 1]
                            if back_t == CellType.CELL_WATER.value
                            else 0.0
                        )
                        p_front = (
                            pressure[i, j, k]
                            if front_t == CellType.CELL_WATER.value
                            else 0.0
                        )
                        w_new[i, j, k] -= scale * (p_front - p_back) / h

        self.scene.grid_u_prev.from_numpy(u.astype(np.float32))
        self.scene.grid_v_prev.from_numpy(v.astype(np.float32))
        self.scene.grid_w_prev.from_numpy(w.astype(np.float32))
        self.scene.grid_u.from_numpy(u_new.astype(np.float32))
        self.scene.grid_v.from_numpy(v_new.astype(np.float32))
        self.scene.grid_w.from_numpy(w_new.astype(np.float32))

    def handle_divergence(self, dt):
        if dt <= 0.0:
            return

        u = self.scene.grid_u.to_numpy()
        v = self.scene.grid_v.to_numpy()
        w = self.scene.grid_w.to_numpy()
        cell_type = self.scene.grid_cell_type.to_numpy()
        solid_vel = self.scene.grid_solid_velocity.to_numpy()

        A, b, index = self._build_linear_system(dt, u, v, w, cell_type, solid_vel)
        nx, ny, nz = self.scene.grid_resolution
        pressure = np.zeros((nx, ny, nz), dtype=np.float64)
        if A is not None:
            x = self._solve_pressure(A, b)
            fluid_mask = index >= 0
            pressure[fluid_mask] = x[index[fluid_mask]]

        self._project_velocity(dt, u, v, w, cell_type, solid_vel, pressure)
