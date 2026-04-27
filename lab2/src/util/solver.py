import numpy as np
import scipy.sparse as sp
import pyamgx


class AmgxSolver:
    _initialized = False
    _live_solvers = 0

    @classmethod
    def _initialize_runtime(cls):
        if not cls._initialized:
            pyamgx.initialize()
            cls._initialized = True

    @classmethod
    def _finalize_runtime(cls):
        if cls._initialized and cls._live_solvers == 0:
            pyamgx.finalize()
            cls._initialized = False

    def __init__(self, max_iters: int, tolerance: float):
        AmgxSolver._initialize_runtime()
        AmgxSolver._live_solvers += 1
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
        self._destroyed = False

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
        if self._destroyed:
            return

        self.solver.destroy()
        self.x.destroy()
        self.b.destroy()
        self.A.destroy()
        self.rsrc.destroy()
        self.cfg.destroy()
        self._destroyed = True

        AmgxSolver._live_solvers = max(0, AmgxSolver._live_solvers - 1)
        AmgxSolver._finalize_runtime()
