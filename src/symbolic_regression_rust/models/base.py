"""Model interface shared by every solver.

A model maps parameters ``theta`` (p,) to predictions (m,) on fixed inputs ``x``
and provides an analytic Jacobian (m, p). Models that are conditionally linear in a
subset of parameters (``linear_idx``) can be used by VarPro:

    f(x; alpha, beta) = Phi(beta) @ alpha + c(beta)

The default ``separable`` / ``nonlinear_jacobian`` are derived from ``predict`` and
``jacobian``; specialised models may override them for speed.
"""

from __future__ import annotations

import numpy as np


class Model:
    def __init__(self, name: str, x: np.ndarray, param_names: list[str], linear_idx=()):
        self.name = name
        self.x = np.asarray(x, dtype=float)
        self.param_names = list(param_names)
        self.auto_linear = False  # linear set found by detect_linear (re-run near real starts)
        self.linear_idx = np.asarray(sorted(linear_idx), dtype=int)
        mask = np.ones(self.p, bool)
        mask[self.linear_idx] = False
        self.nonlinear_idx = np.flatnonzero(mask)

    # ---- shape ----
    @property
    def p(self) -> int:
        return len(self.param_names)

    @property
    def m(self) -> int:
        return self.x.shape[0]

    @property
    def n_linear(self) -> int:
        return self.linear_idx.size

    # ---- kernels (override) ----
    def predict(self, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # ---- separable structure (VarPro) ----
    def split(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return theta[self.linear_idx], theta[self.nonlinear_idx]

    def join(self, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        theta = np.empty(self.p)
        theta[self.linear_idx] = alpha
        theta[self.nonlinear_idx] = beta
        return theta

    def separable(self, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(Phi(beta), c(beta)) such that f = Phi @ alpha + c."""
        theta = self.join(np.zeros(self.n_linear), beta)
        return self.jacobian(theta)[:, self.linear_idx], self.predict(theta)

    def nonlinear_jacobian(self, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """df/dbeta at (alpha, beta), shape (m, n_nonlinear)."""
        return self.jacobian(self.join(alpha, beta))[:, self.nonlinear_idx]

    def detect_linear(
        self, center: np.ndarray | None = None, trials: int = 3, rtol: float = 1e-8
    ) -> list[int]:
        """Greedy set L of parameters in which f is jointly affine, tested numerically:
        f is affine in L  <=>  every column dF/dtheta_j (j in L) is unchanged when theta_L moves.

        Evaluated at ``trials`` random points near ``center`` (use a real starting point:
        at arbitrary points columns can underflow to 0, which proves nothing). Each column
        is compared relative to its own norm; zero or non-finite columns reject (conservative).
        """
        rng = np.random.default_rng(12345)
        c = np.ones(self.p) if center is None else np.asarray(center, float)
        c = np.where(c != 0, c, 1.0)
        base = [c * rng.uniform(0.8, 1.2, self.p) for _ in range(trials)]
        moved = [b * rng.uniform(0.5, 1.5, self.p) for b in base]
        with np.errstate(all="ignore"):
            J0 = [self.jacobian(b) for b in base]
        chosen: list[int] = []
        for j in range(self.p):
            cand = [*chosen, j]
            if all(
                self._unchanged(b, mv, J, cand, rtol)
                for b, mv, J in zip(base, moved, J0, strict=True)
            ):
                chosen.append(j)
        self.auto_linear = True
        return chosen

    def _unchanged(self, b, mv, J, cand, rtol) -> bool:
        th = b.copy()
        th[cand] = mv[cand]
        with np.errstate(all="ignore"):
            J1 = self.jacobian(th)
        for k in cand:
            u, v = J[:, k], J1[:, k]
            if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
                return False
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if min(nu, nv) < 1e-200 or np.linalg.norm(u - v) > rtol * max(nu, nv):
                return False
        return True

    def set_linear(self, idx) -> None:
        self.linear_idx = np.asarray(sorted(idx), dtype=int)
        mask = np.ones(self.p, bool)
        mask[self.linear_idx] = False
        self.nonlinear_idx = np.flatnonzero(mask)

    def describe(self) -> str:
        lin = [self.param_names[i] for i in self.linear_idx]
        return f"{self.name}: p={self.p}, m={self.m}, linear={lin or 'none'}"
