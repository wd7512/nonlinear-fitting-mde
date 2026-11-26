"""F: Variable projection + LM (Kaufman Jacobian), for any separable model.

For f = Phi(beta) @ alpha + c(beta) the linear parameters are eliminated: alpha(beta)
solves the (weighted) linear least-squares problem via a rank-revealing SVD (no
explicit inverse). The outer LM sees the full residual r = W (Phi alpha + c - y),
i.e. the original residual at the reconstructed theta = [alpha, beta], so it is
scored exactly like every other solver; the inner solve counts as fev time.

Kaufman's approximation to the projected Jacobian:  J = (I - U U^T) W df/dbeta,
with U spanning range(W Phi). Models without linear parameters fall back to LM;
fully linear models are solved by the single inner solve.
"""

from __future__ import annotations

import numpy as np

from .base import Solver
from .least_squares import run_least_squares

_EPS = np.finfo(float).eps


class VarProLM(Solver):
    name = "varpro_lm"
    iteration_meaning = "interval between Jacobian evaluations (outer MINPACK LM)"

    def label(self, model) -> str:
        if model.n_linear == 0:
            return f"{self.name} (LM fallback)"
        if model.n_linear == model.p:
            return f"{self.name} (linear solve)"
        return self.name

    def run(self, theta0, ev):
        model = ev.model
        if model.n_linear == 0:
            return run_least_squares(ev.residual, ev.jacobian, theta0, ev, "lm")

        y, w = ev.y, ev.w
        cache: dict = {}

        def inner(beta):
            key = beta.tobytes()
            if cache.get("key") == key:
                return cache
            with ev.timed("fev"):
                phi, c = model.separable(beta)
                if w is not None:
                    phi, c = w[:, None] * phi, w * c
                if not (np.all(np.isfinite(phi)) and np.all(np.isfinite(c))):
                    ev.counters.n_nonfinite += 1  # overflowing trial: reject, don't crash SVD
                    cache.clear()
                    cache.update(key=key, r=np.full(y.size, np.inf))
                    return cache
                target = ev.y_eff - c
                U, sv, Vt = np.linalg.svd(phi, full_matrices=False)
                tol = (sv[0] if sv.size else 0.0) * max(phi.shape) * _EPS
                rank = int(np.sum(sv > tol))
                U = U[:, :rank]
                alpha = Vt[:rank].T @ ((U.T @ target) / sv[:rank])
                r = phi @ alpha - target
                ev.record(model.join(alpha, beta), r)
            cache.update(key=key, U=U, alpha=alpha, r=r)
            return cache

        def residual(beta):
            if not np.all(np.isfinite(beta)):
                return np.full(y.size, np.inf)
            return inner(beta)["r"]

        def jacobian(beta):
            c = inner(beta)
            if "U" not in c:
                return np.full((y.size, beta.size), np.nan)
            with ev.timed("jev"):
                D = model.nonlinear_jacobian(c["alpha"], beta)
                if w is not None:
                    D = w[:, None] * D
                J = D - c["U"] @ (c["U"].T @ D)
            return J

        _, beta0 = model.split(theta0)
        if beta0.size == 0:  # fully linear model: one inner solve is the exact answer
            inner(beta0)
            return "converged"
        return run_least_squares(residual, jacobian, beta0.copy(), ev, "lm")
