"""B: L-BFGS (SciPy L-BFGS-B, no bounds) on scaled parameters u = theta / param_scale."""

from __future__ import annotations

from scipy.optimize import minimize

from .. import config
from .base import Solver


class LBFGS(Solver):
    name = "lbfgs"
    iteration_meaning = "one L-BFGS iteration (SciPy callback; may include several fevs)"

    def run(self, theta0, ev):
        s = ev.param_scale

        def fun(u):
            loss, g = ev.loss_and_grad(u * s)
            return loss, g * s

        res = minimize(
            fun,
            theta0 / s,
            jac=True,
            method="L-BFGS-B",
            callback=lambda *_: ev.mark_iteration(),
            options={
                "maxcor": config.LBFGS["maxcor"],
                "ftol": config.LBFGS["ftol"],
                "gtol": config.LBFGS["gtol"],
                "maxiter": 10**9,
                "maxfun": 10**9,
            },
        )
        return "converged" if res.success else f"stopped: {res.message}"
