"""D / E: SciPy ``least_squares`` with method 'lm' (MINPACK) or 'trf' (unbounded).

SciPy does not expose per-iteration timing for these methods; one iteration is
taken to be the interval between successive Jacobian evaluations.
"""

from __future__ import annotations

from scipy.optimize import least_squares

from .. import config
from .base import Solver


def run_least_squares(fun, jac_eval, x0, ev, method: str) -> str:
    def jac(x):
        ev.mark_iteration()
        return jac_eval(x)

    res = least_squares(fun, x0, jac=jac, method=method, max_nfev=10**6, **config.LEAST_SQUARES_TOL)
    return "converged" if res.status > 0 else f"stopped: {res.message}"


class LevenbergMarquardt(Solver):
    name = "lm"
    iteration_meaning = "interval between Jacobian evaluations (MINPACK)"

    def run(self, theta0, ev):
        return run_least_squares(ev.residual, ev.jacobian, theta0, ev, "lm")


class TrustRegionReflective(Solver):
    name = "trf"
    iteration_meaning = "interval between Jacobian evaluations"

    def run(self, theta0, ev):
        return run_least_squares(ev.residual, ev.jacobian, theta0, ev, "trf")
