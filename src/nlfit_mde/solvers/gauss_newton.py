"""C: Gauss-Newton with SVD least-squares steps and step-halving line search.

Non-finite or non-decreasing trials are rejected (halved); if no decrease is found
after ``max_backtracks`` halvings the run stops as "stalled".
"""

from __future__ import annotations

import numpy as np

from .. import config
from ..evaluator import NumericalFailure
from .base import Solver


class GaussNewton(Solver):
    name = "gauss_newton"
    iteration_meaning = "one GN step (1 jev + SVD solve + >=1 trial fev)"

    def run(self, theta0, ev):
        max_bt = config.GAUSS_NEWTON["max_backtracks"]
        xtol = config.GAUSS_NEWTON["xtol"]
        theta = theta0.copy()
        r = ev.residual(theta)
        loss = ev.loss_of(r)
        if not np.isfinite(loss):
            raise NumericalFailure("non-finite initial residual")
        while True:
            if loss <= config.LOSS_FLOOR:
                return "converged"
            J = ev.jacobian(theta)
            if not np.all(np.isfinite(J)):
                raise NumericalFailure("non-finite Jacobian")
            step = np.linalg.lstsq(J, -r, rcond=None)[0]
            alpha = 1.0
            for _ in range(max_bt):
                trial = theta + alpha * step
                r_t = ev.residual(trial)
                loss_t = ev.loss_of(r_t)
                if np.isfinite(loss_t) and loss_t < loss:
                    break
                alpha *= 0.5
            else:
                return "stalled"
            dx = np.linalg.norm(alpha * step)
            theta, r, loss = trial, r_t, loss_t
            ev.mark_iteration()
            if dx <= xtol * (xtol + np.linalg.norm(theta)):
                return "converged"
