"""A: full-batch Adam on scaled parameters u = theta / param_scale."""

from __future__ import annotations

import numpy as np

from .. import config
from ..evaluator import NumericalFailure
from .base import Solver


class Adam(Solver):
    name = "adam"
    iteration_meaning = "one gradient step (1 fev + 1 jev)"

    def __init__(self, lr=None, beta1=None, beta2=None, eps=None):
        d = config.ADAM
        self.lr = lr or d["lr"]
        self.b1 = beta1 or d["beta1"]
        self.b2 = beta2 or d["beta2"]
        self.eps = eps or d["eps"]

    def run(self, theta0, ev):
        s = ev.param_scale
        u = theta0 / s
        m1 = np.zeros_like(u)
        m2 = np.zeros_like(u)
        t = 0
        while True:
            loss, g_theta = ev.loss_and_grad(u * s)
            if loss <= config.LOSS_FLOOR:
                return "converged"
            g = g_theta * s
            if not np.all(np.isfinite(g)):
                raise NumericalFailure("non-finite gradient")
            t += 1
            m1 = self.b1 * m1 + (1 - self.b1) * g
            m2 = self.b2 * m2 + (1 - self.b2) * g * g
            mhat = m1 / (1 - self.b1**t)
            vhat = m2 / (1 - self.b2**t)
            u = u - self.lr * mhat / (np.sqrt(vhat) + self.eps)
            ev.mark_iteration()
