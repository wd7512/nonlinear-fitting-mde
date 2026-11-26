"""Registered baselines. Add one by writing a function of a ``Job`` that returns parameters:

    @baseline
    def my_method(job):
        return scipy.optimize.something(job.residual, job.p0).x

The function name is the label in reports. The harness times the call, records the loss of
every ``job.residual`` evaluation, stops the solver at the deadline and scores the result.
Baselines use SciPy's own defaults (finite-difference Jacobian, default tolerances).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from scipy.optimize import least_squares


@dataclass(frozen=True)
class Job:
    residual: Callable[[Any], Any]  # p -> weighted residual vector
    p0: Any  # starting parameters
    x: Any  # (m, d) inputs
    y: Any  # (m,) observations
    weights: Any  # (m,) residual weights, or None
    expr: str | None  # the equation, when it has a symbolic form
    budget_ms: float


BASELINES: dict[str, Callable[[Job], Any]] = {}


def baseline(fn):
    BASELINES[fn.__name__] = fn
    return fn


@baseline
def scipy_trf(job):
    return least_squares(job.residual, job.p0, method="trf").x


@baseline
def scipy_lm(job):
    return least_squares(job.residual, job.p0, method="lm").x
