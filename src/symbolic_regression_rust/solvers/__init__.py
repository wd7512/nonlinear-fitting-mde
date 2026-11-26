"""Solver registry (IDs A-F in the original benchmark design)."""

from __future__ import annotations

from .adam import Adam
from .base import Solver
from .gauss_newton import GaussNewton
from .lbfgs import LBFGS
from .least_squares import LevenbergMarquardt, TrustRegionReflective
from .varpro import VarProLM

SOLVERS: dict[str, Solver] = {
    s.name: s
    for s in (
        Adam(),
        LBFGS(),
        GaussNewton(),
        LevenbergMarquardt(),
        TrustRegionReflective(),
        VarProLM(),
    )
}

__all__ = ["SOLVERS", "Solver"]
