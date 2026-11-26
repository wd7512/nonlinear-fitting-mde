"""Solver interface.

A solver drives its own loop, calls the shared ``TimedEvaluator`` for every
residual / Jacobian, calls ``ev.mark_iteration()`` once per (solver-defined)
iteration and returns a termination reason. Deadline and numerical failures are
signalled with the evaluator's exceptions and handled by the runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..evaluator import TimedEvaluator


class Solver(ABC):
    name: str = ""
    #: what one recorded "iteration" means for this solver (reported, not compared)
    iteration_meaning: str = ""

    def label(self, model) -> str:
        """Label for the results (VarPro reports when it fell back to LM)."""
        return self.name

    @abstractmethod
    def run(self, theta0: np.ndarray, ev: TimedEvaluator) -> str:
        """Optimise from ``theta0``; return a termination reason string."""
