"""Timed, instrumented evaluator shared by every solver.

Every residual / Jacobian evaluation goes through here, which:
  * checks the cooperative per-run deadline *before* starting work,
  * times evaluations with a monotonic high-resolution clock,
  * records every finite trial loss (accepted or rejected) with the time at which
    it became available, and keeps the best parameters actually evaluated,
  * counts non-finite trials.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

clock_ns = time.perf_counter_ns


class DeadlineExceeded(Exception):
    """Raised cooperatively when the per-run budget is exhausted."""


class NumericalFailure(Exception):
    """Raised when a solver cannot continue because of non-finite numbers."""


@dataclass
class Counters:
    nfev: int = 0
    njev: int = 0
    fev_ns: int = 0
    jev_ns: int = 0
    n_nonfinite: int = 0
    iter_ns: list[int] = field(default_factory=list)  # duration of each completed iteration


class TimedEvaluator:
    def __init__(self, problem, deadline_ms: float, snapshot_ms=()):
        self.problem = problem
        self.model = problem.model
        self.param_scale = problem.param_scale
        self.w = problem.weights  # residual weights (1/sigma) or None
        self.y = problem.y
        self.y_eff = self.y if self.w is None else self.w * self.y
        self.m = self.y.size
        self.inv_scale = 1.0 / problem.denom
        self.deadline_ns = int(deadline_ms * 1e6)
        self.counters = Counters()
        self.hist_t_ns: list[int] = []
        self.hist_loss: list[float] = []
        self.best_loss = np.inf
        self.best_theta: np.ndarray | None = None
        # best parameters available at each checkpoint (no late credit)
        self._snap_ns = [int(b * 1e6) for b in sorted(snapshot_ms)]
        self.snapshots: list[np.ndarray | None] = []
        self.t0 = clock_ns()
        self._last_iter_mark = self.t0

    # ---- time ----
    def elapsed_ns(self) -> int:
        return clock_ns() - self.t0

    def check_deadline(self) -> None:
        if clock_ns() - self.t0 >= self.deadline_ns:
            raise DeadlineExceeded

    def start_iterations(self) -> None:
        self._last_iter_mark = clock_ns()

    def mark_iteration(self) -> None:
        now = clock_ns()
        self.counters.iter_ns.append(now - self._last_iter_mark)
        self._last_iter_mark = now

    @contextmanager
    def timed(self, kind: str):
        """Deadline check + timing + counting for one evaluation ('fev' or 'jev')."""
        self.check_deadline()
        t = clock_ns()
        with np.errstate(all="ignore"):
            yield
        dt = clock_ns() - t
        c = self.counters
        if kind == "fev":
            c.nfev += 1
            c.fev_ns += dt
        else:
            c.njev += 1
            c.jev_ns += dt

    # ---- loss bookkeeping ----
    def loss_of(self, r: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            return float(np.dot(r, r)) / self.m * self.inv_scale

    def record(self, theta: np.ndarray, r: np.ndarray) -> float:
        """Record the loss of a full-parameter residual. Returns the loss (may be inf/nan)."""
        loss = self.loss_of(r)
        if np.isfinite(loss):
            t = clock_ns() - self.t0
            while (
                len(self.snapshots) < len(self._snap_ns) and t > self._snap_ns[len(self.snapshots)]
            ):
                self.snapshots.append(self.best_theta)
            self.hist_t_ns.append(t)
            self.hist_loss.append(loss)
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_theta = np.array(theta, copy=True)
        else:
            self.counters.n_nonfinite += 1
        return loss

    # ---- shared kernels ----
    def residual(self, theta: np.ndarray) -> np.ndarray:
        with self.timed("fev"):
            r = self.model.predict(theta) - self.y
            if self.w is not None:
                r = self.w * r
            self.record(theta, r)
        return r

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        with self.timed("jev"):
            J = self.model.jacobian(theta)
            if self.w is not None:
                J = self.w[:, None] * J
        return J

    def loss_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Normalised MSE and its gradient (for first-order / quasi-Newton methods)."""
        r = self.residual(theta)
        loss = self.loss_of(r)
        if not np.isfinite(loss):
            return np.inf, np.full(theta.size, np.nan)
        J = self.jacobian(theta)
        with np.errstate(all="ignore"):
            g = (2.0 / self.m * self.inv_scale) * (J.T @ r)
        return loss, g

    def snapshot_thetas(self) -> list[np.ndarray | None]:
        """Best theta at each checkpoint. A checkpoint no later record crossed saw no change
        after it, so the final best is exactly what was available then."""
        return self.snapshots + [self.best_theta] * (len(self._snap_ns) - len(self.snapshots))
