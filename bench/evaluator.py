"""Timed residual shared by every baseline.

Every evaluation goes through here, which checks the cooperative per-run deadline *before*
starting work, records every finite trial loss with the time at which it became available,
keeps the best parameters actually evaluated and counts non-finite trials.
"""

from __future__ import annotations

import time

import numpy as np

clock_ns = time.perf_counter_ns


class DeadlineExceeded(Exception):
    """Raised cooperatively when the per-run budget is exhausted."""


class TimedEvaluator:
    def __init__(self, problem, deadline_ms: float, snapshot_ms=()):
        self.model = problem.model
        self.w = problem.weights  # residual weights (1/sigma) or None
        self.y = problem.y
        self.m = self.y.size
        self.inv_scale = 1.0 / problem.denom
        self.deadline_ns = int(deadline_ms * 1e6)
        self.nfev = 0
        self.fev_ns = 0
        self.n_nonfinite = 0
        self.hist_t_ns: list[int] = []
        self.hist_loss: list[float] = []
        self.best_loss = np.inf
        self.best_theta: np.ndarray | None = None
        # best parameters available at each checkpoint (no late credit)
        self._snap_ns = [int(b * 1e6) for b in sorted(snapshot_ms)]
        self.snapshots: list[np.ndarray | None] = []
        self.t0 = clock_ns()

    def elapsed_ns(self) -> int:
        return clock_ns() - self.t0

    def loss_of(self, r: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            return float(np.dot(r, r)) / self.m * self.inv_scale

    def record(self, theta: np.ndarray, r: np.ndarray) -> None:
        loss = self.loss_of(r)
        if not np.isfinite(loss):
            self.n_nonfinite += 1
            return
        t = self.elapsed_ns()
        while len(self.snapshots) < len(self._snap_ns) and t > self._snap_ns[len(self.snapshots)]:
            self.snapshots.append(self.best_theta)
        self.hist_t_ns.append(t)
        self.hist_loss.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_theta = np.array(theta, copy=True)

    def residual(self, theta: np.ndarray) -> np.ndarray:
        if self.elapsed_ns() >= self.deadline_ns:
            raise DeadlineExceeded
        t = clock_ns()
        with np.errstate(all="ignore"):
            r = self.model.predict(theta) - self.y
            if self.w is not None:
                r = self.w * r
            self.record(theta, r)
        self.fev_ns += clock_ns() - t
        self.nfev += 1
        return r

    def snapshot_thetas(self) -> list[np.ndarray | None]:
        """Best theta at each checkpoint. A checkpoint no later record crossed saw no change
        after it, so the final best is exactly what was available then."""
        return self.snapshots + [self.best_theta] * (len(self._snap_ns) - len(self.snapshots))
