"""Timed residual shared by every baseline.

Every evaluation goes through here, which checks the cooperative per-run deadline *before*
starting work, records the loss of finite trials (the first and every improvement, or all of
them) with the time at which each became available, keeps the best parameters actually
evaluated and counts non-finite trials.
"""

from __future__ import annotations

import time

import numpy as np

clock_ns = time.perf_counter_ns


class DeadlineExceeded(BaseException):
    """Raised cooperatively when the per-run budget is exhausted.

    A BaseException, so a baseline's own ``except Exception`` cannot swallow it.
    """


class TimedEvaluator:
    """Callers run it under ``np.errstate(all="ignore")``; it does not set that itself."""

    def __init__(self, problem, deadline_ms: float, snapshot_ms=(), full_history=False):
        self.model = problem.model
        self.lin = problem.model.linear_idx
        self.w = problem.weights  # residual weights (1/sigma) or None
        self.y = problem.y
        self.k = 1.0 / (self.y.size * problem.denom)  # sum of squares -> normalised loss
        self.deadline_ns = int(deadline_ms * 1e6)
        self.full_history = full_history  # else keep only the first trial and improvements
        self.nfev = 0
        self.fev_ns = 0
        self.n_nonfinite = 0
        self.n_trials = 0
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
        return float(np.dot(r, r)) * self.k

    def weighted_residual(self, theta: np.ndarray) -> np.ndarray:
        r = self.model.predict(theta) - self.y
        return r if self.w is None else self.w * r

    def record(self, theta: np.ndarray, r: np.ndarray, t: int) -> None:
        """Record a trial whose residual ``r`` became available at ``t`` ns after the start."""
        loss = self.loss_of(r)
        if not np.isfinite(loss):
            self.n_nonfinite += 1
            return
        self.n_trials += 1
        while len(self.snapshots) < len(self._snap_ns) and t > self._snap_ns[len(self.snapshots)]:
            self.snapshots.append(self.best_theta)
        if self.full_history or loss < self.best_loss:
            self.hist_t_ns.append(t)
            self.hist_loss.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_theta = np.array(theta, copy=True)

    def residual(self, theta: np.ndarray) -> np.ndarray:
        t1 = clock_ns()
        if t1 - self.t0 >= self.deadline_ns:
            raise DeadlineExceeded
        r = self.weighted_residual(theta)
        t2 = clock_ns()
        self.fev_ns += t2 - t1
        self.nfev += 1
        self.record(theta, r, t2 - self.t0)
        return r

    def separable(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(r0, Phi)`` with residual(theta) = r0(beta) + Phi(beta) @ alpha, where alpha =
        theta[linear_idx] and beta is the rest: the separable form, for variable projection.

        Timed and deadline-checked like ``residual``, but not recorded: it has no loss.
        """
        t1 = clock_ns()
        if t1 - self.t0 >= self.deadline_ns:
            raise DeadlineExceeded
        th = np.array(theta, dtype=float)
        th[self.lin] = 0.0
        r0, phi = self.model.predict(th) - self.y, self.model.jacobian(th)[:, self.lin]
        if self.w is not None:
            r0, phi = self.w * r0, self.w[:, None] * phi
        self.fev_ns += clock_ns() - t1
        self.nfev += 1
        return r0, phi

    def snapshot_thetas(self) -> list[np.ndarray | None]:
        """Best theta at each checkpoint. A checkpoint no later record crossed saw no change
        after it, so the final best is exactly what was available then."""
        return self.snapshots + [self.best_theta] * (len(self._snap_ns) - len(self.snapshots))
