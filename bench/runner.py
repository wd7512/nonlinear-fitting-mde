"""Run one baseline on one problem and return a self-contained record."""

from __future__ import annotations

import numpy as np

from . import config
from .baselines import BASELINES, Job
from .evaluator import DeadlineExceeded, TimedEvaluator, clock_ns
from .problem import Problem


def run_job(
    problem: Problem,
    start: str,
    solver_name: str,
    deadline_ms: float = config.RUN_DEADLINE_MS,
    budgets_ms=(),
    full_history: bool = False,
) -> dict:
    t_job = clock_ns()  # includes per-fit setup, for end-to-end throughput
    fn = BASELINES[solver_name]
    theta0 = problem.starts[start].copy()
    if prepare := getattr(fn, "prepare", None):
        prepare(problem.y.size, problem.p)  # untimed one-off setup per shape (JIT compile)
    ev = TimedEvaluator(problem, deadline_ms, budgets_ms, full_history)
    job = Job(
        residual=ev.residual,
        p0=theta0.copy(),
        budget_ms=deadline_ms,
        linear_idx=tuple(int(i) for i in problem.model.linear_idx),
        separable=ev.separable,
    )
    status, error, theta = "returned", None, None
    with np.errstate(all="ignore"):
        loss0 = ev.loss_of(ev.residual(theta0))  # initial loss, tracked at its actual time
        try:  # only the baseline itself; harness errors propagate
            theta = np.asarray(fn(job), dtype=float)
        except DeadlineExceeded:
            status = "deadline"
        except Exception as exc:  # noqa: BLE001 - one failing baseline must not stop the run
            status, error = "failure", f"{type(exc).__name__}: {exc}"[:200]
        elapsed_ns = ev.elapsed_ns()  # the solver's time; the checks below are not charged to it
        if theta is not None:
            r = ev.weighted_residual(theta) if theta.shape == (problem.p,) else None
            if r is None or not np.isfinite(ev.loss_of(r)):
                status, error = "failure", "non-finite or misshapen result"
            elif elapsed_ns < ev.deadline_ns:
                ev.record(theta, r, elapsed_ns)
    small = problem.p <= config.MAX_THETA_P

    def _lst(th):
        return None if th is None else th.tolist()

    return {
        "type": "fit",
        "problem": problem.pid,
        "dataset": problem.dataset,
        "model": problem.model_name,
        "curve": problem.curve,
        "p": problem.p,
        "m": problem.m,
        "n_linear": problem.model.n_linear,
        "loss0": loss0 if np.isfinite(loss0) else None,  # loss at the start point
        "loss_norm": problem.loss_norm,
        "ref_loss": problem.ref_loss,
        "ref_mode": problem.ref_mode,
        "ref_params": _lst(problem.ref_params) if small else None,
        **{f"tag_{k}": v for k, v in problem.tags.items()},
        "start": start,
        "solver": solver_name,
        "solver_label": solver_name,
        "status": status,
        "error": error,
        "elapsed_ms": elapsed_ns / 1e6,
        "job_wall_ms": (clock_ns() - t_job) / 1e6,
        "best_loss": ev.best_loss,
        "best_theta": _lst(ev.best_theta) if small else None,
        "theta_at": [_lst(t) for t in ev.snapshot_thetas()] if small and budgets_ms else None,
        "nfev": ev.nfev,
        "fev_ms": ev.fev_ns / 1e6,
        "n_nonfinite": ev.n_nonfinite,
        "n_trials": ev.n_trials,
        # first loss + every improvement (or every finite trial with full_history), ms since start
        "hist_t_ms": [t / 1e6 for t in ev.hist_t_ns],
        "hist_loss": ev.hist_loss,
    }
