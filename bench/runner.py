"""Run one baseline on one problem and return a self-contained record."""

from __future__ import annotations

import numpy as np

from . import config
from .baselines import BASELINES, Job
from .evaluator import DeadlineExceeded, TimedEvaluator, clock_ns
from .problem import Problem

MAX_THETA_P = 128  # store per-checkpoint parameters only for models up to this size


def compress_history(t_ns: list[int], loss: list[float]) -> tuple[list[float], list[float]]:
    """Keep the first entry and every strict improvement: exactly what best-so-far scoring needs."""
    ts, ls, best = [], [], np.inf
    for t, v in zip(t_ns, loss, strict=True):
        if not ts or v < best:
            ts.append(t / 1e6)
            ls.append(v)
            best = min(best, v)
    return ts, ls


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
    with np.errstate(all="ignore"):
        loss0 = float(problem.loss_at(theta0))  # untimed; for the reduction success test
    ev = TimedEvaluator(problem, deadline_ms, snapshot_ms=budgets_ms)
    status, error = "returned", None
    try:
        ev.residual(theta0)  # initial loss, tracked at its actual time
        job = Job(
            ev.residual,
            theta0.copy(),
            problem.model.x,
            problem.y,
            problem.weights,
            problem.model.expression,
            deadline_ms,
        )
        with np.errstate(all="ignore"):
            theta = np.asarray(fn(job), dtype=float)
            if theta.shape != (problem.p,) or not np.isfinite(problem.loss_at(theta)):
                status, error = "failure", "non-finite or misshapen result"
            elif ev.elapsed_ns() < ev.deadline_ns:
                r = problem.model.predict(theta) - problem.y
                ev.record(theta, r if problem.weights is None else problem.weights * r)
    except DeadlineExceeded:
        status = "deadline"
    except Exception as exc:  # noqa: BLE001 - one failing baseline must not stop the run
        status, error = "failure", f"{type(exc).__name__}: {exc}"[:200]
    elapsed_ns = ev.elapsed_ns()
    if full_history:
        hist_t, hist_l = [t / 1e6 for t in ev.hist_t_ns], list(ev.hist_loss)
    else:
        hist_t, hist_l = compress_history(ev.hist_t_ns, ev.hist_loss)
    small = problem.p <= MAX_THETA_P

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
        "n_trials": len(ev.hist_loss),
        # first loss + every improvement (or every finite trial with full_history), ms since start
        "hist_t_ms": hist_t,
        "hist_loss": hist_l,
    }
