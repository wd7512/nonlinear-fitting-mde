"""``nlfit-mde inspect``: validate a config and show what would be fitted (no timing)."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .datasets import build_problems
from .experiment import ExperimentConfig


def inspect(cfg: ExperimentConfig) -> None:
    problems = build_problems(cfg, np.random.default_rng(cfg.seed))
    n_jobs = sum(len(p.starts) for p in problems) * len(cfg.solvers)
    print(
        f"experiment {cfg.name!r}: {len(problems)} problems, {n_jobs} fits "
        f"({len(cfg.solvers)} solvers), budgets {cfg.budgets_ms} ms, targets {cfg.targets}"
    )
    est = n_jobs * cfg.run_deadline_ms / 1e3
    print(
        f"worst-case solver time {est:.1f} s; worker deadline {cfg.worker_deadline_s} s, "
        f"watchdog {cfg.watchdog_s} s"
        + (
            "  ** may truncate: raise limits or max_curves **"
            if est > cfg.worker_deadline_s
            else ""
        )
    )
    print()
    groups = Counter((p.dataset, p.model_name) for p in problems)
    seen = set()
    for pb in problems:
        key = (pb.dataset, pb.model_name)
        if key in seen:
            continue
        seen.add(key)
        print(f"[{pb.dataset}] {pb.model.describe()}")
        print(
            f"    curves: {groups[key]}   m (first curve): {pb.m}   weights: {'1/sigma' if pb.weights is not None else 'none'}"
        )
        ref = "unknown" if pb.ref_loss is None else f"{pb.ref_loss:.3e}"
        print(f"    reference: {pb.ref_mode} (loss {ref})")
        for name, th in pb.starts.items():
            loss = pb.loss_at(th)
            print(
                f"    start {name:>8}: loss {loss:.3e}  theta[:6] = {np.array2string(th[:6], precision=4)}"
            )
