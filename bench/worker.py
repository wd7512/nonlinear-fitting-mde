"""Benchmark worker: one process runs every fit and streams JSONL records.

Run under the parent watchdog (``bench.cli``), which sets single-thread BLAS
environment variables before this process imports numpy. Records are flushed one
per line so partial results survive if the watchdog kills the worker.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

T_START = time.perf_counter()

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402

from . import config  # noqa: E402
from .experiment import load_config  # noqa: E402
from .runner import run_job  # noqa: E402
from .suites import build_problems  # noqa: E402
from .suites.synthetic import make_warmup_fixture  # noqa: E402


def _prepare(problems) -> None:
    """Evaluate every model once (first-call costs, shape checks) outside the timed loop."""
    for pb in problems:
        theta = next(iter(pb.starts.values()))
        f = pb.model.predict(theta)
        if f.shape != (pb.m,):
            raise ValueError(f"{pb.pid}: bad prediction shape {f.shape}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job")
    args = ap.parse_args(argv)
    threadpool_limits(1)  # belt and braces on top of the env vars
    t_import = time.perf_counter() - T_START

    with open(args.out, "w") as out:

        def emit(rec: dict) -> None:
            out.write(json.dumps(rec) + "\n")
            out.flush()

        # --- setup (reported separately) ---
        t0 = time.perf_counter()
        cfg = load_config(args.config)
        rng = np.random.default_rng(cfg.seed)
        problems = build_problems(cfg, rng)
        _prepare(problems)
        t_problems = time.perf_counter() - t0

        t0 = time.perf_counter()
        warm = make_warmup_fixture()
        for name in cfg.solvers:
            for start in warm.starts:
                run_job(warm, start, name, cfg.run_deadline_ms, cfg.budgets_ms)
        t_warmup = time.perf_counter() - t0

        jobs = [(pb, s, a) for pb in problems for s in pb.starts for a in cfg.solvers]
        jobs = [jobs[i] for i in rng.permutation(len(jobs))]
        shard, n_shards = map(int, args.shard.split("/"))
        jobs = jobs[shard::n_shards]

        emit(
            {
                "type": "meta",
                "experiment": cfg.name,
                "config": cfg.model_dump(mode="json"),
                "n_jobs": len(jobs),
                "n_problems": len(problems),
                "import_s": t_import,
                "problem_setup_s": t_problems,
                "warmup_s": t_warmup,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "machine": platform.machine(),
                "threadpools": [
                    {k: d.get(k) for k in ("user_api", "internal_api", "num_threads")}
                    for d in threadpool_info()
                ],
                "thread_env": {k: os.environ.get(k) for k in config.SINGLE_THREAD_ENV},
            }
        )

        # --- timed solver loop ---
        t_loop = time.perf_counter()
        done = 0
        for problem, start, name in jobs:
            if time.perf_counter() - T_START >= cfg.worker_deadline_s:
                emit({"type": "truncated", "completed": done, "n_jobs": len(jobs)})
                break
            emit(
                run_job(problem, start, name, cfg.run_deadline_ms, cfg.budgets_ms, cfg.full_history)
            )
            done += 1
        emit(
            {
                "type": "end",
                "completed": done,
                "loop_s": time.perf_counter() - t_loop,
                "worker_total_s": time.perf_counter() - T_START,
            }
        )


if __name__ == "__main__":
    main()
