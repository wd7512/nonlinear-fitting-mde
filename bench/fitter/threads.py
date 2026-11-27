"""Success against time when N fits run at once, through the installed package.

  uv run python -m bench.fitter.threads [SUITE ...] [--threads 1 2 3 4]

For every suite and every N, fits each (problem, start) with ``Pool(threads=N).fit_many`` and
writes ``results/<suite>_threads/records.jsonl`` in the benchmark's record schema, one solver
per N (``symbfitrs_t<N>``), so ``bench.figure`` scores them like any other run. Problems with
no single-expression form (MGH residual vectors, Python-callable candidates) or whose
expression the package cannot compile are skipped and listed in the meta record.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np

CPU_BUDGET_MS = 25.0  # the figure's grid ends here
FLOAT_MAX = 1.7976931348623157e308  # records use f64::MAX where no finite loss was found
SOLVER = "symbfitrs_t{n}"


def expression(model) -> tuple[str, list[str], list[str]] | None:
    """(expression string, variables, parameters) for a benchmark model, or None if it has no
    single-expression form."""
    import sympy as sp

    from ..models import SymbolicModel
    from ..models.families import CoupledTanh, ExpMixture, SinMixture

    if isinstance(model, SymbolicModel):
        expr = model.compiled.expr.replace(  # wrap(d) = atan2(sin d, cos d) = 2 atan(sin d / (1 + cos d))
            lambda x: x.func.__name__ == "wrap",
            lambda x: 2 * sp.atan(sp.sin(x.args[0]) / (1 + sp.cos(x.args[0]))),
        )
        e = sp.sstr(expr)
        return re.sub(r"\bAbs\(", "abs(", e), list(model.variables), list(model.param_names)
    if isinstance(model, ExpMixture | SinMixture):
        k = model.k
        fn, sign = ("exp", "-") if isinstance(model, ExpMixture) else ("sin", "")
        e = " + ".join(f"a{i}*{fn}({sign}b{i}*x)" for i in range(k))
        return (e + " + c" if model.offset else e), ["x"], list(model.param_names)
    if isinstance(model, CoupledTanh):
        p = model.p
        s = " + ".join(f"t{k}*cos({k}*pi*x)" for k in range(p))
        return f"tanh(({s})/sqrt({p}))", ["x"], list(model.param_names)
    return None


def columns(model, variables: list[str]) -> list[np.ndarray]:
    x = np.asarray(model.x, float).reshape(len(model.x), -1)
    return [np.ascontiguousarray(x[:, j]) for j in range(len(variables))]


def record(pb, start: str, theta0, res, solver: str, job_wall_ms: float) -> dict:
    """One benchmark fit record from a ``FitResult``. The package reports the mean squared
    (weighted) residual; the benchmark's loss divides that by the problem's denominator."""
    denom = pb.denom
    ok = res.status != "error" and math.isfinite(res.loss)
    best = res.loss / denom if ok else None
    small = pb.p <= 128  # bench.config.MAX_THETA_P
    ref_loss = None if pb.ref_loss is None or not math.isfinite(pb.ref_loss) else float(pb.ref_loss)
    loss0 = pb.loss_at(np.asarray(theta0, float))
    return {
        "type": "fit",
        "dataset": pb.dataset,
        "model": pb.model_name,
        "curve": pb.curve,
        "loss_norm": pb.loss_norm,
        "ref_loss": ref_loss,
        "ref_mode": pb.ref_mode,
        "ref_params": (
            None if pb.ref_params is None or not small else [float(v) for v in pb.ref_params]
        ),
        **{f"tag_{k}": v for k, v in pb.tags.items()},
        "problem": pb.pid,
        "p": pb.p,
        "m": pb.m,
        "n_linear": int(pb.model.n_linear),
        "loss0": float(loss0) if math.isfinite(loss0) else None,
        "start": start,
        "solver": solver,
        "solver_label": solver,
        "status": res.status,
        "error": res.message if res.status in ("failure", "error") else None,
        "elapsed_ms": float(res.time_ms),
        "job_wall_ms": float(job_wall_ms),
        "best_loss": float(best) if best is not None and math.isfinite(best) else FLOAT_MAX,
        "best_theta": [float(v) for v in res.params] if ok and small else None,
        "theta_at": None,
        "nfev": int(res.nfev),
        "fev_ms": None,
        "n_nonfinite": None,
        "n_trials": None,
        "hist_t_ms": [float(t) for t in res.hist_t_ms],
        "hist_loss": [float(v) / denom for v in res.hist_loss],
    }


def shuffled(n: int, seed: int) -> list[int]:
    return np.random.default_rng(seed).permutation(n).tolist()


def run_suite(suite: str, threads: list[int], out: Path, budget_ms: float | None = None) -> dict:
    import symbfitrs

    from ..config import CONFIG_DIR
    from ..experiment import load_config
    from ..suites import build_problems

    cfg = load_config(CONFIG_DIR / f"{suite}.toml")
    deadline = max(CPU_BUDGET_MS, *cfg.budgets_ms) if budget_ms is None else budget_ms
    problems = build_problems(cfg, np.random.default_rng(cfg.seed))
    jobs, skipped = [], {}
    for pb in problems:
        form = expression(pb.model)
        if form is None:
            skipped[pb.pid] = f"{type(pb.model).__name__} has no single-expression form"
            continue
        expr, variables, params = form
        try:
            symbfitrs.compile(expr, variables, params)
        except ValueError as exc:
            skipped[pb.pid] = f"package cannot compile it: {exc}"
            continue
        x = columns(pb.model, variables)
        if not (np.all(np.isfinite(pb.y)) and all(np.all(np.isfinite(c)) for c in x)):
            skipped[pb.pid] = "non-finite data"
            continue
        w = None if pb.weights is None else np.asarray(pb.weights, float)
        for name, theta0 in pb.starts.items():
            jobs.append((pb, name, np.asarray(theta0, float), expr, variables, x, w))
    order = shuffled(len(jobs), cfg.seed)
    jobs = [jobs[i] for i in order]
    print(f"{suite}: {len(jobs)} fits from {len(problems) - len(skipped)}/{len(problems)} problems")

    if not jobs:
        print(f"{suite}: nothing the package can fit; no records written")
        return {"skipped": skipped}
    by_vars: dict[tuple, list[int]] = {}
    for i, j in enumerate(jobs):
        by_vars.setdefault(tuple(j[4]), []).append(i)
    records = []
    for n in threads:
        pool = symbfitrs.Pool(threads=n)
        pool.fit_many([(jobs[0][3], jobs[0][5], jobs[0][0].y)], variables=jobs[0][4],
                      budget_ms=5, progress=False)  # warm-up  # fmt: skip
        results = [None] * len(jobs)
        t0 = time.perf_counter()
        for variables, idx in by_vars.items():
            batch = [(jobs[i][3], jobs[i][5], jobs[i][0].y, jobs[i][2], jobs[i][6]) for i in idx]
            for i, r in zip(idx, pool.fit_many(batch, variables=list(variables),
                                               budget_ms=deadline, progress=False), strict=True):  # fmt: skip
                results[i] = r
        wall = (time.perf_counter() - t0) * 1e3
        print(f"  {n} thread(s): {wall / 1e3:.1f} s")
        for (pb, start, theta0, *_), r in zip(jobs, results, strict=True):
            records.append(record(pb, start, theta0, r, SOLVER.format(n=n), wall / len(jobs)))
    meta = {
        "type": "meta",
        "experiment": f"{suite}_threads",
        "engine": "symbfitrs package",
        "config": {**cfg.model_dump(mode="json"), "solvers": [SOLVER.format(n=n) for n in threads]},
        "threads": threads,
        "deadline_ms": deadline,
        "n_jobs": len(records),
        "n_problems": len(problems) - len(skipped),
        "skipped": skipped,
        "version": symbfitrs.__version__,
    }
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "records.jsonl", "w") as fh:
        for r in [meta, *records]:
            fh.write(json.dumps(r) + "\n")
    return meta


def main(argv: list[str] | None = None) -> None:
    from ..gate import SUITES

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("suites", nargs="*", default=SUITES)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--results", type=Path, default=Path("results"))
    a = ap.parse_args(argv)
    for s in a.suites:
        run_suite(s, a.threads, a.results / f"{s}_threads")


if __name__ == "__main__":
    main()
