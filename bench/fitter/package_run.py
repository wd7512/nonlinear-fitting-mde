"""Run the benchmark through the package (``symbfitrs.Pool.fit_many``) on N workers.

  uv run python -m bench.fitter.package_run [SUITE ...] [--threads 1 2 3 4]

For every suite and every N, fits each (problem, start) with ``Pool(threads=N)`` exactly as
``bench run --fitter`` does (``bench.fitter.run.bench_package``) and writes
``results/<suite>_threads/records.jsonl``, one solver per N (``symbfitrs_t<N>``), so
``bench.figure`` scores them like any other run.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from .run import bench_package, export_config

CPU_BUDGET_MS = 25.0  # the figure's grid ends here
SOLVER = "symbfitrs_t{n}"


def run_suite(suite: str, threads: list[int], out: Path, budget_ms: float | None = None) -> dict:
    from ..config import CONFIG_DIR
    from ..experiment import load_config

    cfg = load_config(CONFIG_DIR / f"{suite}.toml")
    deadline = max(CPU_BUDGET_MS, *cfg.budgets_ms) if budget_ms is None else budget_ms
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        problems = Path(tmp) / "problems.json"
        export_config(cfg, problems)
        runs = {SOLVER.format(n=n): n for n in threads}
        return bench_package(problems, out / "records.jsonl", [deadline], runs, cfg.seed)[0]


def main(argv: list[str] | None = None) -> None:
    from ..gate import SUITES

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("suites", nargs="*", default=SUITES)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--budget-ms", type=float, help="per-fit budget (default: the suite's)")
    ap.add_argument("--kind", default="threads", help="results/<suite>_<kind>")
    ap.add_argument("--results", type=Path, default=Path("results"))
    a = ap.parse_args(argv)
    for s in a.suites:
        run_suite(s, a.threads, a.results / f"{s}_{a.kind}", a.budget_ms)


if __name__ == "__main__":
    main()
