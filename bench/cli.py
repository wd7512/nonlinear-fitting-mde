"""Benchmark nonlinear fitters on millisecond CPU budgets.

  python -m bench run CONFIG [--fitter]     fit every problem, then report
  python -m bench report CONFIG [--fitter]  rebuild the report from saved records
  python -m bench gate build | run          fixed dev/test cases for changes to the fitter

CONFIG is a TOML path or a name from bench/configs/. Without --fitter the registered SciPy
baselines run (results/<name>/); with it the Rust fitter runs (results/<name>_rust/, merged
with the baseline records when they exist).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import CONFIG_DIR

T_INVOCATION = time.perf_counter()


def resolve_config(name: str) -> Path:
    for path in (Path(name), CONFIG_DIR / f"{name}.toml"):
        if path.is_file():
            return path
    sys.exit(f"no config {name!r} (a path, or a name in {CONFIG_DIR})")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report"):
        sp = sub.add_parser(name)
        sp.add_argument("config", help="TOML path or config name")
        sp.add_argument("--fitter", action="store_true", help="the Rust fitter (rust/)")
        sp.add_argument("--out", type=Path, default=None, help="default: results/<name>")
        sp.add_argument("--no-plot", action="store_true")
        if name == "run":
            sp.add_argument(
                "--baseline-records", type=Path, default=None,
                help="with --fitter: SciPy records to merge (default: results/<name>/records.jsonl)",
            )  # fmt: skip
            sp.add_argument(
                "--jobs", type=int, nargs="?", const=4, default=1,
                help="parallel workers (bare flag: 4; with --fitter, Pool threads); timings get noisier",
            )  # fmt: skip
    gp = sub.add_parser("gate")
    gp.add_argument("action", choices=["build", "run"])
    gp.add_argument("--tier", choices=["quick", "dev", "test", "rank"], default="quick")
    gp.add_argument("--baseline", type=Path, help="records of an earlier gate run to compare with")
    args = ap.parse_args(argv)

    if args.cmd == "gate":
        from . import gate

        if args.action == "build":
            gate.build()
        else:
            gate.run(args.tier, args.baseline)
        return

    from .experiment import load_config  # validates the TOML before spawning anything

    path = resolve_config(args.config)
    try:
        cfg = load_config(path)
    except Exception as exc:  # show config errors without a traceback
        sys.exit(f"config error in {path}:\n{exc}")

    plot = not args.no_plot
    if args.fitter:
        from .fitter.run import run
        from .report import build_report

        out = args.out or Path("results") / f"{cfg.name}_rust"
        if args.cmd == "run":
            python_records = args.baseline_records or Path("results") / cfg.name / "records.jsonl"
            if not python_records.exists():
                print(f"no baseline records at {python_records}; reporting the fitter alone")
            run(cfg, out, python_records, plot, T_INVOCATION, args.jobs)
        else:
            merged = out / "records.jsonl"
            records = merged if merged.exists() else out / "records_rust.jsonl"
            build_report(records, out, run_info=None, t_invocation=T_INVOCATION, plot=plot)
        return

    out = args.out or Path("results") / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    records = out / "records.jsonl"
    run_info = None
    if args.cmd == "run":
        from .watchdog import run_worker  # numpy is loaded only in the worker

        outcome = run_worker(path, records, cfg.watchdog_s, args.jobs)
        run_info = {
            "worker_wall_s": outcome.wall_s,
            "killed_by_watchdog": outcome.killed,
            "returncode": outcome.returncode,
        }
        if outcome.returncode not in (0, None) and not outcome.killed:
            print(f"worker exited with code {outcome.returncode}", file=sys.stderr)

    from .report import build_report  # heavy imports (pandas) after the timed run

    build_report(records, out, run_info=run_info, t_invocation=T_INVOCATION, plot=plot)
