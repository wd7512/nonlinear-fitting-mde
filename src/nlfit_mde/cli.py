"""nlfit-mde: benchmark nonlinear least-squares optimisers under tiny CPU-time budgets.

  nlfit-mde run CONFIG [--rust]     fit every problem with every solver, then report
  nlfit-mde report CONFIG [--rust]  rebuild the report from saved records
  nlfit-mde inspect CONFIG          validate a config and preview its problems
  nlfit-mde gate build | run        fixed dev/test cases for changes to the Rust optimiser

CONFIG is a TOML path or a config name, e.g. `nist_strd` for configs/benchmarks/nist_strd.toml.
Results go to results/<name>/ (Python solvers) or results/<name>_rust/ (Rust optimiser, merged
with the Python records when they exist).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

T_INVOCATION = time.perf_counter()
CONFIG_DIRS = [Path("configs/benchmarks"), Path("configs/examples"), Path("configs")]


def resolve_config(name: str) -> Path:
    path = Path(name)
    if path.exists():
        return path
    for d in CONFIG_DIRS:
        if (d / f"{name}.toml").exists():
            return d / f"{name}.toml"
    sys.exit(f"no config {name!r} (a path, or a name in {', '.join(map(str, CONFIG_DIRS))})")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="nlfit-mde", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report", "inspect"):
        sp = sub.add_parser(name)
        sp.add_argument("config", help="TOML path or config name")
        if name != "inspect":
            sp.add_argument("--rust", action="store_true", help="the Rust optimiser (rust/)")
            sp.add_argument("--out", type=Path, default=None, help="default: results/<name>")
            sp.add_argument("--no-plot", action="store_true")
    gp = sub.add_parser("gate")
    gp.add_argument("action", choices=["build", "run"])
    gp.add_argument("--tier", choices=["quick", "dev", "test"], default="quick")
    gp.add_argument("--policies", default="auto", help="comma-separated Rust policies")
    gp.add_argument("--baseline", default="auto", help="policy the others are compared with")
    args = ap.parse_args(argv)

    if args.cmd == "gate":
        from .rust_bridge import gate

        gate.build() if args.action == "build" else gate.run(
            args.tier, args.policies, args.baseline
        )
        return

    from .experiment import load_config  # validates the TOML before spawning anything

    path = resolve_config(args.config)
    try:
        cfg = load_config(path)
    except Exception as exc:  # show config errors without a traceback
        sys.exit(f"config error in {path}:\n{exc}")

    if args.cmd == "inspect":
        from .inspect_cmd import inspect

        inspect(cfg)
        return

    plot = not args.no_plot
    if args.rust:
        from .report import build_report
        from .rust_bridge.run import run

        out = args.out or Path("results") / f"{cfg.name}_rust"
        if args.cmd == "run":
            run(cfg, out, Path("results") / cfg.name / "records.jsonl", plot, T_INVOCATION)
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

        outcome = run_worker(path, records, cfg.watchdog_s)
        run_info = {
            "worker_wall_s": outcome.wall_s,
            "killed_by_watchdog": outcome.killed,
            "returncode": outcome.returncode,
        }
        if outcome.returncode not in (0, None) and not outcome.killed:
            print(f"worker exited with code {outcome.returncode}", file=sys.stderr)

    from .report import build_report  # heavy imports (pandas) after the timed run

    build_report(records, out, run_info=run_info, t_invocation=T_INVOCATION, plot=plot)


if __name__ == "__main__":
    main()
