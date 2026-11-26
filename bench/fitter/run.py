"""``python -m bench run CONFIG --fitter``: export a config's problems, run the fitter, report.

Problems are built exactly as the Python worker builds them (same config, same seed), so
Rust records are merged with a Python run of the same config when one exists and scored
against shared best-found references.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from ..watchdog import merge_shards

RUST_DIR = Path(__file__).resolve().parents[2] / "rust"
BINARY = RUST_DIR / "target" / "release" / "nlfit-rs"


def ensure_binary() -> Path:
    """Build (or confirm up to date) the release binary; a no-op when nothing changed."""
    subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=RUST_DIR, check=True)
    return BINARY


def export_config(cfg, path: Path) -> dict:
    import numpy as np

    from ..suites import build_problems
    from .export import export_problems

    t0 = time.perf_counter()
    problems = build_problems(cfg, np.random.default_rng(cfg.seed))
    t_build = time.perf_counter() - t0
    meta = {"experiment": cfg.name, "config": cfg.model_dump(mode="json"), "build_s": t_build}
    info = export_problems(problems, meta, path)
    info["build_s"] = t_build
    info["n_problems"] = len(problems)
    return info


def merge_records(rust: Path, python: Path, out: Path) -> None:
    """Python fits for the exported problems + all Rust fits, under one meta record."""

    def read(p):
        with open(p) as fh:
            return [json.loads(line) for line in fh if line.strip()]

    rs, py = read(rust), read(python)
    rmeta = next(r for r in rs if r["type"] == "meta")
    pmeta = next((r for r in py if r["type"] == "meta"), {})
    rfits = [r for r in rs if r["type"] == "fit"]
    pids = {r["problem"] for r in rfits}
    pfits = [r for r in py if r["type"] == "fit" and r["problem"] in pids]
    meta = dict(rmeta)
    py_solvers = pmeta.get("config", {}).get("solvers", sorted({r["solver"] for r in pfits}))
    meta["config"] = {**rmeta["config"], "solvers": py_solvers + rmeta["config"]["solvers"]}
    meta["n_jobs"] = len(rfits) + len(pfits)
    meta["merged_python"] = str(python)
    with open(out, "w") as fh:
        for r in [meta, *pfits, *rfits]:
            fh.write(json.dumps(r) + "\n")


def bench(
    problems_json: Path,
    records: Path,
    budgets,
    extra: tuple[str, ...] = (),
    jobs: int = 1,
    timeout: float | None = None,
) -> None:
    """Run ``nlfit-rs bench``; with ``jobs`` > 1, one process per share of the problems."""
    binary = str(ensure_binary())

    def cmd(problems: Path, out: Path) -> list[str]:
        return [binary, "bench", "--problems", str(problems), "--out", str(out),
                "--policies", "auto", "--budgets", ",".join(f"{b:g}" for b in budgets),
                *extra]  # fmt: skip

    if jobs <= 1:
        subprocess.run(cmd(problems_json, records), check=True, timeout=timeout)
        return
    doc = json.loads(problems_json.read_text())
    parts, outs = [], []
    for i in range(jobs):
        parts.append(records.with_suffix(f".part{i}.json"))
        outs.append(records.with_suffix(f".shard{i}.jsonl"))
        parts[i].write_text(json.dumps({**doc, "problems": doc["problems"][i::jobs]}))
    procs = [subprocess.Popen(cmd(p, o)) for p, o in zip(parts, outs, strict=True)]
    try:
        codes = [p.wait(timeout=timeout) for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        for part in parts:
            part.unlink(missing_ok=True)
    if any(codes):
        sys.exit(f"nlfit-rs exited with codes {codes}")
    merge_shards(outs, records)


def run(
    cfg, out_dir: Path, python_records: Path, plot: bool, t_invocation: float, jobs: int = 1
) -> None:
    from ..report import build_report

    out_dir.mkdir(parents=True, exist_ok=True)
    problems_json = out_dir / "problems.json"
    info = export_config(cfg, problems_json)
    print(
        f"exported {info['n_exported']}/{info['n_problems']} problems "
        f"({info['n_programs']} tapes) in {info['build_s'] + info['export_s']:.1f} s"
        + (f"; skipped {len(info['skipped'])} (no symbolic form)" if info["skipped"] else "")
    )
    records = out_dir / "records_rust.jsonl"
    extra = ("--seed", str(cfg.seed), "--max-wall-s", str(cfg.worker_deadline_s))
    bench(problems_json, records, cfg.budgets_ms, extra, jobs, cfg.watchdog_s + 60)
    final = records
    if python_records.exists():
        final = out_dir / "records.jsonl"
        merge_records(records, python_records, final)
        print(f"merged with {python_records}")
    build_report(final, out_dir, run_info=None, t_invocation=t_invocation, plot=plot)
