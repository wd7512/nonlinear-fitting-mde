"""Parent-side watchdog: launch the single benchmark worker with single-thread BLAS
and kill it if it exceeds the global wall-clock cap."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class WorkerOutcome:
    records_path: Path
    wall_s: float
    killed: bool
    returncode: int | None


def merge_shards(shards: list[Path], out: Path) -> None:
    """Join the records of parallel workers under one meta record, then delete the shards."""
    metas, fits, ends, truncated = [], [], [], []
    for shard in shards:
        if not shard.exists():
            continue
        for line in shard.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # half-written final line after a kill
                continue
            kind = rec.get("type")
            {"meta": metas, "fit": fits, "end": ends, "truncated": truncated}[kind].append(rec)
        shard.unlink()
    records = []
    if metas:
        n_jobs = sum(m["n_jobs"] for m in metas)
        records.append({**metas[0], "n_jobs": n_jobs, "parallel_jobs": len(shards)})
    records += fits
    if truncated:
        records.append({"type": "truncated", "completed": len(fits), "n_jobs": n_jobs})
    if len(ends) == len(shards):
        slowest = {k: max(e[k] for e in ends) for k in ("loop_s", "worker_total_s")}
        records.append({"type": "end", "completed": len(fits), **slowest})
    out.write_text("".join(json.dumps(r) + "\n" for r in records))


def run_worker(
    config_path: Path, records_path: Path, watchdog_s: float, jobs: int = 1
) -> WorkerOutcome:
    """One benchmark worker, or ``jobs`` single-threaded workers sharing the fits."""
    env = {**os.environ, **config.SINGLE_THREAD_ENV}
    outs = [records_path]
    if jobs > 1:
        outs = [records_path.with_suffix(f".shard{i}.jsonl") for i in range(jobs)]
    t0 = time.perf_counter()
    procs = []
    for i, out in enumerate(outs):
        cmd = [
            sys.executable,
            "-m",
            "symbolic_regression_rust.worker",
            "--config",
            str(config_path),
        ]
        cmd += ["--out", str(out), "--shard", f"{i}/{len(outs)}"]
        procs.append(subprocess.Popen(cmd, env=env))
    killed = False
    try:
        for proc in procs:
            proc.wait(timeout=max(watchdog_s - (time.perf_counter() - t0), 0))
    except subprocess.TimeoutExpired:
        killed = True
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    if jobs > 1:
        merge_shards(outs, records_path)
    returncode = next((p.returncode for p in procs if p.returncode), 0)
    return WorkerOutcome(records_path, time.perf_counter() - t0, killed, returncode)
