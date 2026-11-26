"""Parent-side watchdog: launch the single benchmark worker with single-thread BLAS
and kill it if it exceeds the global wall-clock cap."""

from __future__ import annotations

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


def run_worker(config_path: Path, records_path: Path, watchdog_s: float) -> WorkerOutcome:
    env = {**os.environ, **config.SINGLE_THREAD_ENV}
    cmd = [
        sys.executable,
        "-m",
        "nlfit_mde.worker",
        "--config",
        str(config_path),
        "--out",
        str(records_path),
    ]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, env=env)
    killed = False
    try:
        proc.wait(timeout=watchdog_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        killed = True
    return WorkerOutcome(records_path, time.perf_counter() - t0, killed, proc.returncode)
