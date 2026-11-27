"""``python -m bench run CONFIG --fitter``: export a config's problems, fit them with the
``symbfitrs`` package and with the native ``varpro`` crate baseline (``varpro_rs``), report.

symbfitrs runs through its public API (``compile`` + ``Pool.fit_many``) from the local build in
``python/`` (``uv run`` rebuilds it when ``rust/src`` changes), never an installed release.
Problems are built exactly as the Python worker builds them (same config, same seed), so these
records are merged with a Python run of the same config when one exists and scored against
shared best-found references.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..config import MAX_THETA_P
from ..watchdog import merge_shards

ROOT = Path(__file__).resolve().parents[2]
RUST_DIR = ROOT / "rust"
BINARY = RUST_DIR / "target" / "release" / "symbfitrs-bench"
FITTER = "symbfitrs"
CHUNK = 2000  # fits per fit_many call, so the wall-time cap is checked between chunks


def ensure_binary() -> Path:
    """Build (or confirm up to date) the release binary; a no-op when nothing changed."""
    subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=RUST_DIR, check=True)
    return BINARY


def local_package():
    """The ``symbfitrs`` built from this checkout, with its version and git state."""
    import symbfitrs

    path = Path(symbfitrs.__file__).resolve()
    if not path.is_relative_to(ROOT / "python"):
        raise SystemExit(f"symbfitrs is imported from {path}, not this checkout's python/")

    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout

    info = {
        "version": symbfitrs.__version__,
        "git_sha": git("rev-parse", "HEAD").strip(),
        "git_dirty": bool(git("status", "--porcelain", "--", "rust", "python").strip()),
    }
    return symbfitrs, info


def export_config(cfg, path: Path) -> dict:
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


def _fit_record(pb: dict, model, start: str, theta0, loss0: float, r, theta_at) -> dict:
    denom = pb["denom"]
    small = len(model.params) <= MAX_THETA_P
    failed = r.status in ("failure", "error")
    best = r.loss / denom if np.isfinite(r.loss) else sys.float_info.max
    return {
        "type": "fit",
        **pb["record"],
        "problem": pb["pid"],
        "p": len(model.params),
        "m": len(pb["y"]),
        "n_linear": len(model.linear),
        "loss0": loss0 if np.isfinite(loss0) else None,
        "start": start,
        "solver": FITTER,
        "solver_label": FITTER,
        "status": "failure" if failed else r.status,
        "error": r.message if failed else None,
        "elapsed_ms": r.time_ms,
        "job_wall_ms": r.time_ms,
        "best_loss": best,
        "best_theta": r.params.tolist() if small and r.history_loss.size else None,
        "theta_at": theta_at if small else None,
        "nfev": r.nfev,
        "fev_ms": None,  # the package does not time evaluations separately
        "n_nonfinite": r.n_nonfinite,
        "n_trials": None,
        "hist_t_ms": r.history_ms.tolist(),
        "hist_loss": (r.history_loss / denom).tolist(),
    }


def bench_package(
    problems_json: Path,
    records: Path,
    budgets,
    threads: int = 1,
    seed: int = 42,
    max_wall_s: float = float("inf"),
) -> None:
    """Fit every (problem, start) with ``symbfitrs.Pool(threads).fit_many``; records follow the
    Python harness schema. Each fit gets the largest budget; ``theta_at`` (parameters at each
    checkpoint, for problems with reference parameters) comes from refits at the smaller ones."""
    t_start = time.perf_counter()
    sf, pkg = local_package()
    doc = json.loads(problems_json.read_text())
    t_import = time.perf_counter() - t_start

    t0 = time.perf_counter()
    models, skipped, probs = {}, dict(doc["meta"].get("skipped", {})), []
    for pb in doc["problems"]:
        key = (pb["expr"], tuple(pb["variables"]), tuple(pb["params"]))
        if key not in models:
            try:
                models[key] = sf.compile(*key)
            except ValueError as e:
                models[key] = str(e)
        if isinstance(models[key], str):
            skipped[pb["pid"]] = f"compile: {models[key]}"
            continue
        x = np.array(pb["x"], dtype=np.float64).reshape(len(pb["variables"]), -1)
        y = np.array(pb["y"], dtype=np.float64)
        w = None if pb["w"] is None else np.array(pb["w"], dtype=np.float64)
        probs.append((pb, models[key], x, y, w))
    t_setup = time.perf_counter() - t0

    deadline = max(budgets)
    pool = sf.Pool(threads=threads)
    t0 = time.perf_counter()
    if probs:  # warm-up: caches, page faults, thread start
        pb, model, x, y, w = probs[0]
        pool.fit_many([(model, x, y, pb["starts"][0][1], w)], budget_ms=5.0, progress=False)
    t_warm = time.perf_counter() - t0

    jobs = [(i, s) for i, (pb, *_) in enumerate(probs) for s in range(len(pb["starts"]))]
    jobs = [jobs[k] for k in np.random.default_rng(seed).permutation(len(jobs))]

    def loss0(i, s):
        pb, model, x, y, w = probs[i]
        with np.errstate(all="ignore"):
            r = model.predict(x, pb["starts"][s][1]) - y
            if w is not None:
                r = r * w
            return float(np.mean(r * r) / pb["denom"])

    def fit_jobs(batch, budget):
        spec = [(probs[i][1], *probs[i][2:4], probs[i][0]["starts"][s][1], probs[i][4])
                for i, s in batch]  # fmt: skip
        return pool.fit_many(spec, budget_ms=budget, progress=False)

    meta = dict(doc["meta"])
    meta["config"] = {**meta.get("config", {}), "solvers": [FITTER], "budgets_ms": list(budgets)}
    meta.update(
        type="meta",
        engine="symbfitrs (python package)",
        package=pkg,
        threads=threads,
        n_jobs=len(jobs),
        n_problems=len(probs),
        skipped=skipped,
        import_s=t_import,
        problem_setup_s=t_setup,
        warmup_s=t_warm,
        deadline_ms=deadline,
        theta_at="refit at each smaller budget, problems with reference parameters only",
        threadpools=[],
        thread_env={"symbfitrs": f"Pool(threads={threads})"},
        python=platform.python_version(),
        numpy=np.__version__,
        machine=platform.machine(),
    )
    out = [meta]
    t_loop = time.perf_counter()
    done = 0
    for c in range(0, len(jobs), CHUNK):
        if time.perf_counter() - t_start >= max_wall_s:
            out.append({"type": "truncated", "completed": done, "n_jobs": len(jobs)})
            break
        batch = jobs[c : c + CHUNK]
        results = fit_jobs(batch, deadline)
        # parameters at each smaller checkpoint, for problems scored on them (NIST LRE)
        need = [k for k, (i, _) in enumerate(batch) if probs[i][0]["record"].get("ref_params")]
        snaps = {k: [] for k in need}
        for b in [b for b in budgets if b < deadline] if need else []:
            for k, r in zip(need, fit_jobs([batch[k] for k in need], b), strict=True):
                snaps[k].append(r.params.tolist() if r.history_loss.size else None)
        for k, ((i, s), r) in enumerate(zip(batch, results, strict=True)):
            pb, model = probs[i][:2]
            best = r.params.tolist() if r.history_loss.size else None
            theta_at = snaps[k] + [best] if k in snaps else None
            out.append(_fit_record(pb, model, pb["starts"][s][0], pb["starts"][s][1],
                                   loss0(i, s), r, theta_at))  # fmt: skip
            done += 1
    out.append(
        {
            "type": "end",
            "completed": done,
            "loop_s": time.perf_counter() - t_loop,
            "worker_total_s": time.perf_counter() - t_start,
        }
    )
    records.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"{FITTER}: {done}/{len(jobs)} fits on {threads} thread(s) in "
          f"{time.perf_counter() - t_loop:.2f} s -> {records}")  # fmt: skip


def bench_varpro_rs(
    problems_json: Path,
    records: Path,
    budgets,
    extra: tuple[str, ...] = (),
    jobs: int = 1,
    timeout: float | None = None,
) -> None:
    """Run ``symbfitrs-bench bench``; with ``jobs`` > 1, one process per share of the problems."""
    binary = str(ensure_binary())

    def cmd(problems: Path, out: Path) -> list[str]:
        return [binary, "bench", "--problems", str(problems), "--out", str(out),
                "--budgets", ",".join(f"{b:g}" for b in budgets),
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
        sys.exit(f"symbfitrs-bench exited with codes {codes}")
    merge_shards(outs, records)


def merge_records(parts: list[Path], out: Path) -> None:
    """All fits of ``parts`` on the problems of the first, under the first's meta record."""

    def read(p):
        with open(p) as fh:
            return [json.loads(line) for line in fh if line.strip()]

    first, *rest = (read(p) for p in parts)
    meta = dict(next(r for r in first if r["type"] == "meta"))
    fits = [r for r in first if r["type"] == "fit"]
    pids = {r["problem"] for r in fits}
    solvers = list(meta["config"]["solvers"])
    for p, recs in zip(parts[1:], rest, strict=True):
        m = next((r for r in recs if r["type"] == "meta"), {})
        more = [r for r in recs if r["type"] == "fit" and r["problem"] in pids]
        solvers += m.get("config", {}).get("solvers", sorted({r["solver"] for r in more}))
        fits += more
        meta.setdefault("merged", []).append(str(p))
    meta["config"] = {**meta["config"], "solvers": list(dict.fromkeys(solvers))}
    meta["n_jobs"] = len(fits)
    with open(out, "w") as fh:
        for r in [meta, *fits]:
            fh.write(json.dumps(r) + "\n")


def run(
    cfg, out_dir: Path, python_records: Path, plot: bool, t_invocation: float, jobs: int = 1
) -> None:
    from ..report import build_report

    out_dir.mkdir(parents=True, exist_ok=True)
    problems_json = out_dir / "problems.json"
    info = export_config(cfg, problems_json)
    print(
        f"exported {info['n_exported']}/{info['n_problems']} problems "
        f"({info['n_models']} models) in {info['build_s'] + info['export_s']:.1f} s"
        + (f"; skipped {len(info['skipped'])} (no symbolic form)" if info["skipped"] else "")
    )
    ours = out_dir / "records_symbfitrs.jsonl"
    bench_package(problems_json, ours, cfg.budgets_ms, jobs, cfg.seed, cfg.worker_deadline_s)
    native = out_dir / "records_varpro_rs.jsonl"
    extra = ("--seed", str(cfg.seed), "--max-wall-s", str(cfg.worker_deadline_s))
    bench_varpro_rs(problems_json, native, cfg.budgets_ms, extra, jobs, cfg.watchdog_s + 60)
    parts = [ours, native] + ([python_records] if python_records.exists() else [])
    final = out_dir / "records.jsonl"
    merge_records(parts, final)
    build_report(final, out_dir, run_info=None, t_invocation=t_invocation, plot=plot)
