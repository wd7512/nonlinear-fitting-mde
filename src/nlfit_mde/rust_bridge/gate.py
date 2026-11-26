"""Gate for Rust optimiser changes: fixed case sets across all benchmark suites.

``build`` freezes each suite's exported problems (``results/<suite>_rust/problems.json``)
with a fixed reference loss (the suite's certified value, or the best any solver found in its
full merged run) and the suite's own success rule. Cases are split into dev and test by group
(suite/model: a whole NIST problem, MGH function, synthetic family or tuning equation is on
one side), so a rule tuned on dev is judged on families it never saw.

Tiers (``data/gate/<tier>.json``, problems.json format):
  quick  8 dev cases per suite (48), seconds; for iterating
  dev    up to 150 dev cases per suite; for tuning rules
  test   up to 150 test cases per suite; for accepting a change

``run`` runs nlfit-rs policies on a tier and prints success per suite at each budget, with
the change against the baseline policy (default ``auto``). Main checkpoint: 5 ms.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

SUITES = [
    "nist_strd",
    "mgh",
    "fitbenchmarking",
    "sagefit_spb",
    "synthetic_scale",
    "stringer2019_gratings",
]
GATE_DIR = Path("data/gate")
BUDGETS = [1.0, 5.0, 10.0, 25.0]
MAIN_MS = 5.0
TEST_SHARE = 0.4
PER_SUITE = {"quick": 8, "dev": 150, "test": 150}


def _side(group: str) -> str:
    h = int(hashlib.sha1(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < TEST_SHARE else "dev"


def _refs(records: Path) -> pd.Series:
    """Fixed per-problem reference from a suite's full merged run."""
    from ..report import load_records
    from ..scoring import fill_reference

    _, fits, _ = load_records(records)
    return fill_reference(fits).groupby("problem").ref.first()


def _sample(cases: list[tuple[int, int]], groups: list[str], n: int, rng) -> list:
    """Up to n cases, spread over groups (one per group per round, random order)."""
    order = rng.permutation(len(cases))
    seen: dict[str, int] = {}
    ranked = []
    for i in order:
        g = groups[i]
        ranked.append((seen.get(g, 0), len(ranked), cases[i]))
        seen[g] = seen.get(g, 0) + 1
    return [c for *_, c in sorted(ranked)[:n]]


def build(results: Path = Path("results"), out: Path = GATE_DIR, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    tiers: dict[str, list] = {t: [] for t in PER_SUITE}
    programs: list[dict] = []
    for suite in SUITES:
        d = results / f"{suite}_rust"
        if not (d / "problems.json").exists():  # export only; no fitting needed for this
            from ..experiment import load_config
            from .run import export_config

            d.mkdir(parents=True, exist_ok=True)
            export_config(load_config(Path("configs/benchmarks") / f"{suite}.toml"),
                          d / "problems.json")  # fmt: skip
        doc = json.loads((d / "problems.json").read_text())
        cfg = doc["meta"]["config"]
        merged = d / "records.jsonl"
        refs = _refs(merged) if merged.exists() else pd.Series(dtype=float)
        base = len(programs)
        programs += doc["programs"]
        side_cases: dict[str, tuple[list, list]] = {"dev": ([], []), "test": ([], [])}
        probs = doc["problems"]
        for i, pb in enumerate(probs):
            rec = pb["record"]
            if rec["ref_mode"] in ("best_found", "given_or_best"):
                if pb["pid"] not in refs.index:
                    raise SystemExit(f"{suite}: no merged run with {pb['pid']}; run it first")
                ref = float(refs[pb["pid"]])
            else:
                ref = 0.0 if rec["ref_mode"] == "none" else float(rec["ref_loss"])
            group = f"{suite}/{rec['model']}"
            rec.update(
                suite=suite,
                ref_loss=ref,
                ref_mode="given",
                gate_group=group,
                gate_mode=cfg.get("success", "absolute"),
                gate_target=cfg["targets"][0],
                gate_floor=cfg.get("relative_floor", 1e-20),
            )
            for k in ("prog_f", "prog_fj", "prog_sep"):
                if pb.get(k) is not None:
                    pb[k] += base
            cases, groups = side_cases[_side(group)]
            for s in range(len(pb["starts"])):
                cases.append((i, s))
                groups.append(group)
        for tier, n in PER_SUITE.items():
            cases, groups = side_cases["test" if tier == "test" else "dev"]
            picked: dict[int, list[int]] = {}
            for i, s in _sample(cases, groups, n, rng):
                picked.setdefault(i, []).append(s)
            for i, ss in picked.items():
                pb = dict(probs[i])
                pb["starts"] = [pb["starts"][s] for s in sorted(ss)]
                tiers[tier].append(pb)
    out.mkdir(parents=True, exist_ok=True)
    for tier, probs in tiers.items():
        used = sorted({pb[k] for pb in probs for k in ("prog_f", "prog_fj", "prog_sep")
                       if pb.get(k) is not None})  # fmt: skip
        remap = {old: new for new, old in enumerate(used)}
        for pb in probs:
            for k in ("prog_f", "prog_fj", "prog_sep"):
                if pb.get(k) is not None:
                    pb[k] = remap[pb[k]]
        meta = {"experiment": f"gate_{tier}", "config": {"name": f"gate_{tier}"}}
        doc = {"meta": meta, "programs": [programs[i] for i in used], "problems": probs}
        (out / f"{tier}.json").write_text(json.dumps(doc, allow_nan=False))
        n_cases = sum(len(pb["starts"]) for pb in probs)
        print(f"{tier:5s}: {n_cases} cases, {len(probs)} problems, {len(used)} tapes")


def score_gate(records: Path) -> pd.DataFrame:
    """Per-fit success at each budget under the case's own suite rule."""
    from ..report import load_records
    from ..scoring import score

    _, fits, _ = load_records(records)
    parts = []
    for (mode, target, floor), g in fits.groupby(["gate_mode", "gate_target", "gate_floor"]):
        parts.append(score(g, BUDGETS, [target], mode, floor).assign(target=target))
    df = pd.concat(parts)
    for b in BUDGETS:
        df[f"ok_{b:g}"] = [r[f"ok_{r.target:.0e}_{b:g}ms"] for _, r in df.iterrows()]
    return df


def table(df: pd.DataFrame, baseline: str) -> str:
    main = f"ok_{MAIN_MS:g}"
    per_suite = df.pivot_table(index="solver", columns="suite", values=main, aggfunc="mean")
    per_suite = per_suite[[s for s in SUITES if s in per_suite.columns]]
    # overall = mean over suites, so a large suite does not dominate
    overall = {f"{b:g} ms": df.groupby(["solver", "suite"])[f"ok_{b:g}"].mean()
               .groupby("solver").mean() for b in BUDGETS}  # fmt: skip
    t = pd.concat([pd.DataFrame(overall), per_suite.add_prefix(f"{MAIN_MS:g} ms ")], axis=1)
    shown = t.map(lambda v: f"{v:.1%}")
    shown["median fit ms"] = df.groupby("solver").elapsed_ms.median().map(lambda v: f"{v:.3f}")
    lines = ["success (mean over suites; per-suite columns at the main checkpoint)", ""]
    lines.append(shown.to_markdown())
    if baseline in t.index:
        delta = (t - t.loc[baseline]) * 100
        lines += ["", f"change vs {baseline} (percentage points)", ""]
        lines.append(delta.drop(index=baseline).map(lambda v: f"{v:+.1f}").to_markdown())
    return "\n".join(lines)


def run(tier: str, policies: str, baseline: str, out: Path | None = None) -> None:
    from .run import ensure_binary

    problems = GATE_DIR / f"{tier}.json"
    if not problems.exists():
        raise SystemExit(f"{problems} missing: run `nlfit-mde gate build` first")
    pols = policies.split(",")
    if baseline not in pols:
        pols.insert(0, baseline)
    out = out or Path("results") / "gate" / f"{tier}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(ensure_binary()), "bench", "--problems", str(problems), "--out", str(out),
           "--policies", ",".join(pols), "--budgets", ",".join(f"{b:g}" for b in BUDGETS)]  # fmt: skip
    subprocess.run(cmd, check=True)
    text = table(score_gate(out), f"rs_{baseline}")
    out.with_suffix(".md").write_text(text + "\n")
    print(text)
