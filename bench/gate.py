"""Gate for fitter changes: fixed case sets across all benchmark suites.

``build`` freezes each suite's exported problems (``results/<suite>_rust/problems.json``)
with a fixed reference loss (the suite's certified value, or the best any solver found in its
full merged run) and the suite's own success rule. Cases are split into dev and test by group
(suite/model: a whole NIST problem, MGH function, synthetic family or tuning equation is on
one side), so a rule picked on dev is judged on families it never saw.

Tiers (``data/gate/<tier>.json``, problems.json format):
  quick  16 dev cases each from NIST, MGH and synthetic (48), seconds; for iterating
  dev    up to 150 dev cases per suite; for choosing rules
  test   up to 150 test cases per suite; for accepting a change
  rank   every candidate equation on the SAGE-Fit curves and 100 Stringer cells, 10 and 100 ms

``run`` runs the fitter on a tier: success per suite at each budget (main checkpoint 5 ms), or,
on ``rank``, ranking quality. ``--baseline`` compares with the records of an earlier run.

Acceptance, fixed before any experiment (test tier, ``rank`` tier, 3 repeats for timing):
  gain     mean-over-suites success at 5 ms up >= 1.0 pp, or median fit time down >= 10%
           with success at 5 ms within 0.5 pp
  and      no budget down > 0.5 pp, no suite down > 2 pp at 5 ms, and Kendall tau at 10 and
           100 ms down by no more than 0.02
"""

from __future__ import annotations

import hashlib
import json
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
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
BUDGETS = [1.0, 5.0, 10.0, 25.0]
RANK_BUDGETS = [10.0, 100.0]
MAIN_MS = 5.0
TEST_SHARE = 0.4
# tier -> (side of the split, suites, cases per suite)
TIERS = {
    "quick": ("dev", ["nist_strd", "mgh", "synthetic_scale"], 16),
    "dev": ("dev", SUITES, 150),
    "test": ("test", SUITES, 150),
}
RANK_CURVES = {"sagefit_spb": None, "stringer2019_gratings": 100}  # None = all curves
FITTER = ("symbfitrs", "rs_auto")  # record labels; rs_auto = runs before the rename
BASELINE = "baseline"


def _side(group: str) -> str:
    h = int(hashlib.sha1(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < TEST_SHARE else "dev"


def _refs(records: Path) -> pd.Series:
    """Fixed per-problem reference from a suite's full merged run."""
    from .report import load_records
    from .scoring import fill_reference

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
    tiers: dict[str, list] = {t: [] for t in [*TIERS, "rank"]}
    programs: list[dict] = []
    for suite in SUITES:
        d = results / f"{suite}_rust"
        if not (d / "problems.json").exists():  # export only; no fitting needed for this
            from .experiment import load_config
            from .fitter.run import export_config

            d.mkdir(parents=True, exist_ok=True)
            export_config(load_config(CONFIG_DIR / f"{suite}.toml"),
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
        for tier, (side, suites, n) in TIERS.items():
            if suite not in suites:
                continue
            cases, groups = side_cases[side]
            picked: dict[int, list[int]] = {}
            for i, s in _sample(cases, groups, n, rng):
                picked.setdefault(i, []).append(s)
            for i, ss in picked.items():
                pb = dict(probs[i])
                pb["starts"] = [pb["starts"][s] for s in sorted(ss)]
                tiers[tier].append(pb)
        if suite in RANK_CURVES:  # ranking needs every candidate and start of a curve
            curves = sorted({pb["record"]["curve"] for pb in probs})
            if (k := RANK_CURVES[suite]) is not None and k < len(curves):
                curves = list(rng.choice(curves, k, replace=False))
            keep = set(curves)
            tiers["rank"] += [dict(pb) for pb in probs if pb["record"]["curve"] in keep]
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


def score_gate(records: Path, budgets: list[float] = BUDGETS) -> pd.DataFrame:
    """Per-fit success and time-to-target under the case's own suite rule."""
    from .report import load_records
    from .scoring import score

    _, fits, _ = load_records(records)
    parts = []
    for (mode, target, floor), g in fits.groupby(["gate_mode", "gate_target", "gate_floor"]):
        scored = score(g, budgets, [target], mode, floor)
        for b in budgets:
            scored[f"ok_{b:g}"] = scored[f"ok_{target:.0e}_{b:g}ms"]
        scored["ttt"] = scored[f"ttt_{target:.0e}"]
        parts.append(scored)
    return pd.concat(parts)


def table(df: pd.DataFrame, baseline: str) -> str:
    main = f"ok_{MAIN_MS:g}"
    per_suite = df.pivot_table(index="solver", columns="suite", values=main, aggfunc="mean")
    per_suite = per_suite[[s for s in SUITES if s in per_suite.columns]]
    # overall = mean over suites, so a large suite does not dominate
    overall = {f"{b:g} ms": df.groupby(["solver", "suite"])[f"ok_{b:g}"].mean()
               .groupby("solver").mean() for b in BUDGETS}  # fmt: skip
    t = pd.concat([pd.DataFrame(overall), per_suite.add_prefix(f"{MAIN_MS:g} ms ")], axis=1)
    lines = ["success (mean over suites; per-suite columns at the main checkpoint)", ""]
    lines.append(t.map(lambda v: f"{v:.1%}").to_markdown())
    if baseline in t.index:
        delta = (t - t.loc[baseline]) * 100
        lines += ["", f"change vs {baseline} (percentage points)", ""]
        lines.append(delta.drop(index=baseline).map(lambda v: f"{v:+.1f}").to_markdown())
    g = df.groupby("solver")
    speed = pd.DataFrame(
        {
            "median fit ms": g.elapsed_ms.median().map("{:.3f}".format),
            "median ms to target": g.ttt.median().map("{:.3f}".format),
            "failures": g.status.apply(lambda s: s.str.startswith("fail").sum()),
            "overruns (>10%)": g.elapsed_ms.apply(lambda s: (s > 1.1 * max(BUDGETS)).sum()),
            "fits": g.size(),
        }
    )
    lines += ["", "speed and failures", "", speed.to_markdown()]
    return "\n".join(lines)


def rank_table(df: pd.DataFrame) -> str:
    """Ranking quality per suite; never averaged with fitting success."""
    from .scoring import ranking_scores

    lines = []
    for suite, g in df.groupby("suite"):
        t = ranking_scores(g, RANK_BUDGETS)
        t.columns = [f"{kind} @ {b:g} ms" for kind, b in t.columns]
        lines += [f"{suite}: {g.curve.nunique()} curves", "", t.to_markdown(floatfmt=".3f"), ""]
    return "\n".join(lines)


def run(tier: str, baseline: Path | None = None, out: Path | None = None) -> None:
    from .fitter.run import bench

    problems = GATE_DIR / f"{tier}.json"
    if not problems.exists():
        raise SystemExit(f"{problems} missing: run `python -m bench gate build` first")
    out = out or Path("results") / "gate" / f"{tier}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    budgets = RANK_BUDGETS if tier == "rank" else BUDGETS
    bench(problems, out, budgets)
    df = score_gate(out, budgets)
    if baseline is not None:
        old = score_gate(baseline, budgets).query("solver in @FITTER")
        if set(zip(old.problem, old.start, strict=True)) != set(
            zip(df.problem, df.start, strict=True)
        ):
            raise SystemExit(f"{baseline} was not run on the same cases as {problems}")
        df = pd.concat([old.assign(solver=BASELINE), df])
    text = rank_table(df) if tier == "rank" else table(df, BASELINE)
    out.with_suffix(".md").write_text(text + "\n")
    print(text)
