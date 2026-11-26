"""Gate for Rust optimiser changes: fixed case sets across all benchmark suites.

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

``run`` runs nlfit-rs policies on a tier and compares them with the baseline policy (default
``auto``): success per suite at each budget (main checkpoint 5 ms), or, on ``rank``, ranking
quality. ``compare`` shows on which cases and feature ranges a challenger beats the baseline.

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
WIN_MARGIN = 0.1  # both reach the target: faster by more than this share wins


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
    tiers: dict[str, list] = {t: [] for t in [*TIERS, "rank"]}
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
    from ..report import load_records
    from ..scoring import score

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
    from ..scoring import ranking_scores

    lines = []
    for suite, g in df.groupby("suite"):
        t = ranking_scores(g, RANK_BUDGETS)
        t.columns = [f"{kind} @ {b:g} ms" for kind, b in t.columns]
        lines += [f"{suite}: {g.curve.nunique()} curves", "", t.to_markdown(floatfmt=".3f"), ""]
    return "\n".join(lines)


def _run_policies(tier: str, policies: list[str], budgets, out: Path | None) -> Path:
    from .run import bench

    problems = GATE_DIR / f"{tier}.json"
    if not problems.exists():
        raise SystemExit(f"{problems} missing: run `symreg-rs gate build` first")
    out = out or Path("results") / "gate" / f"{tier}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    bench(problems, out, ",".join(policies), budgets)
    return out


def run(tier: str, policies: str, baseline: str, out: Path | None = None) -> None:
    pols = policies.split(",")
    if baseline not in pols:
        pols.insert(0, baseline)
    budgets = RANK_BUDGETS if tier == "rank" else BUDGETS
    out = _run_policies(tier, pols, budgets, out)
    df = score_gate(out, budgets)
    text = rank_table(df) if tier == "rank" else table(df, f"rs_{baseline}")
    out.with_suffix(".md").write_text(text + "\n")
    print(text)


def pair(df: pd.DataFrame, baseline: str, challenger: str) -> pd.DataFrame:
    """One row per case: both policies' outcome, the case features and who won."""
    keys = ["problem", "start"]
    main = f"ok_{MAIN_MS:g}"
    base = df[df.solver == baseline].set_index(keys)
    chal = df[df.solver == challenger].set_index(keys).reindex(base.index)
    feats = [c for c in base.columns if c.startswith(("feat_", "probe_"))]
    t = base[["suite", *feats]].copy()
    for name, side in (("base", base), ("chal", chal)):
        t[f"ok_{name}"] = side[main].astype(bool)
        t[f"ttt_{name}"] = side.ttt.where(t[f"ok_{name}"])
    faster = t.ttt_chal < (1 - WIN_MARGIN) * t.ttt_base
    slower = t.ttt_base < (1 - WIN_MARGIN) * t.ttt_chal
    win = (t.ok_chal & ~t.ok_base) | faster
    loss = (t.ok_base & ~t.ok_chal) | slower
    t["outcome"] = np.select([win, loss], ["win", "loss"], "tie")
    return t.reset_index()


def routed(t: pd.DataFrame, route: np.ndarray, probe: bool) -> tuple[float, float]:
    """(mean-over-suites success at 5 ms, geometric mean ms to target) if ``route`` cases go to the
    challenger. A probe feature must be computed for every case, so its cost is charged."""
    ok = np.where(route, t.ok_chal, t.ok_base)
    ttt = np.where(route, t.ttt_chal, t.ttt_base)
    if probe:
        ttt = ttt + t.probe_ms.fillna(0).to_numpy()
        ok = ok & (ttt <= MAIN_MS)
    ms = ttt[ok]
    return (
        float(pd.Series(ok, index=t.index).groupby(t.suite).mean().mean()),
        float(np.exp(np.log(np.maximum(ms, 1e-4)).mean())) if ms.size else np.nan,
    )


def best_threshold(t: pd.DataFrame, feature: str) -> dict | None:
    """Best "challenger when feature <= / > x" rule: highest success, then fastest."""
    x = pd.to_numeric(t[feature], errors="coerce").to_numpy(float)
    values = np.unique(x[np.isfinite(x)])
    if values.size < 2:
        return None
    best = None
    for cut in (values[:-1] + values[1:]) / 2:
        for op, route in (("<=", x <= cut), (">", x > cut)):
            ok, ms = routed(t, route, feature.startswith("probe_"))
            key = (round(ok, 6), -ms if np.isfinite(ms) else -np.inf)
            if best is None or key > best["key"]:
                best = {"key": key, "feature": feature, "rule": f"{op} {cut:.4g}",
                        "success": ok, "ms to target": ms, "routed": float(route.mean())}  # fmt: skip
    return best


def compare_report(t: pd.DataFrame, baseline: str, challenger: str) -> str:
    counts = pd.crosstab(t.suite, t.outcome, margins=True)
    lines = [f"{challenger} vs {baseline}: cases by outcome at {MAIN_MS:g} ms", ""]
    lines.append(counts.to_markdown())
    skip = {"probe_ms", "probe_ok"}
    features = [c for c in t.columns if c.startswith(("feat_", "probe_")) and c not in skip]
    rules = []
    for f in features:
        x = pd.to_numeric(t[f], errors="coerce").astype(float)
        x = x.where(np.isfinite(x))
        if x.nunique() < 2:
            continue
        bins = x if x.nunique() <= 5 else pd.qcut(x, 4, duplicates="drop")
        lines += ["", f"### {f}", "", pd.crosstab(bins, t.outcome).to_markdown()]
        if (rule := best_threshold(t, f)) is not None:
            rules.append(rule)
    never = routed(t, np.zeros(len(t), bool), False)
    always = routed(t, np.ones(len(t), bool), False)
    rows = [
        {"feature": f"never route ({baseline})", "success": never[0], "ms to target": never[1]},
        {
            "feature": f"always route ({challenger})",
            "success": always[0],
            "ms to target": always[1],
        },
    ]
    rules.sort(key=lambda r: r.pop("key"), reverse=True)
    lines += ["", "### best single threshold per feature (route to challenger when true)", ""]
    lines.append(pd.DataFrame(rows + rules).fillna("").to_markdown(index=False, floatfmt=".3f"))
    return "\n".join(lines)


def compare(tier: str, challenger: str, baseline: str, records: Path | None = None) -> None:
    out = records or _run_policies(tier, [baseline, challenger], BUDGETS, None)
    base, chal = f"rs_{baseline}", f"rs_{challenger}"
    t = pair(score_gate(out), base, chal)
    text = compare_report(t, base, chal)
    stem = out.with_name(f"{out.stem}_{challenger}")
    t.to_csv(stem.with_suffix(".cases.csv"), index=False)
    stem.with_suffix(".compare.md").write_text(text + "\n")
    print(text)
    print(f"\nper-case outcomes: {stem.with_suffix('.cases.csv')}")
