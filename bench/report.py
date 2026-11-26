"""Summaries and outputs (markdown, CSV, plot)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .scoring import ok_col, ranking_scores, score, target_label

MAX_MD_ROWS = 200  # larger per-run tables go to CSV only


def load_records(path: Path) -> tuple[dict, pd.DataFrame, dict]:
    meta, fits, tail = {}, [], {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # half-written final line after a kill
                continue
            kind = rec.get("type")
            if kind == "meta":
                meta = rec
            elif kind == "fit":
                fits.append(rec)
            else:
                tail[kind] = rec
    return meta, pd.DataFrame(fits), tail


def _pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{100 * x:.0f}%"


class Report:
    def __init__(
        self,
        df: pd.DataFrame,
        budgets: list[float],
        targets: list[float],
        solvers: list[str],
        mode: str = "absolute",
    ):
        self.df, self.budgets, self.targets, self.mode = df, budgets, targets, mode
        self.solvers = [s for s in solvers if s in set(df.solver)]

    def lab(self, tg: float) -> str:
        return target_label(tg, self.mode)

    def _ordered(self, t: pd.DataFrame) -> pd.DataFrame:
        return t.reindex(self.solvers)

    def success(self, by: str | None, target: float) -> pd.DataFrame:
        cols = {ok_col(target, b): f"{b:g} ms" for b in self.budgets}
        if by is None:
            t = self.df.groupby("solver")[list(cols)].mean().rename(columns=cols)
        else:
            t = self.df.groupby(["solver", by])[list(cols)].mean().unstack(by)
            t.columns = [f"{grp} @ {cols[c]}" for c, grp in t.columns]
        return self._ordered(t)

    def pooled(self) -> pd.DataFrame:
        parts = []
        for tg in self.targets:
            t = self.success(None, tg)
            t.columns = [f"{self.lab(tg)} @ {c}" for c in t.columns]
            parts.append(t)
        return pd.concat(parts, axis=1)

    def multistart(self) -> pd.DataFrame:
        """Best over all starts of a problem, run sequentially: budgets add up."""
        n = self.df.groupby("problem").start.nunique()
        label = f"{n.min()}×" if n.min() == n.max() else f"{n.min()}–{n.max()}×"
        out = {}
        for tg in self.targets:
            for b in self.budgets:
                per = self.df.groupby(["solver", "problem"])[ok_col(tg, b)].any()
                out[f"{self.lab(tg)} @ {label}{b:g} ms"] = per.groupby("solver").mean()
        return self._ordered(pd.DataFrame(out))

    def outcomes(self) -> pd.DataFrame:
        df = self.df
        t = pd.crosstab(df.solver, df.status.str.split(":").str[0])
        t["runs w/ non-finite trials"] = df.groupby("solver").had_nonfinite.sum()
        t["no finite loss"] = df.groupby("solver").best_loss.apply(
            lambda s: (~np.isfinite(s)).sum()
        )
        return self._ordered(t)

    def costs(self) -> pd.DataFrame:
        rows = {}
        last = max(self.budgets)
        for s, g in self.df.groupby("solver"):
            row = {
                f"success {self.lab(tg)} @{last:g}ms": _pct(g[ok_col(tg, last)].mean())
                for tg in self.targets
            }
            row |= {
                f"median time-to-{tg:.0e} (ms, successes)": g[f"ttt_{tg:.0e}"].median()
                for tg in self.targets
            }
            row |= {
                "median nfev": g.nfev.median(),
                "fev time total (ms)": g.fev_ms.sum(),
                "µs / fev": 1e3 * g.fev_ms.sum() / max(g.nfev.sum(), 1),
                "max run elapsed (ms)": g.elapsed_ms.max(),
                "fits/s (end-to-end)": len(g) / (g.job_wall_ms.sum() / 1e3),
            }
            rows[s] = row
        return self._ordered(pd.DataFrame(rows).T)

    def checkpoint_losses(self, budget: float) -> pd.DataFrame:
        t = self.df.pivot_table(
            index=["group", "p", "curve", "start"],
            columns="solver",
            values=f"best_{budget:g}ms",
            aggfunc="min",
        )
        t = t[[s for s in self.solvers if s in t.columns]]
        return t.map(lambda v: "inf" if not np.isfinite(v) else f"{v:.2e}")

    def ranking(self) -> pd.DataFrame | None:
        """Candidate-equation ranking quality (see ``scoring.ranking_scores``)."""
        t = ranking_scores(self.df, self.budgets)
        if t is None:
            return None
        formats = {
            "tau": ("Kendall τ", lambda v: "–" if pd.isna(v) else f"{v:.3f}"),
            "top1": ("top-1", _pct),
            "lost": ("lost rate", lambda v: f"{v:.3f}"),
        }
        out = {
            f"{label} @ {b:g} ms": t[kind, b].map(fmt)
            for kind, (label, fmt) in formats.items()
            for b in self.budgets
        }
        return self._ordered(pd.DataFrame(out))

    def parameter_accuracy(self) -> pd.DataFrame | None:
        """Log relative error (correct significant digits, capped at 11) of the parameters at
        each checkpoint versus known reference parameters (e.g. NIST certified values);
        reported as min over parameters, then the share of runs with LRE >= 4 and >= 6."""
        from .suites.nist import lre

        df = self.df
        sel = df.ref_mode.isin(["given", "given_or_best"]) & df.ref_params.notna()
        if "theta_at" not in df or not sel.any():
            return None
        sub = df[sel]
        rows = {}
        for i, b in enumerate(self.budgets):
            vals = [
                np.nan if th is None or th[i] is None else lre(th[i], rp).min()
                for th, rp in zip(sub.theta_at, sub.ref_params, strict=True)
            ]
            v = pd.Series(vals, index=sub.index).fillna(0.0)
            g = v.groupby(sub.solver)
            for s in g.groups:
                rows.setdefault(s, {})
                rows[s][f"median LRE @ {b:g} ms"] = f"{g.get_group(s).median():.1f}"
                rows[s][f"LRE≥4 @ {b:g} ms"] = _pct((g.get_group(s) >= 4).mean())
                rows[s][f"LRE≥6 @ {b:g} ms"] = _pct((g.get_group(s) >= 6).mean())
        t = pd.DataFrame(rows).T
        order = [
            c for kind in ("median", "LRE≥4", "LRE≥6") for c in t.columns if c.startswith(kind)
        ]
        return self._ordered(t[order])


def build_report(
    records: Path, out_dir: Path, run_info: dict | None, t_invocation: float, plot: bool = True
) -> None:
    meta, fits, tail = load_records(records)
    if fits.empty:
        print("No completed fit records found.")
        return
    cfg = meta.get("config", {})
    budgets = cfg.get("budgets_ms", list(config.BUDGETS_MS))
    targets = cfg.get("targets", list(config.TARGETS))
    solvers = cfg.get("solvers", sorted(fits.solver.unique()))
    mode = cfg.get("success", "absolute")
    df = score(fits, budgets, targets, mode, cfg.get("relative_floor", 1e-20))
    rep = Report(df, budgets, targets, solvers, mode)
    n_expected = meta.get("n_jobs", len(df))
    complete = len(df) == n_expected
    costs = rep.costs()
    sections: list[tuple[str, str]] = []

    inv = {
        "experiment": meta.get("experiment"),
        "problems": meta.get("n_problems"),
        "completed fits": f"{len(df)} / {n_expected}"
        + ("" if complete else "  **INCOMPLETE COVERAGE**"),
        "worker import (s)": meta.get("import_s"),
        "problem setup (s)": meta.get("problem_setup_s"),
        "solver warm-up (s)": meta.get("warmup_s"),
        "timed solver loop (s)": tail.get("end", {}).get("loop_s"),
        "sum of per-run solver time (s)": df.elapsed_ms.sum() / 1e3,
        "nominal budget (s)": n_expected * max(budgets) / 1e3,
        "overall fits/s (loop)": len(df) / tail["end"]["loop_s"] if "end" in tail else None,
        "worker total (s)": tail.get("end", {}).get("worker_total_s"),
        "BLAS threadpools": ", ".join(
            f"{t['internal_api']}×{t['num_threads']}" for t in meta.get("threadpools", [])
        )
        or "not visible to threadpoolctl (e.g. Accelerate)",
        "thread env": ", ".join(f"{k}={v}" for k, v in meta.get("thread_env", {}).items()),
        "versions": f"python {meta.get('python')}, numpy {meta.get('numpy')}, scipy {meta.get('scipy')} ({meta.get('machine')})",
    }
    if run_info:
        inv["worker wall incl. spawn (s)"] = run_info["worker_wall_s"]
        inv["killed by watchdog"] = run_info["killed_by_watchdog"]
    if "truncated" in tail:
        inv["truncated by worker deadline"] = True
    inv_df = pd.DataFrame(
        {"value": {k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in inv.items()}}
    )
    sections.append(("Invocation & coverage", inv_df.to_markdown()))

    sections.append(("Success rate (pooled)", rep.pooled().map(_pct).to_markdown()))
    group_bys = [("group", "dataset/model"), ("p", "parameter count")]
    for col in sorted(c for c in df.columns if c.startswith("tag_")):
        if df[col].dtype == object and 2 <= df[col].nunique() <= 30:
            group_bys.append((col, col.removeprefix("tag_")))
    for by, title in group_bys:
        if by != "group" and not 2 <= df[by].nunique() <= 30:
            continue
        for tg in targets:
            sections.append(
                (f"Success {rep.lab(tg)} by {title}", rep.success(by, tg).map(_pct).to_markdown())
            )
    sections.append(
        ("Multi-start (best over starts, budgets added)", rep.multistart().map(_pct).to_markdown())
    )
    if (rk := rep.ranking()) is not None:
        sections.append(
            ("Candidate ranking (curves with several candidate equations)", rk.to_markdown())
        )
    if (pa := rep.parameter_accuracy()) is not None:
        sections.append(("Parameter accuracy vs reference parameters (LRE)", pa.to_markdown()))
    sections.append(("Outcomes (termination status counts)", rep.outcomes().to_markdown()))
    sections.append(
        ("Cost: time to target, evaluations, throughput", costs.to_markdown(floatfmt=".3g"))
    )
    for b in budgets:
        t = rep.checkpoint_losses(b)
        body = (
            t.to_markdown()
            if len(t) <= MAX_MD_ROWS
            else f"{len(t)} rows — see records.csv (`best_{b:g}ms`)."
        )
        sections.append((f"Best loss at {b:g} ms (per problem and start)", body))

    notes = []
    for label in sorted(df.loc[df.solver_label != df.solver, "solver_label"].unique()):
        grps = sorted(df.loc[df.solver_label == label, "group"].unique())
        notes.append(f"`{label}` on: {', '.join(grps)}.")
    notes.append(
        {
            "absolute": "Success: best loss − reference ≤ target.",
            "relative": f"Success: best loss − R ≤ target · max(R, {cfg.get('relative_floor', 1e-20):g}).",
            "reduction": "Success (Moré & Wild 2009): best loss − R ≤ target · (L0 − R), L0 = loss at the start.",
        }[mode]
    )
    modes = df.groupby("group").ref_mode.first()
    if (modes == "given_or_best").any():
        notes.append(
            "Reference = min(published/certified value, best loss found) for: "
            + ", ".join(modes[modes == "given_or_best"].index[:20])
            + (" …" if (modes == "given_or_best").sum() > 20 else "")
            + "."
        )
    if (modes == "best_found").any():
        notes.append(
            "Reference = best loss found by any solver/start (no known minimum) for: "
            + ", ".join(modes[modes == "best_found"].index[:20])
            + (" …" if (modes == "best_found").sum() > 20 else "")
            + ". Success there means 'matched the best found'."
        )
    if (modes == "none").any():
        notes.append(
            "Raw loss (reference 0) used for: " + ", ".join(modes[modes == "none"].index) + "."
        )
    if not complete:
        notes.append("Coverage is incomplete: missing runs are NOT counted as failures.")
    if "tag_regime" in df and df.tag_regime.notna().any():
        notes.append(
            "High-order synthetic mixtures (p ≥ 32) are deliberately poorly identifiable; see the regime tables."
        )
    notes.append(
        f"Total invocation time (including reporting): {time.perf_counter() - t_invocation:.2f} s."
    )

    md = [f"# eqnfitrs benchmark — {meta.get('experiment', '')}", ""]
    for title, body in sections:
        md += [f"## {title}", "", body, ""]
    md += ["## Notes", ""] + [f"- {n}" for n in notes]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    listy = ["hist_t_ms", "hist_loss", "theta_at", "best_theta", "ref_params"]
    df.drop(columns=[c for c in listy if c in df]).to_csv(out_dir / "records.csv", index=False)

    if plot:
        from .plotting import plot_success_vs_time

        plot_success_vs_time(
            df, out_dir, budgets, targets, rep.solvers, mode, cfg.get("relative_floor", 1e-20)
        )

    show = (
        "Invocation",
        "Success rate (pooled)",
        f"Success {rep.lab(targets[0])} by",
        "Candidate ranking",
        "Parameter accuracy",
        "Multi-start",
        "Outcomes",
        "Cost",
    )
    for title, body in sections:
        if title.startswith(show) and "parameter count" not in title:
            print(f"\n## {title}\n\n{body}")
    print("\n" + "\n".join(f"- {n}" for n in notes))
    print(
        f"\nWrote {out_dir}/summary.md, records.csv, records.jsonl"
        + (", success_vs_time.png" if plot else "")
    )
