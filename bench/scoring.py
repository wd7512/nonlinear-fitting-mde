"""Turn raw per-run histories into checkpoint scores.

Success at a checkpoint compares the best loss so far L with a reference R:

  absolute   L - R <= target
  relative   L - R <= target * max(R, floor)
  reduction  L - R <= target * (L0 - R)   (More & Wild 2009; L0 = loss at the start)

R is the known minimum: 0 for noise-free synthetic data, a certified/given value,
the best loss any solver/start found (``best_found``), or the smaller of a published
value and the best found (``given_or_best``, for local-minimum-prone problems).
A loss is credited at checkpoint B only if it became available at t <= B, so late
results (including native-call overruns) are never credited at earlier checkpoints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def best_at(t: np.ndarray, loss: np.ndarray, budget_ms: float) -> float:
    mask = t <= budget_ms
    return float(loss[mask].min()) if mask.any() else np.inf


def time_to_target(t: np.ndarray, loss: np.ndarray, threshold: float, cap_ms: float) -> float:
    hit = np.flatnonzero((loss <= threshold) & (t <= cap_ms))
    return float(t[hit[0]]) if hit.size else np.nan


def best_so_far_curve(t: np.ndarray, loss: np.ndarray, grid_ms: np.ndarray) -> np.ndarray:
    """Best loss available by each grid time (inf before the first evaluation)."""
    if t.size == 0:
        return np.full(grid_ms.size, np.inf)
    running = np.minimum.accumulate(loss)
    idx = np.searchsorted(t, grid_ms, side="right") - 1
    return np.where(idx >= 0, running[np.clip(idx, 0, None)], np.inf)


def ok_col(target: float, budget: float) -> str:
    return f"ok_{target:.0e}_{budget:g}ms"


def fill_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric ``ref`` column from ``ref_mode``."""
    df = df.copy()
    given = pd.to_numeric(df.ref_loss, errors="coerce")
    best = df.groupby("problem").best_loss.transform("min")
    ref = given.copy()
    ref = ref.where(df.ref_mode != "best_found", best)
    ref = ref.where(df.ref_mode != "given_or_best", np.fmin(given, best))
    df["ref"] = ref.where(df.ref_mode != "none", 0.0).fillna(0.0)
    return df


def tolerance(
    df: pd.DataFrame, target: float, mode: str = "absolute", floor: float = 1e-20
) -> pd.Series:
    """Allowed excess L - R for each run."""
    if mode == "absolute":
        return pd.Series(target, index=df.index)
    if mode == "relative":
        return target * np.maximum(df.ref, floor)
    if mode == "reduction":
        l0 = pd.to_numeric(df.get("loss0"), errors="coerce")
        return target * (l0 - df.ref).clip(lower=0).fillna(np.inf)
    raise ValueError(f"unknown success mode {mode!r}")


def target_label(target: float, mode: str = "absolute") -> str:
    return {
        "absolute": f"≤{target:.0e}",
        "relative": f"≤{target:.0e}·R",
        "reduction": f"≤{target:.0e}·(L0−R)",
    }[mode]


def score(
    fits: pd.DataFrame,
    budgets: list[float],
    targets: list[float],
    mode: str = "absolute",
    floor: float = 1e-20,
) -> pd.DataFrame:
    """Add best-at-checkpoint, success flags and time-to-target columns."""
    df = fill_reference(fits)
    cap = max(budgets)
    hist = [
        (np.asarray(t, float), np.asarray(ls, float))
        for t, ls in zip(df.hist_t_ms, df.hist_loss, strict=True)
    ]
    tol = {tg: tolerance(df, tg, mode, floor) for tg in targets}
    for b in budgets:
        df[f"best_{b:g}ms"] = [best_at(t, ls, b) for t, ls in hist]
        for tg in targets:
            df[ok_col(tg, b)] = (df[f"best_{b:g}ms"] - df.ref) <= tol[tg]
    for tg in targets:
        df[f"ttt_{tg:.0e}"] = [
            time_to_target(t, ls, r + e, cap)
            for (t, ls), r, e in zip(hist, df.ref, tol[tg], strict=True)
        ]
    df["had_nonfinite"] = df.n_nonfinite > 0
    df["group"] = df.dataset + "/" + df.model
    return df


def ranking_scores(df: pd.DataFrame, budgets: list[float]) -> pd.DataFrame | None:
    """Candidate-equation ranking quality per solver, for curves fitted by >= 2 models.

    Truth: candidates ordered by their reference loss. Each solver's ordering uses its best
    loss over starts at the checkpoint. Columns: (metric, budget) with metric in "tau" (mean
    Kendall tau), "top1" (picked the truly best candidate) and "lost" (share of candidates
    with loss > 3 x reference, as in SAGE-Fit).
    """
    from scipy.stats import kendalltau

    n_models = df.groupby(["dataset", "curve"]).model.transform("nunique")
    sub = df[n_models >= 2]
    if sub.empty:
        return None
    rows: dict = {}
    for b in budgets:
        per = sub.groupby(["solver", "dataset", "curve", "model"]).agg(
            L=(f"best_{b:g}ms", "min"), R=("ref", "first")
        )
        for s, g in per.groupby(level="solver"):
            taus, top1 = [], []
            for _, c in g.groupby(level=["dataset", "curve"]):
                L = c.L.replace([np.inf, -np.inf], np.nan).fillna(np.finfo(float).max)
                if c.R.nunique() > 1:
                    tau = kendalltau(c.R, L).statistic
                    taus.append(0.0 if np.isnan(tau) else tau)
                top1.append(L.idxmin() == c.R.idxmin())
            row = rows.setdefault(s, {})
            row["tau", b] = np.mean(taus) if taus else np.nan
            row["top1", b] = np.mean(top1)
            row["lost", b] = (g.L > 3 * g.R).mean()
    return pd.DataFrame(rows).T
