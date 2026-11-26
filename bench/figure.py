"""The README figure: share of fits reaching the target at 1, 5 and 25 ms, mean of six suites,
one row per solver.

  uv run python -m bench.figure    # after `bench run SUITE` and `bench run SUITE --fitter`

Reads ``results/<suite>_rust/records.jsonl`` (fitter and SciPy records merged) and judges each
suite by its own first target and success rule. Writes a light and a dark PNG to ``docs/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import CONFIG_DIR  # noqa: E402
from .gate import SUITES  # noqa: E402

LABELS = {  # display names; * marks the separable-only solvers
    "symbfitrs": "symbfitrs",
    "scipy_lm": "SciPy lm", "scipy_trf": "SciPy trf", "scipy_dogbox": "SciPy dogbox",
    "optx_lm": "Optimistix LM", "optx_indirect_lm": "Optimistix indirect LM",
    "optx_dogleg": "Optimistix dogleg", "optx_gauss_newton": "Optimistix Gauss–Newton",
    "ceres_lm": "Ceres LM", "sagefit": "SAGE-Fit",
    "varpropy": "VarProPy *", "varpro_rs": "varpro crate (Rust) *",
}  # fmt: skip
BUDGETS = (1, 5, 25)  # ms, one marker each
GRID = np.logspace(-1, np.log10(25), 300)  # ms; 25 ms is a checkpoint in every suite
# categorical slots 1-3 (one per budget) and text inks, per mode (validated light and dark)
THEMES = {
    "light": dict(series=["#257de0", "#f47230", "#00b894"], surface="#ffffff",
                  text="#191919", muted="#666666", grid="#e6e6e6"),
    "dark": dict(series=["#338ced", "#e76624", "#00a888"], surface="#0d1117",
                 text="#efefef", muted="#9c9c9c", grid="#30363d"),
}  # fmt: skip


def suite_curves(suite: str, results: Path) -> dict[str, np.ndarray]:
    """Share of runs at the suite's first target by each grid time, per solver in the records."""
    from .experiment import load_config
    from .report import load_records
    from .scoring import best_so_far_curve, fill_reference, tolerance

    cfg = load_config(CONFIG_DIR / f"{suite}.toml")
    _, fits, _ = load_records(results / f"{suite}_rust" / "records.jsonl")
    df = fill_reference(fits)
    df["tol"] = tolerance(df, cfg.targets[0], cfg.success, cfg.relative_floor)
    out = {}
    for solver, g in df.groupby("solver"):
        best = np.vstack(
            [
                best_so_far_curve(np.asarray(t, float), np.asarray(ls, float), GRID)
                for t, ls in zip(g.hist_t_ms, g.hist_loss, strict=True)
            ]
        )
        out[solver] = ((best - g.ref.to_numpy()[:, None]) <= g.tol.to_numpy()[:, None]).mean(0)
    return out


def draw(at: dict[str, list[float]], path: Path, theme: dict) -> None:
    order = sorted(at, key=lambda s: at[s][1])  # by 5 ms success, best at the top
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(order) + 1.3))
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    for row, s in enumerate(order):
        ax.plot([min(at[s]), max(at[s])], [row, row], color=theme["grid"], linewidth=2, zorder=1)
    for k, (b, color) in enumerate(zip(BUDGETS, theme["series"], strict=True)):
        ax.scatter([at[s][k] for s in order], range(len(order)), s=55, color=color, zorder=2,
                   edgecolors=theme["surface"], linewidths=1.5, label=f"{b} ms")  # fmt: skip
    ax.set_yticks(range(len(order)), [LABELS.get(s, s) for s in order])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(True, axis="x", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    for tick, s in zip(ax.get_yticklabels(), order, strict=True):
        tick.set_color(theme["text"])
        tick.set_fontweight("bold" if s == "symbfitrs" else "normal")
    ax.set_title("Fits that reach the target, mean of six benchmark suites",
                 loc="left", color=theme["text"], fontsize=11, pad=24)  # fmt: skip
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1), ncol=3, fontsize=9,
              labelcolor=theme["text"], handletextpad=0.2, borderaxespad=0.1)  # fmt: skip
    fig.text(0.01, 0.01, "* variable projection only: misses every problem without a separable form",
             color=theme["muted"], fontsize=8)  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200, facecolor=theme["surface"])
    plt.close(fig)


def main(results: Path = Path("results"), out: Path = Path("docs")) -> None:
    per_suite = [suite_curves(s, results) for s in SUITES]
    solvers = [s for s in per_suite[0] if all(s in c for c in per_suite)]
    mean = {s: np.mean([c[s] for c in per_suite], axis=0) for s in solvers}
    idx = [min(np.searchsorted(GRID, b), GRID.size - 1) for b in BUDGETS]
    at = {s: [float(mean[s][i]) for i in idx] for s in solvers}
    for mode, theme in THEMES.items():
        draw(at, out / f"benchmark-{mode}.png", theme)
    print("| solver | " + " | ".join(f"{b} ms" for b in BUDGETS) + " |\n|:--|--:|--:|--:|")
    for s in sorted(solvers, key=lambda s: -at[s][1]):
        print(f"| {LABELS.get(s, s)} | " + " | ".join(f"{v:.1%}" for v in at[s]) + " |")


if __name__ == "__main__":
    main()
