"""Success-rate-vs-time curves (built outside the timed solver loop)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .scoring import best_so_far_curve, target_label, tolerance  # noqa: E402

# Validated categorical palette, fixed slot order (light surface).
COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#6a4ce2"]
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"


def _curves(df: pd.DataFrame, grid: np.ndarray) -> dict[str, np.ndarray]:
    return {
        s: np.vstack(
            [
                best_so_far_curve(np.asarray(t, float), np.asarray(ls, float), grid)
                for t, ls in zip(g.hist_t_ms, g.hist_loss, strict=True)
            ]
        )
        for s, g in df.groupby("solver")
    }


def _style(ax, title: str, budgets) -> None:
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, color=TEXT, fontsize=10, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    for b in budgets:
        ax.axvline(b, color=MUTED, linewidth=0.6, linestyle=":")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))


def plot_success_vs_time(
    df: pd.DataFrame,
    out_dir: Path,
    budgets,
    targets,
    solvers,
    mode: str = "absolute",
    floor: float = 1e-20,
    max_groups: int = 6,
) -> Path:
    """Pooled panels (one per target) + one panel per dataset/model group (first target)."""
    grid = np.logspace(-1, np.log10(max(budgets)), 300)
    groups = list(df.group.unique())[:max_groups]
    panels = [(None, tg) for tg in targets]
    panels += [(None, None)] * (-len(panels) % 3)  # legend slot(s) to finish the first row
    if len(groups) > 1:
        panels += [(g, targets[0]) for g in groups]
    ncols = 3
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(13, 3.8 * nrows), sharex=True, sharey=True, squeeze=False
    )
    flat = list(axes.flat)
    legend_ax = None
    for i, ax in enumerate(flat):
        if i >= len(panels) or panels[i][1] is None:
            ax.axis("off")
            legend_ax = legend_ax or (ax if i < len(panels) else None)
            continue
        grp, tg = panels[i]
        sub = df if grp is None else df[df.group == grp]
        curves = _curves(sub, grid)
        sub = sub.assign(_tol=tolerance(sub, tg, mode, floor))
        refs = sub.groupby("solver").ref.apply(lambda r: r.to_numpy()[:, None])
        tols = sub.groupby("solver")._tol.apply(lambda r: r.to_numpy()[:, None])
        for k, s in enumerate(solvers):
            if s in curves:
                ax.plot(
                    grid,
                    ((curves[s] - refs[s]) <= tols[s]).mean(axis=0),
                    color=COLORS[k % len(COLORS)],
                    linewidth=2,
                    label=s,
                )
        _style(
            ax,
            f"{'All problems' if grp is None else grp} — excess {target_label(tg, mode)}",
            budgets,
        )
    handles, labels = flat[0].get_legend_handles_labels()
    (legend_ax or flat[0]).legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        fontsize=10,
        labelcolor=TEXT,
        title="Solver",
        title_fontsize=10,
    )
    for ax in axes[-1]:
        ax.set_xlabel("wall-clock time per run (ms, log)", color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of runs at target", color=MUTED, fontsize=9)
    marks = ", ".join(f"{b:g}" for b in budgets)
    fig.suptitle(
        f"Success rate vs time budget (dotted lines: {marks} ms checkpoints)",
        color=TEXT,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    path = out_dir / "success_vs_time.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path
