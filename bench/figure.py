"""The README figure: success against time, averaged over the six suites.

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

SOLVERS = ["symbfitrs", "scipy_trf", "scipy_lm"]
GRID = np.logspace(-1, np.log10(25), 300)  # ms; 25 ms is a checkpoint in every suite
# categorical slots 1-3 and text inks, per mode (validated light and dark)
THEMES = {
    "light": dict(series=["#257de0", "#f47230", "#00b894"], surface="#ffffff",
                  text="#191919", muted="#666666", grid="#e6e6e6"),
    "dark": dict(series=["#338ced", "#e76624", "#00a888"], surface="#0d1117",
                 text="#efefef", muted="#9c9c9c", grid="#30363d"),
}  # fmt: skip


def suite_curves(suite: str, results: Path) -> dict[str, np.ndarray]:
    """Share of runs at the suite's first target by each grid time, per solver."""
    from .experiment import load_config
    from .report import load_records
    from .scoring import best_so_far_curve, fill_reference, tolerance

    cfg = load_config(CONFIG_DIR / f"{suite}.toml")
    _, fits, _ = load_records(results / f"{suite}_rust" / "records.jsonl")
    df = fill_reference(fits[fits.solver.isin(SOLVERS)])
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


def draw(curves: dict[str, np.ndarray], path: Path, theme: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    for solver, color in zip(SOLVERS, theme["series"], strict=True):
        ax.plot(GRID, curves[solver], color=color, linewidth=2, label=solver)
    inner = slice(len(GRID) // 8, -len(GRID) // 4)  # clear of the axis and the legend
    for solver in SOLVERS:  # label each line where it is furthest from the others
        y = curves[solver]
        gaps = np.vstack([y - curves[o] for o in SOLVERS if o != solver])
        nearest = gaps[np.abs(gaps).argmin(0), np.arange(len(GRID))]
        i = inner.start + np.abs(nearest[inner]).argmax()
        up = nearest[i] > 0  # success only rises, so above-left and below-right are clear
        ax.annotate(solver, (GRID[i], y[i]), xytext=(-5, 5) if up else (5, -5),
                    textcoords="offset points", ha="right" if up else "left",
                    va="bottom" if up else "top", fontsize=9,
                    color=theme["text"])  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlim(GRID[0], GRID[-1])
    ax.set_ylim(0, 1)
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 25], ["0.1", "0.3", "1", "3", "10", "25 ms"])
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(True, axis="y", color=theme["grid"], linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["muted"])
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    ax.set_xlabel("time per fit (log scale)", color=theme["muted"], fontsize=9)
    ax.set_title("Fits that reach the target, mean of six benchmark suites", loc="left",
                 color=theme["text"], fontsize=11, pad=12)  # fmt: skip
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=theme["text"])
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=theme["surface"])
    plt.close(fig)


def main(results: Path = Path("results"), out: Path = Path("docs")) -> None:
    per_suite = [suite_curves(s, results) for s in SUITES]
    mean = {s: np.mean([c[s] for c in per_suite], axis=0) for s in SOLVERS}
    for mode, theme in THEMES.items():
        draw(mean, out / f"benchmark-{mode}.png", theme)
    for b in (1, 5, 25):
        i = np.searchsorted(GRID, b) - (b == 25)
        print(f"{b:>2} ms: " + "  ".join(f"{s} {mean[s][i]:.1%}" for s in SOLVERS))


if __name__ == "__main__":
    main()
