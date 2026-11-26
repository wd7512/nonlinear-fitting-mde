"""The README figure: success against time for every solver, averaged over the six suites.

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
    "eqnfitrs": "eqnfitrs",
    "scipy_lm": "SciPy lm", "scipy_trf": "SciPy trf", "scipy_dogbox": "SciPy dogbox",
    "optx_lm": "Optimistix LM", "optx_indirect_lm": "Optimistix indirect LM",
    "optx_dogleg": "Optimistix dogleg", "optx_gauss_newton": "Optimistix Gauss–Newton",
    "ceres_lm": "Ceres LM", "sagefit": "SAGE-Fit",
    "varpropy": "VarProPy *", "varpro_rs": "varpro crate (Rust) *",
}  # fmt: skip
BUDGETS = (1, 5, 25)  # ms, tabulated
# colour = library (categorical slot), dash = solver within it
STYLE = {
    "eqnfitrs": (0, "-"),
    "scipy_lm": (1, "-"), "scipy_trf": (1, "--"), "scipy_dogbox": (1, ":"),
    "optx_lm": (2, "-"), "optx_indirect_lm": (2, "--"), "optx_dogleg": (2, ":"),
    "optx_gauss_newton": (2, "-."),
    "ceres_lm": (3, "-"), "sagefit": (4, "-"), "varpro_rs": (5, "-"), "varpropy": (5, "--"),
}  # fmt: skip
GRID = np.logspace(-1, np.log10(25), 300)  # ms; 25 ms is a checkpoint in every suite
# categorical slots (blue, orange, aqua, yellow, magenta, violet) and text inks, per mode
THEMES = {
    "light": dict(series=["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#6a4ce2"],
                  surface="#ffffff", text="#191919", muted="#666666", grid="#e6e6e6"),
    "dark": dict(series=["#338ced", "#e76624", "#00a888", "#d28e05", "#e24d8b", "#9887ef"],
                 surface="#0d1117", text="#efefef", muted="#9c9c9c", grid="#30363d"),
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


def draw(curves: dict[str, np.ndarray], path: Path, theme: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    i5 = min(np.searchsorted(GRID, 5), GRID.size - 1)
    for solver in sorted(curves, key=lambda s: -curves[s][i5]):  # legend in 5 ms order
        slot, dash = STYLE.get(solver, (0, ":"))
        top = solver == "eqnfitrs"
        ax.plot(GRID, curves[solver], color=theme["series"][slot], linestyle=dash,
                linewidth=2.6 if top else 1.6, label=LABELS.get(solver, solver),
                zorder=3 if top else 2)  # fmt: skip
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
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9,
              labelcolor=theme["text"], handlelength=2.6)  # fmt: skip
    fig.text(0.01, 0.01, "* variable projection only: misses every problem without a separable form",
             color=theme["muted"], fontsize=8)  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200, facecolor=theme["surface"])
    plt.close(fig)


def main(results: Path = Path("results"), out: Path = Path("docs")) -> None:
    per_suite = [suite_curves(s, results) for s in SUITES]
    solvers = [s for s in per_suite[0] if all(s in c for c in per_suite)]
    mean = {s: np.mean([c[s] for c in per_suite], axis=0) for s in solvers}
    for mode, theme in THEMES.items():
        draw(mean, out / f"benchmark-{mode}.png", theme)
    idx = [min(np.searchsorted(GRID, b), GRID.size - 1) for b in BUDGETS]
    print("| solver | " + " | ".join(f"{b} ms" for b in BUDGETS) + " |\n|:--|--:|--:|--:|")
    for s in sorted(solvers, key=lambda s: -mean[s][idx[1]]):
        print(f"| {LABELS.get(s, s)} | " + " | ".join(f"{mean[s][i]:.1%}" for i in idx) + " |")


if __name__ == "__main__":
    main()
