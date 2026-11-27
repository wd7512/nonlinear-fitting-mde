"""The README figure: success against time for every solver, averaged over the six suites.

  uv run python -m bench.figure    # after `bench run SUITE` and `bench run SUITE --fitter`
  uv run python -m bench.figure --threads    # after `python -m bench.fitter.threads`

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
BUDGETS = (1, 5, 25)  # ms, tabulated
# colour = library (categorical slot), dash = solver within it
STYLE = {
    "symbfitrs": (0, "-"),
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


def suite_curves(suite: str, results: Path, kind: str = "rust") -> dict[str, np.ndarray]:
    """Share of runs at the suite's first target by each grid time, per solver in the records
    of ``results/<suite>_<kind>``."""
    from .experiment import load_config
    from .report import load_records
    from .scoring import best_so_far_curve, fill_reference, tolerance

    cfg = load_config(CONFIG_DIR / f"{suite}.toml")
    _, fits, _ = load_records(results / f"{suite}_{kind}" / "records.jsonl")
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
        top = solver == "symbfitrs"
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


# one hue, light to dark: the more fits at once, the lighter; one at a time is the darkest
THREAD_RAMPS = {  # lightest (most in parallel) first
    "light": ["#84b9f3", "#519ced", "#2370cd", "#0f4a96"],
    "dark": ["#a8cdf8", "#6aa8f2", "#338ced", "#1d6fd0"],
}


NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def thread_count(solver: str) -> int:
    return int(solver.rpartition("_t")[2])


def draw_threads(
    curves: dict[str, np.ndarray], path: Path, theme: dict, mode: str, n_suites: int
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    counts = sorted(curves, key=thread_count)  # 1 first (darkest)
    ramp = THREAD_RAMPS[mode][::-1][: len(counts)]  # darkest first
    for solver, color in zip(counts, ramp, strict=False):
        n = thread_count(solver)
        ax.plot(GRID, curves[solver], color=color, linestyle="-", linewidth=1.6,
                label="1 at a time" if n == 1 else f"{n} in parallel")  # fmt: skip
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
    ax.set_title("symbfitrs fits that reach the target, with N fits running in parallel,"
                 f" mean of {NUMBER_WORDS[n_suites]} benchmark suites", loc="left", color=theme["text"],
                 fontsize=11, pad=12)  # fmt: skip
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9,
              labelcolor=theme["text"], handlelength=2.6)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=theme["surface"])
    plt.close(fig)


def table(mean: dict[str, np.ndarray], label) -> None:
    idx = [min(np.searchsorted(GRID, b), GRID.size - 1) for b in BUDGETS]
    print("| solver | " + " | ".join(f"{b} ms" for b in BUDGETS) + " |\n|:--|--:|--:|--:|")
    for s in sorted(mean, key=lambda s: -mean[s][idx[1]]):
        print(f"| {label(s)} | " + " | ".join(f"{mean[s][i]:.1%}" for i in idx) + " |")


def main_threads(results: Path = Path("results"), out: Path = Path("docs")) -> None:
    have = [s for s in SUITES if (results / f"{s}_threads" / "records.jsonl").exists()]
    per_suite = [suite_curves(s, results, "threads") for s in have]
    solvers = [s for s in per_suite[0] if all(s in c for c in per_suite)]
    mean = {s: np.mean([c[s] for c in per_suite], axis=0) for s in solvers}
    for mode, theme in THEMES.items():
        draw_threads(mean, out / f"benchmark-threads-{mode}.png", theme, mode, len(have))
    print(f"suites: {', '.join(have)}")
    table(
        mean, lambda s: "1 at a time" if thread_count(s) == 1 else f"{thread_count(s)} in parallel"
    )


def main(results: Path = Path("results"), out: Path = Path("docs")) -> None:
    per_suite = [suite_curves(s, results) for s in SUITES]
    solvers = [s for s in per_suite[0] if all(s in c for c in per_suite)]
    mean = {s: np.mean([c[s] for c in per_suite], axis=0) for s in solvers}
    for mode, theme in THEMES.items():
        draw(mean, out / f"benchmark-{mode}.png", theme)
    table(mean, lambda s: LABELS.get(s, s))


if __name__ == "__main__":
    import sys

    main_threads() if "--threads" in sys.argv[1:] else main()
