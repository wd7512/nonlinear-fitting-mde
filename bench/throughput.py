"""How fast a queue of fits clears with N workers, and whether the results stay the same.

  uv run python -m bench.fitter.threads --budget-ms 100 --kind throughput   # the run
  uv run python -m bench.throughput                                         # table + figure

Reads ``results/<suite>_throughput/records.jsonl``: every fit of every suite is queued on
``Pool(threads=N)``. Time is the wall time to finish the whole queue (summed over suites);
agreement compares each fit's final loss with the same fit's N=1 result.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .figure import THEMES, THREAD_RAMPS  # noqa: E402
from .gate import SUITES  # noqa: E402

RTOL, ATOL = 1e-6, 1e-12  # losses agree if |a - b| <= ATOL + RTOL * max(a, b)


def load(results: Path) -> tuple[dict[int, float], dict[int, dict]]:
    """(total wall seconds per N, {N: {(problem, start): record}})."""
    wall: dict[int, float] = {}
    fits: dict[int, dict] = {}
    for s in SUITES:
        path = results / f"{s}_throughput" / "records.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["type"] == "meta":
                for n, w in r["wall_s"].items():
                    wall[int(n)] = wall.get(int(n), 0.0) + w
            else:
                n = int(r["solver"].rpartition("_t")[2])
                fits.setdefault(n, {})[(r["problem"], r["start"])] = r
    return wall, fits


def agreement(fits: dict[int, dict]) -> dict[int, dict[str, float]]:
    """Per N: share of fits whose loss matches N=1, split by how the N=1 fit ended."""
    base = fits[1]
    out = {}
    for n, rs in sorted(fits.items()):
        same = {"all": [], "settled": [], "deadline": []}
        for k, b in base.items():
            a = rs[k]
            ok = abs(a["best_loss"] - b["best_loss"]) <= ATOL + RTOL * max(
                a["best_loss"], b["best_loss"]
            )
            same["all"].append(ok)
            both = b["status"] != "deadline" and a["status"] != "deadline"
            same["settled" if both else "deadline"].append(ok)
        out[n] = {k: (float(np.mean(v)) if v else float("nan")) for k, v in same.items()}
        out[n]["n_deadline"] = len(same["deadline"])
        out[n]["worst_rel"] = max(
            abs(rs[k]["best_loss"] - b["best_loss"]) / max(b["best_loss"], ATOL)
            for k, b in base.items()
            if b["status"] != "deadline" and rs[k]["status"] != "deadline"
        )
    return out


def draw(wall: dict[int, float], path: Path, theme: dict, mode: str, n_fits: int) -> None:
    ns = sorted(wall)
    color = THREAD_RAMPS[mode][-1]  # the N=1 blue
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    ax.plot(ns, [wall[1] / wall[n] for n in ns], color=color, linewidth=2.6, marker="o",
            markersize=7, label="symbfitrs", zorder=3)  # fmt: skip
    ax.plot(ns, ns, color=theme["muted"], linewidth=1.6, linestyle="--", label="ideal")
    for n in ns:
        ax.annotate(f"{wall[n]:.0f} s", (n, wall[1] / wall[n]), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=9, color=theme["text"])  # fmt: skip
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlim(ns[0] - 0.3, ns[-1] + 0.3)
    ax.set_ylim(0, ns[-1] + 0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.grid(True, axis="y", color=theme["grid"], linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["muted"])
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    ax.set_xlabel("workers", color=theme["muted"], fontsize=9)
    ax.set_title(f"Speed-up in clearing a queue of {n_fits:,} fits, by number of workers"
                 " (wall time under each point)", loc="left", color=theme["text"],
                 fontsize=11, pad=12)  # fmt: skip
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9,
              labelcolor=theme["text"], handlelength=2.6)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=theme["surface"])
    plt.close(fig)


def main(results: Path = Path("results"), out: Path = Path("docs")) -> None:
    wall, fits = load(results)
    n_fits = len(fits[1])
    for mode, theme in THEMES.items():
        draw(wall, out / f"benchmark-throughput-{mode}.png", theme, mode, n_fits)
    agree = agreement(fits)
    print(f"{n_fits} fits queued per N\n")
    print("| workers | wall time | speed-up | loss matches N=1 (all) | (settled in both) |")
    print("|--:|--:|--:|--:|--:|")
    for n in sorted(wall):
        a = agree[n]
        print(f"| {n} | {wall[n]:.1f} s | {wall[1] / wall[n]:.2f}× | {a['all']:.2%} "
              f"| {a['settled']:.2%} (worst rel. diff {a['worst_rel']:.1e}) |")  # fmt: skip
    print(f"\nfits that hit the deadline at N=1 or N: {agree[max(agree)]['n_deadline']}")


if __name__ == "__main__":
    main()
