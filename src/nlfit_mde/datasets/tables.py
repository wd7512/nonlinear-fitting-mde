"""Read curves from tabular files (CSV/TSV/whitespace/Parquet/Feather/JSON/NPZ).

Two layouts:
  long  one row per observation: x column(s), y, optional curve id and sigma
  wide  one x column set shared by all curves, one y column per curve
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_EXT = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "whitespace",
    ".dat": "whitespace",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".feather": "feather",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".npz": "npz",
}


@dataclass
class Curve:
    curve_id: str
    x: np.ndarray  # (m, d)
    y: np.ndarray  # (m,)
    sigma: np.ndarray | None


def read_table(path: Path, fmt: str = "auto", **opts) -> pd.DataFrame:
    if fmt == "auto":
        fmt = _EXT.get(path.suffix.lower(), "csv")
    if fmt == "csv":
        return pd.read_csv(path, **opts)
    if fmt == "tsv":
        return pd.read_csv(path, sep="\t", **opts)
    if fmt == "whitespace":
        return pd.read_csv(path, sep=r"\s+", **opts)
    if fmt == "parquet":
        return pd.read_parquet(path, **opts)
    if fmt == "feather":
        return pd.read_feather(path, **opts)
    if fmt in ("json", "jsonl"):
        return pd.read_json(path, lines=fmt == "jsonl", **opts)
    if fmt == "npz":
        with np.load(path, **opts) as z:
            return pd.DataFrame({k: np.ravel(z[k]) for k in z.files})
    raise ValueError(f"unsupported format {fmt!r}")


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    if missing := [c for c in cols if c not in df.columns]:
        raise ValueError(
            f"dataset {name!r}: columns {missing} not found; available: {list(df.columns)}"
        )


def iter_curves(spec, rng: np.random.Generator | None = None) -> Iterator[Curve]:
    df = read_table(spec.path, spec.format, **spec.read_options)
    if spec.layout == "wide":
        _require(df, spec.x, spec.name)
        ycols = spec.y_columns or [c for c in df.columns if c not in spec.x]
        _require(df, ycols, spec.name)
        groups = [(str(c), df[[*spec.x, c]].rename(columns={c: "__y"})) for c in ycols]
        ycol, scol = "__y", None
    else:
        cols = [*spec.x, spec.y] + [c for c in (spec.curve, spec.sigma) if c]
        _require(df, cols, spec.name)
        ycol, scol = spec.y, spec.sigma
        groups = (
            [(str(k), g) for k, g in df.groupby(spec.curve, sort=False)]
            if spec.curve
            else [("0", df)]
        )

    if spec.max_curves is not None and len(groups) > spec.max_curves:
        if spec.curve_sample == "random":
            rng = rng or np.random.default_rng(0)
            idx = np.sort(rng.choice(len(groups), spec.max_curves, replace=False))
            groups = [groups[i] for i in idx]
        else:
            groups = groups[: spec.max_curves]

    for cid, g in groups:
        g = g.dropna(subset=[*spec.x, ycol] + ([scol] if scol else []))
        if g.empty:
            continue
        sigma = g[scol].to_numpy(float) if scol else None
        if sigma is not None and np.any(sigma <= 0):
            raise ValueError(f"dataset {spec.name!r}, curve {cid}: sigma must be > 0")
        yield Curve(cid, g[spec.x].to_numpy(float), g[ycol].to_numpy(float), sigma)
