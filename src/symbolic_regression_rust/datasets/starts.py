"""Starting points for table-dataset models."""

from __future__ import annotations

import numpy as np


def bounds_arrays(spec, names: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    if not spec.bounds:
        return None
    if missing := [n for n in names if n not in spec.bounds]:
        raise ValueError(f"starts.bounds missing parameters {missing}")
    lo = np.array([spec.bounds[n][0] for n in names], float)
    hi = np.array([spec.bounds[n][1] for n in names], float)
    if np.any(hi < lo):
        raise ValueError("starts.bounds: upper < lower")
    return lo, hi


def make_starts(
    spec, names: list[str], rng: np.random.Generator, ref_params: np.ndarray | None
) -> dict[str, np.ndarray]:
    p = len(names)
    starts: dict[str, np.ndarray] = {}
    for key, vals in spec.values.items():
        if len(vals) != p:
            raise ValueError(f"start {key!r} has {len(vals)} values, model has {p} params {names}")
        starts[key] = np.asarray(vals, float)
    box = bounds_arrays(spec, names)
    if spec.n_random:
        if box is None:
            raise ValueError("starts.n_random needs starts.bounds")
        for i in range(spec.n_random):
            starts[f"random{i}"] = rng.uniform(*box)
    if spec.near_scale is not None:
        if ref_params is None or box is None:
            raise ValueError("starts.near_scale needs reference.params and starts.bounds")
        starts["near"] = ref_params + spec.near_scale * (box[1] - box[0]) * rng.standard_normal(p)
    if not starts:
        raise ValueError("no starts: give starts.values, or starts.bounds + n_random")
    return starts
