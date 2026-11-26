"""Stringer, Michaelos & Pachitariu (2019) V1 drifting-gratings recording x ten candidate
orientation-tuning equations.

Not a benchmark: the true equation is unknown. Each cell is one curve (response vs. grating
angle theta over all its trials) and every candidate equation is fitted to it; the reference
is the best loss any solver found (``best_found``), so results measure how far and how fast
each solver drives the MSE down, and how candidate equations rank per cell.

Data: ``data/external/stringer2019/gratings.npz`` from ``scripts/fetch_benchmarks.py
stringer2019`` (orientation-tuned cells, responses scaled to norm 100). ``wrap`` maps an angle
to (-pi, pi]; ``step(d)`` = [d >= 0] selects left/right widths.
Parameters are unconstrained here (the paper model's clips are not applied).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from ..models.symbolic import compile_sympy, step, wrap

P = "pi"
EQUATIONS: dict[str, tuple[list[str], str]] = {
    "idea_148": (
        ["b", "a1", "a2", "m", "s", "r1", "q1", "kp", "r2", "q2", "ko"],
        f"b + a1*exp(-0.5*(wrap(theta-m)/s)**2)*(1 + r1*tanh(wrap(theta-m)/s - q1)*(0.5 + 0.5*kp))"
        f" + a2*exp(-0.5*(wrap(theta-m-{P})/s)**2)*(1 + r2*tanh(wrap(theta-m-{P})/s - q2)*(0.5 + 0.5*ko))",
    ),
    "idea_125": (
        ["b", "a1", "a2", "m", "s", "r1", "g1", "r2", "g2"],
        "b + a1*exp(-0.5*(wrap(theta-m)/s)**2)*(1 + r1*sin(wrap(theta-m)/s) + g1*sin(2*wrap(theta-m)/s))"
        f" + a2*exp(-0.5*(wrap(theta-m-{P})/s)**2)*(1 + r2*sin(wrap(theta-m-{P})/s)"
        f" + g2*sin(2*wrap(theta-m-{P})/s))",
    ),
    "paper": (
        ["b", "a1", "a2", "m", "w1l", "w1r", "w2l", "w2r", "p1", "p2", "dlt"],
        "b + a1*exp(-0.5*(Abs(wrap(theta-m))/(w1l + (w1r-w1l)*step(wrap(theta-m))))**p1)"
        f" + a2*exp(-0.5*(Abs(wrap(theta-m-{P}-dlt))"
        f"/(w2l + (w2r-w2l)*step(wrap(theta-m-{P}-dlt))))**p2)",
    ),
    "idea_88": (
        ["b", "a1", "a2", "mu1", "mu2", "s", "c", "q", "h", "phi"],
        "b + a1*exp(-0.5*(wrap(theta-mu1)/(s + c*tanh(wrap(theta-mu1)/q)))**2)"
        " + a2*exp(-0.5*(wrap(theta-mu2)/(s + c*tanh(wrap(theta-mu2)/q)))**2) + h*cos(theta-phi)",
    ),
    "double_gaussian": (
        ["b", "a1", "a2", "m", "s"],
        f"b + a1*exp(-0.5*(wrap(theta-m)/s)**2) + a2*exp(-0.5*(wrap(theta-m-{P})/s)**2)",
    ),
    "idea_35": (
        ["b", "a1", "a2", "m", "k", "r", "s", "h", "phi"],
        "b + a1*exp(k*(cos(wrap(theta-m))-1))*(1 + r*tanh(wrap(theta-m)/s))"
        f" + a2*exp(k*(cos(wrap(theta-m-{P}))-1))*(1 - r*tanh(wrap(theta-m-{P})/s)) + h*cos(theta-phi)",
    ),
    "idea_132": (
        ["b", "c", "h", "A1", "A2", "m", "d1", "d2", "w1", "w2"],
        "b + c*cos(2*theta - h)"
        " + 4*A1/(exp(wrap(theta-m-d1)/w1) + exp(-wrap(theta-m-d1)/w1))**2"
        f" + 4*A2/(exp(wrap(theta-m-{P}-d2)/w2) + exp(-wrap(theta-m-{P}-d2)/w2))**2",
    ),
    "idea_138": (
        ["b", "a1", "a2", "mu1", "mu2", "s", "k", "h", "phi"],
        "b + a1*exp(-0.5*(wrap(theta-mu1)/(s*exp(k*sin(wrap(theta-mu1)))))**2)"
        " + a2*exp(-0.5*(wrap(theta-mu2)/(s*exp(-k*sin(wrap(theta-mu2)))))**2) + h*cos(theta-phi)",
    ),
    "idea_60": (
        ["b", "a1", "a2", "m", "s1", "s2", "ar", "mr", "dr", "sr"],
        f"b + a1*exp(-0.5*(wrap(theta-m)/s1)**2) + a2*exp(-0.5*(wrap(theta-m-{P})/s2)**2)"
        " + ar*exp(-0.5*((wrap(theta-mr)-dr)/sr)**2)",
    ),
    "idea_37": (
        ["b", "a1", "a2", "m", "s", "p", "r", "w", "h", "phi"],
        "b + a1/(1 + (2*sin(Abs(wrap(theta-m))/2)/s)**p)*(1 + r*tanh(sin(wrap(theta-m))/w))"
        f" + a2/(1 + (2*sin(Abs(wrap(theta-m-{P}))/2)/s)**p)*(1 - r*tanh(sin(wrap(theta-m-{P}))/w))"
        " + h*cos(theta-phi)",
    ),
}


@dataclass
class Peaks:
    b: float
    a1: float
    a2: float
    m: float
    s: float = 0.5


def estimate_peaks(theta: np.ndarray, y: np.ndarray, bins: int = 16) -> Peaks:
    """Coarse, equation-agnostic guess from binned means: preferred angle, baseline, peak heights."""
    idx = np.minimum((np.mod(theta, 2 * np.pi) / (2 * np.pi) * bins).astype(int), bins - 1)
    means = np.array([y[idx == k].mean() if np.any(idx == k) else np.nan for k in range(bins)])
    means = np.where(np.isfinite(means), means, np.nanmean(means))
    smooth = (np.roll(means, 1) + 2 * means + np.roll(means, -1)) / 4  # circular smoothing
    k = int(np.argmax(smooth))
    b = max(float(smooth.min()), 0.0)
    a1 = max(float(smooth[k]) - b, 1e-3)
    a2 = max(float(smooth[(k + bins // 2) % bins]) - b, 0.05 * a1)
    return Peaks(b=b, a1=a1, a2=a2, m=(k + 0.5) * 2 * np.pi / bins)


def initial_params(eq: str, pk: Peaks) -> np.ndarray:
    """Neutral start per equation: skews/offsets 0, exponents 2, widths ~ 0.5 rad."""
    b, a1, a2, m, s = pk.b, pk.a1, pk.a2, pk.m, pk.s
    v = {
        "b": b, "a1": a1, "a2": a2, "m": m, "s": s, "mu1": m, "mu2": m + np.pi,
        "r1": 0.0, "q1": 0.0, "kp": 1.0, "r2": 0.0, "q2": 0.0, "ko": 1.0, "g1": 0.0, "g2": 0.0,
        "w1l": s, "w1r": s, "w2l": s, "w2r": s, "p1": 2.0, "p2": 2.0, "dlt": 0.0,
        "c": 0.0, "q": 1.0, "h": 0.05 * a1, "phi": m, "k": 1.0 / s**2, "r": 0.0,
        "A1": a1, "A2": a2, "d1": 0.0, "d2": 0.0, "w1": 0.6 * s, "w2": 0.6 * s,
        "s1": s, "s2": s, "ar": 0.1 * a1, "mr": m + np.pi / 2, "dr": 0.0, "sr": s,
        "p": 2.0, "w": 1.0,
    }  # fmt: skip
    if eq == "idea_132":
        v["c"] = 0.05 * a1
        v["h"] = 2 * m
    if eq == "idea_138":
        v["k"] = 0.0
    if eq == "idea_35":
        v["s"] = 1.0
    return np.array([v[n] for n in EQUATIONS[eq][0]], float)


def symbolic(eq: str):
    params, src = EQUATIONS[eq]
    theta = sp.Symbol("theta", real=True)
    ps = [sp.Symbol(n, real=True) for n in params]
    local = {
        "theta": theta,
        "wrap": wrap,
        "step": step,
        "Abs": sp.Abs,
        **dict(zip(params, ps, strict=True)),
    }
    e = parse_expr(src, local_dict=local)
    if unknown := {s.name for s in e.free_symbols} - set(local):
        raise ValueError(f"{eq}: undeclared symbols {sorted(unknown)}")
    return e, ps, [theta]


def build(spec, rng: np.random.Generator):
    from ..models import SymbolicModel
    from ..problem import Problem

    d = np.load(spec.path)
    x_all, y_all, cell_ids = d["x"], d["y"], d["cell_ids"]
    rows = np.arange(len(cell_ids))
    if spec.max_cells is not None and spec.max_cells < rows.size:
        rows = (
            np.sort(rng.choice(rows, spec.max_cells, replace=False))
            if spec.cell_sample == "random"
            else rows[: spec.max_cells]
        )
    eqs = list(EQUATIONS) if spec.equations == "all" else list(spec.equations)
    if bad := set(eqs) - set(EQUATIONS):
        raise ValueError(f"unknown equations {sorted(bad)}; choose from {sorted(EQUATIONS)}")
    compiled = {}
    for eq in eqs:
        e, ps, vs = symbolic(eq)
        compiled[eq] = compile_sympy(e, ps, vs)
    linear: dict[str, list[str]] = {}
    out = []
    for r in rows:
        theta, y = x_all[r], y_all[r]
        pk = estimate_peaks(theta, y)
        for eq in eqs:
            names = EQUATIONS[eq][0]
            th0 = initial_params(eq, pk)
            if eq not in linear:  # detect once per equation, near a real start
                probe = SymbolicModel(
                    eq, None, names, ["theta"], theta, linear=[], compiled=compiled[eq]
                )
                linear[eq] = [names[i] for i in probe.detect_linear(th0)]
            model = SymbolicModel(
                eq, None, names, ["theta"], theta, linear=linear[eq], compiled=compiled[eq]
            )
            starts = {"estimate": th0}
            scale = np.maximum(np.abs(th0), 0.1)
            for k in range(spec.n_perturbed):
                starts[f"perturbed{k}"] = th0 + spec.perturb_scale * scale * rng.standard_normal(
                    th0.size
                )
            out.append(
                Problem(
                    pid=f"{spec.name}/{eq}/cell{int(cell_ids[r])}",
                    dataset=spec.name,
                    model_name=eq,
                    curve=f"cell{int(cell_ids[r])}",
                    model=model,
                    y=y,
                    starts=starts,
                    param_scale=scale,
                    ref_mode="best_found",
                    loss_norm=spec.loss_norm,
                    tags={"cell_row": int(r)},
                )
            )
    return out
