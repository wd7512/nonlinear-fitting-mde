"""FitBenchmarking problem files (https://github.com/fitbenchmarking/fitbenchmarking, BSD-3)
in the Mantid format, re-implemented without Mantid.

A Mantid ``function`` string is a ``;``-separated sum of components
``name=<Function>,<Param>=<start>,...``. The components below are translated to
SymPy (formulas from the Mantid fit-function documentation); problems using any
other component, ``ties``/``constraints`` or multi-dataset fits are skipped and
listed by ``scan``.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from ..models.symbolic import expwerfc

X = sp.Symbol("x", real=True)


def _p(names, prefix):
    return {n: sp.Symbol(f"{prefix}{n}", real=True) for n in names}


def _btb(q):  # BackToBackExponential
    intensity, A, B, X0, S = q["I"], q["A"], q["B"], q["X0"], q["S"]
    u = X - X0
    return (
        intensity
        * A
        * B
        / (2 * (A + B))
        * (
            expwerfc(A / 2 * (A * S**2 + 2 * u), (A * S**2 + u) / (sp.sqrt(2) * S))
            + expwerfc(B / 2 * (B * S**2 - 2 * u), (B * S**2 - u) / (sp.sqrt(2) * S))
        )
    )


COMPONENTS = {
    "LinearBackground": (["A0", "A1"], lambda q: q["A0"] + q["A1"] * X),
    "FlatBackground": (["A0"], lambda q: q["A0"]),
    "Gaussian": (
        ["Height", "PeakCentre", "Sigma"],
        lambda q: q["Height"] * sp.exp(-((X - q["PeakCentre"]) ** 2) / (2 * q["Sigma"] ** 2)),
    ),
    "GausDecay": (["A", "Sigma"], lambda q: q["A"] * sp.exp(-((q["Sigma"] * X) ** 2))),
    "ExpDecayOsc": (
        ["A", "Lambda", "Frequency", "Phi"],
        lambda q: (
            q["A"] * sp.exp(-q["Lambda"] * X) * sp.cos(2 * sp.pi * q["Frequency"] * X + q["Phi"])
        ),
    ),
    "GausOsc": (
        ["A", "Sigma", "Frequency", "Phi"],
        lambda q: (
            q["A"]
            * sp.exp(-((q["Sigma"] * X) ** 2) / 2)
            * sp.cos(2 * sp.pi * q["Frequency"] * X + q["Phi"])
        ),
    ),
    "StretchExpMuon": (
        ["A", "Lambda", "Beta"],
        lambda q: q["A"] * sp.exp(-((q["Lambda"] * X) ** q["Beta"])),
    ),
    "BackToBackExponential": (["I", "A", "B", "X0", "S"], _btb),
}


class Unsupported(Exception):
    pass


@dataclass
class FbProblem:
    name: str
    set: str
    file: Path
    expr: sp.Expr
    params: list[sp.Symbol]
    start: np.ndarray
    x: np.ndarray
    y: np.ndarray
    e: np.ndarray | None
    components: list[str] = field(default_factory=list)


def _entries(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _unquote(v: str) -> str:
    return ast.literal_eval(v) if v[:1] in "'\"" else v


def _split_args(comp: str) -> list[tuple[str, str]]:
    parts, depth, cur = [], 0, ""
    for ch in comp:
        depth += ch in "([" and 1 or (-1 if ch in ")]" else 0)
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    out = []
    for p in parts:
        if p.strip():
            k, v = p.split("=", 1)
            out.append((k.strip(), v.strip()))
    return out


def parse_function(func: str) -> tuple[sp.Expr, list[sp.Symbol], np.ndarray, list[str]]:
    expr, params, start, names = sp.Integer(0), [], [], []
    for i, comp in enumerate(c for c in func.split(";") if c.strip()):
        args = _split_args(comp)
        name = dict(args).get("name")
        if any(k in ("ties", "constraints") for k, _ in args):
            raise Unsupported("ties/constraints")
        prefix = f"f{i}_"
        if name == "UserFunction":
            formula = dict(args)["Formula"].replace("^", "**")
            vals = [(k, float(v)) for k, v in args if k not in ("name", "Formula")]
            q = _p([k for k, _ in vals], prefix)
            e = parse_expr(formula, local_dict={"x": X, **{k: q[k] for k in q}})
            params += [q[k] for k, _ in vals]
            start += [v for _, v in vals]
        elif name in COMPONENTS:
            pnames, f = COMPONENTS[name]
            given = {k: float(v) for k, v in args if k != "name"}
            if extra := set(given) - set(pnames):
                raise Unsupported(f"{name} attributes {sorted(extra)}")
            q = _p(pnames, prefix)
            e = f(q)
            params += [q[k] for k in pnames]
            start += [given.get(k, 0.0) for k in pnames]
        else:
            raise Unsupported(f"function {name}")
        expr += e
        names.append(name)
    return expr, params, np.asarray(start, float), names


def _read_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rows = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            rows.append([float(v) for v in s.split()])
        except ValueError:
            continue
    a = np.asarray(rows, float)
    return a[:, 0], a[:, 1], (a[:, 2] if a.shape[1] > 2 else None)


def parse_problem(path: Path, set_name: str) -> FbProblem:
    ent = _entries(path.read_text(errors="replace"))
    if "mantid" not in _unquote(ent.get("software", "")).lower():
        raise Unsupported(f"software {ent.get('software')}")
    if "n_fits" in ent or ent.get("input_file", "").startswith("["):
        raise Unsupported("multi-fit")
    expr, params, start, comps = parse_function(_unquote(ent["function"]))
    x, y, e = _read_data(path.parent / "data_files" / _unquote(ent["input_file"]))
    if "fit_ranges" in ent:
        rng = ast.literal_eval(ent["fit_ranges"])
        lo, hi = rng["x"]
        keep = (x >= lo) & (x <= hi)
        x, y, e = x[keep], y[keep], (None if e is None else e[keep])
    order = np.argsort(x, kind="stable")
    x, y, e = x[order], y[order], (None if e is None else e[order])
    name = _unquote(ent.get("name", path.stem))
    return FbProblem(name, set_name, path, expr, params, start, x, y, e, comps)


def scan(root: Path, sets: list[str]) -> tuple[list[FbProblem], dict[str, str]]:
    ok, skipped = [], {}
    for s in sets:
        for f in sorted((root / s).rglob("*.txt")):
            if "data_files" in f.parts:
                continue
            head = f.read_text(errors="replace")[:200]
            if "FitBenchmark Problem" not in head:
                continue
            try:
                ok.append(parse_problem(f, str(f.parent.relative_to(root))))
            except (Unsupported, KeyError, FileNotFoundError) as exc:
                skipped[str(f.relative_to(root))] = f"{type(exc).__name__}: {exc}"
    return ok, skipped


def build(spec, rng=None):
    from ..models import SymbolicModel
    from ..models.symbolic import compile_sympy
    from ..problem import Problem

    fbs, _ = scan(Path(spec.path), spec.sets)
    out, seen = [], set()
    for fb in fbs:
        key = re.sub(r"[^A-Za-z0-9_.-]+", "_", fb.name).strip("_")
        while key in seen:
            key += "_"
        seen.add(key)
        names = [s.name for s in fb.params]
        model = SymbolicModel(
            key,
            None,
            names,
            ["x"],
            fb.x,
            compiled=compile_sympy(fb.expr, fb.params, [X]),
            linear=[],
        )
        model.set_linear(model.detect_linear(center=fb.start))
        weights = None
        if spec.use_errors and fb.e is not None:
            weights = 1.0 / np.maximum(fb.e, 1e-8)  # FitBenchmarking's trim of tiny errors
        out.append(
            Problem(
                pid=f"{spec.name}/{fb.set}/{key}",
                dataset=spec.name,
                model_name=fb.set,
                curve=key,
                model=model,
                y=fb.y,
                starts={"given": fb.start},
                param_scale=np.where(fb.start != 0, np.abs(fb.start), 1.0),
                weights=weights,
                ref_mode="best_found",
                loss_norm=spec.loss_norm,
                tags={"functions": "+".join(sorted(set(fb.components)))},
            )
        )
    return out
