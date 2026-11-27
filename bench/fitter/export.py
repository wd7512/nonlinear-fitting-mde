"""Export ``Problem``s as expression strings + data: the input of the package runner and of
``symbfitrs-bench`` (the ``varpro`` crate baseline).

Each model is printed from its SymPy form into the package's expression grammar. Problems whose
model has no symbolic form (finite-difference Python candidates) are skipped and listed;
expressions the parser rejects are skipped when they are compiled.

Residual-vector problems (MGH: one expression per residual, no data) become a single
expression through indicator columns: variable ``_r{i}`` is 1 on row i and 0 elsewhere, and
f = Σ_i _r{i}·(row i), fitted to y = 0. Each evaluation then computes every residual on every
row, so these problems are slightly pessimistic for symbfitrs.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import sympy as sp
from sympy.printing.str import StrPrinter

from ..config import MAX_THETA_P
from ..models import SymbolicModel, VectorResidualModel
from ..models.families import CoupledTanh, ExpMixture, SinMixture
from ..problem import Problem


class NotExportable(Exception):
    pass


class _Printer(StrPrinter):
    """SymPy → the grammar of ``symbfitrs.compile`` (``rust/src/expr.rs``)."""

    def _print_Float(self, e):
        return repr(float(e))

    def _print_Rational(self, e):
        return f"({e.p}/{e.q})"

    def _print_Exp1(self, e):
        return "exp(1)"

    def _print_Abs(self, e):
        return f"abs({self._print(e.args[0])})"

    def _minmax(self, name, args):
        s = self._print(args[0])
        for a in args[1:]:
            s = f"{name}({s}, {self._print(a)})"
        return s

    def _print_Max(self, e):
        return self._minmax("max", e.args)

    def _print_Min(self, e):
        return self._minmax("min", e.args)

    def _print_Infinity(self, e):
        raise NotExportable("non-finite constant")

    _print_NegativeInfinity = _print_ComplexInfinity = _print_NaN = _print_Infinity

    def _print_ImaginaryUnit(self, e):
        raise NotExportable("complex constant")


def to_expr(e: sp.Expr) -> str:
    return _Printer().doprint(e)


def _family_expr(model) -> sp.Expr:
    x = sp.Symbol("x", real=True)
    ps = [sp.Symbol(n, real=True) for n in model.param_names]
    if isinstance(model, CoupledTanh):
        p = model.p
        return sp.tanh(sum(ps[k] * sp.cos(k * sp.pi * x) for k in range(p)) / sp.sqrt(p))
    k = model.k
    phi = sp.exp if isinstance(model, ExpMixture) else sp.sin
    sgn = -1 if isinstance(model, ExpMixture) else 1
    e = sum(ps[i] * phi(sgn * ps[k + i] * x) for i in range(k))
    return e + ps[2 * k] if model.offset else e


def symbolic_form(model) -> tuple[str, list[str], list[str], list[np.ndarray]]:
    """(expr, variables, params, data columns) or raise NotExportable."""
    if isinstance(model, SymbolicModel):
        cols = [model.x[:, j] for j in range(model.x.shape[1])]
        expr = to_expr(model.compiled.expr)
        return expr, list(model.variables), list(model.param_names), cols
    if isinstance(model, VectorResidualModel):
        m = len(model.residuals)
        expr = " + ".join(f"_r{i}*({to_expr(r)})" for i, r in enumerate(model.residuals))
        return expr, [f"_r{i}" for i in range(m)], [s.name for s in model.psyms], list(np.eye(m))
    if isinstance(model, ExpMixture | SinMixture | CoupledTanh):
        expr = to_expr(_family_expr(model))
        return expr, ["x"], list(model.param_names), [np.ravel(model.x)]
    raise NotExportable(f"{type(model).__name__} has no symbolic form")


def _finite_or_none(v):
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _list(a):
    return None if a is None else [float(v) for v in np.asarray(a, float).ravel()]


def export_problems(problems: list[Problem], meta: dict, path: Path) -> dict:
    t0 = time.perf_counter()
    out, skipped, models = [], {}, set()
    for pb in problems:
        try:
            expr, variables, names, cols = symbolic_form(pb.model)
        except NotExportable as exc:
            skipped[pb.pid] = str(exc)
            continue
        if not (np.all(np.isfinite(pb.y)) and all(np.all(np.isfinite(c)) for c in cols)):
            skipped[pb.pid] = "non-finite data"
            continue
        models.add((expr, tuple(variables), tuple(names)))
        small = pb.p <= MAX_THETA_P
        record = {
            "dataset": pb.dataset,
            "model": pb.model_name,
            "curve": pb.curve,
            "loss_norm": pb.loss_norm,
            "ref_loss": _finite_or_none(pb.ref_loss),
            "ref_mode": pb.ref_mode,
            "ref_params": _list(pb.ref_params) if small else None,
            **{f"tag_{k}": v for k, v in pb.tags.items()},
        }
        out.append(
            {
                "pid": pb.pid,
                "expr": expr,
                "variables": variables,
                "params": names,
                "x": [_list(c) for c in cols],
                "y": _list(pb.y),
                "w": _list(pb.weights),
                "denom": float(pb.denom),
                "starts": [[k, _list(v)] for k, v in pb.starts.items() if np.all(np.isfinite(v))],
                "record": record,
            }
        )
    meta = {**meta, "export_s": time.perf_counter() - t0, "skipped": skipped}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"meta": meta, "problems": out}, fh, allow_nan=False)
    return {
        "n_exported": len(out),
        "n_models": len(models),
        "skipped": skipped,
        "export_s": meta["export_s"],
    }
