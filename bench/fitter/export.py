"""Export ``Problem``s to the JSON read by ``eqnfitrs-bench``.

Programs are shared between problems with the same symbolic model (e.g. 1,000 curves of
one family) and compiled once. Problems whose model has no symbolic form (finite-difference
Python candidates) are skipped and listed.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import sympy as sp

from ..config import MAX_THETA_P
from ..models import SymbolicModel, VectorResidualModel
from ..models.families import CoupledTanh, ExpMixture, SinMixture
from ..problem import Problem
from .tape import NotExportable, model_programs


def _family_expr(model):
    x = sp.Symbol("x", real=True)
    ps = [sp.Symbol(n, real=True) for n in model.param_names]
    if isinstance(model, CoupledTanh):
        p = model.p
        e = sp.tanh(sum(ps[k] * sp.cos(k * sp.pi * x) for k in range(p)) / sp.sqrt(p))
    else:
        k = model.k
        phi = sp.exp if isinstance(model, ExpMixture) else sp.sin
        sgn = -1 if isinstance(model, ExpMixture) else 1
        e = sum(ps[i] * phi(sgn * ps[k + i] * x) for i in range(k))
        if model.offset:
            e = e + ps[2 * k]
    return [e], ps, [x], [np.ravel(model.x)]


def symbolic_form(model):
    """(rows, psyms, vsyms, data columns, cache key) or raise NotExportable."""
    if isinstance(model, SymbolicModel):
        e = model.compiled.expr
        by_name = {s.name: s for s in e.free_symbols}
        ps = [by_name.get(n, sp.Symbol(n, real=True)) for n in model.param_names]
        vs = [by_name.get(n, sp.Symbol(n, real=True)) for n in model.variables]
        cols = [model.x[:, j] for j in range(model.x.shape[1])]
        key = ("sym", sp.srepr(e), tuple(model.param_names), tuple(model.variables))
        return [e], ps, vs, cols, key
    if isinstance(model, VectorResidualModel):
        key = ("res", model.name, model.p, tuple(sp.srepr(r) for r in model.residuals))
        return list(model.residuals), model.psyms, [], [], key
    if isinstance(model, ExpMixture | SinMixture | CoupledTanh):
        rows, ps, vs, cols = _family_expr(model)
        return rows, ps, vs, cols, ("fam", type(model).__name__, model.p)
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
    programs: list[dict] = []
    cache: dict = {}
    out, skipped = [], {}
    for pb in problems:
        model = pb.model
        try:
            rows, ps, vs, cols, key = symbolic_form(model)
            lin = [int(i) for i in model.linear_idx]
            key = (*key, tuple(lin))
            if key not in cache:
                f, fj, sep = model_programs(rows, ps, vs, lin)
                ids = []
                for prog in (f, fj, sep):
                    if prog is None:
                        ids.append(None)
                    else:
                        programs.append(prog)
                        ids.append(len(programs) - 1)
                cache[key] = ids
            ids = cache[key]
            if ids[2] is None:
                lin = []  # not separable after all: VarPro falls back to LM
        except NotExportable as exc:
            skipped[pb.pid] = str(exc)
            continue
        if not (np.all(np.isfinite(pb.y)) and all(np.all(np.isfinite(c)) for c in cols)):
            skipped[pb.pid] = "non-finite data"
            continue
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
                "p": pb.p,
                "m": pb.m,
                "x": [_list(c) for c in cols],
                "y": _list(pb.y),
                "w": _list(pb.weights),
                "denom": float(pb.denom),
                "linear_idx": lin,
                "prog_f": ids[0],
                "prog_fj": ids[1],
                "prog_sep": ids[2],
                "starts": [[k, _list(v)] for k, v in pb.starts.items() if np.all(np.isfinite(v))],
                "record": record,
            }
        )
    meta = {**meta, "export_s": time.perf_counter() - t0, "skipped": skipped}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"meta": meta, "programs": programs, "problems": out}, fh, allow_nan=False)
    return {
        "n_exported": len(out),
        "n_programs": len(programs),
        "skipped": skipped,
        "export_s": meta["export_s"],
    }
