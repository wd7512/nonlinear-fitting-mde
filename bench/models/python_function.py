"""Models given as Python source, e.g. LLM-proposed candidates
``def equation(t, P, params): return params[0] * P * (1 - P / params[1])``.

The source is first traced symbolically: it runs with ``np`` replaced by a shim
that maps NumPy functions to SymPy, giving an exact expression, analytic Jacobian
and VarPro structure. The traced expression is checked against the original code
numerically; if tracing fails (``np.gradient``, ``np.interp``, slicing, ...) or
disagrees, the model falls back to the original callable with a central
finite-difference Jacobian (as SAGE-Fit does).

The source is executed: only use candidate code you trust.
"""

from __future__ import annotations

import re
import textwrap
import types
import warnings

import numpy as np
import sympy as sp

from .base import Model
from .symbolic import SymbolicModel, compile_sympy

_SHIM = {
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "tanh": sp.tanh,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "arctan": sp.atan,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctanh": sp.atanh,
    "abs": sp.Abs,
    "absolute": sp.Abs,
    "power": sp.Pow,
    "log10": lambda v: sp.log(v, 10),
    "log1p": lambda v: sp.log(1 + v),
    "expm1": lambda v: sp.exp(v) - 1,
    "square": lambda v: v**2,
    "pi": sp.pi,
    "e": sp.E,
    "float64": float,
    "ndarray": object,
}


class _NotTraceable(Exception):
    pass


class _SympyNumpy(types.SimpleNamespace):
    def __getattr__(self, name):
        raise _NotTraceable(f"np.{name} has no symbolic equivalent")


def count_params(src: str) -> int:
    idx = {int(m) for m in re.findall(r"params\[(\d+)\]", src)}
    return max(idx) + 1 if idx else 0


def _exec(src: str, ns: dict) -> dict:
    with warnings.catch_warnings():  # docstrings with LaTeX escapes
        warnings.simplefilter("ignore", SyntaxWarning)
        exec(compile(textwrap.dedent(src), "<candidate>", "exec"), ns)  # noqa: S102 - trusted code
    return ns


def load_callable(src: str, func: str = "equation"):
    return _exec(src, {"np": np})[func]


def _trace(src: str, func: str, variables: list[str], n_params: int):
    ns = _exec(src, {"np": _SympyNumpy(**_SHIM)})
    vsyms = [sp.Symbol(v, real=True) for v in variables]
    psyms = [sp.Symbol(f"p{i}", real=True) for i in range(n_params)]
    out = ns[func](*vsyms, psyms)
    expr = sp.sympify(out)
    if not isinstance(expr, sp.Expr) or (expr.free_symbols - set(vsyms) - set(psyms)):
        raise _NotTraceable("result is not a scalar expression of the inputs")
    return expr, psyms, vsyms


class FiniteDifferenceModel(Model):
    """Original callable + central-difference Jacobian (2p extra evaluations)."""

    jacobian_kind = "finite_difference"

    def __init__(self, name: str, fn, n_params: int, x: np.ndarray, seed_random: bool = False):
        super().__init__(name, x.reshape(len(x), -1), [f"p{i}" for i in range(n_params)])
        self.fn = fn
        self._cols = [self.x[:, j] for j in range(self.x.shape[1])]
        self.seed_random = seed_random
        self._h = np.finfo(float).eps ** (1 / 3)

    def predict(self, theta):
        if self.seed_random:  # candidate calls np.random: make it a deterministic function
            np.random.seed(0)
        try:
            out = self.fn(*self._cols, np.asarray(theta, dtype=float))
            return np.broadcast_to(np.asarray(out, dtype=float), (self.m,)).copy()
        except Exception:  # noqa: BLE001 - candidate code failing at theta = a non-finite trial
            return np.full(self.m, np.nan)

    def jacobian(self, theta):
        theta = np.asarray(theta, dtype=float)
        J = np.empty((self.m, self.p))
        for j in range(self.p):
            h = self._h * max(1.0, abs(theta[j]))
            e = np.zeros(self.p)
            e[j] = h
            J[:, j] = (self.predict(theta + e) - self.predict(theta - e)) / (2 * h)
        return J


def python_function_model(
    name: str,
    src: str,
    variables: list[str],
    x: np.ndarray,
    func: str = "equation",
    n_params: int | None = None,
    check_points: int = 3,
    rtol: float = 1e-9,
) -> Model:
    n = count_params(src) if n_params is None else n_params
    if n == 0:
        raise ValueError(f"{name}: no params[...] found in source")
    fn = load_callable(src, func)
    x2 = np.asarray(x, float).reshape(len(x), -1)
    try:
        expr, psyms, vsyms = _trace(src, func, variables, n)
        model = SymbolicModel(
            name,
            None,
            [s.name for s in psyms],
            variables,
            x2,
            compiled=compile_sympy(expr, psyms, vsyms),
        )
        rng = np.random.default_rng(0)
        for k in range(check_points):  # traced expression must reproduce the original code
            th = np.ones(n) if k == 0 else rng.uniform(0.5, 1.5, n)
            with np.errstate(all="ignore"):
                a = model.predict(th)
                b = np.broadcast_to(
                    np.asarray(fn(*[x2[:, j] for j in range(x2.shape[1])], th), float), a.shape
                )
            ok = np.isfinite(a) == np.isfinite(b)
            fin = np.isfinite(a) & np.isfinite(b)
            if not (ok.all() and np.allclose(a[fin], b[fin], rtol=rtol, atol=1e-300)):
                raise _NotTraceable("traced expression disagrees with the original code")
        return model
    except Exception:  # noqa: BLE001 - any tracing problem means: use the code as is
        model = FiniteDifferenceModel(name, fn, n, x2, seed_random="np.random" in src)
        model.set_linear(
            model.detect_linear(rtol=1e-6)
        )  # FD columns of affine params are exact to ~1e-10
        return model
