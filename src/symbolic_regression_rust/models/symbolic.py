"""Models built with SymPy: analytic Jacobians generated once, evaluated with NumPy.

* ``SymbolicModel``        one expression over data variables, e.g. ``"b1*(1-exp(-b2*x))"``
* ``VectorResidualModel``  a fixed vector of residual expressions in the parameters only
                           (MGH-style problems with no data columns)

Kernels are lambdified with common-subexpression elimination and cached, so many
curves sharing one equation compile it once. Conditionally linear parameters are
detected numerically (``Model.detect_linear``) for VarPro.

Only parse expressions you trust: SymPy's parser evaluates Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from .base import Model


class expwerfc(sp.Function):  # noqa: N801 - SymPy function name
    """exp(a) * erfc(b), evaluated without overflow (used by Mantid's BackToBackExponential)."""

    nargs = 2

    def fdiff(self, argindex=1):
        a, b = self.args
        if argindex == 1:
            return self
        return -2 / sp.sqrt(sp.pi) * sp.exp(a - b**2)


def _np_expwerfc(a, b):
    from scipy.special import erfc, erfcx

    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    with np.errstate(all="ignore"):
        pos = np.exp(a - b * b) * erfcx(np.maximum(b, 0.0))  # stable for b >= 0
        neg = np.exp(a) * erfc(np.minimum(b, 0.0))  # erfc(b) in (1, 2] for b < 0
    return np.where(b >= 0, pos, neg)


class wrap(sp.Function):  # noqa: N801 - SymPy function name
    is_extended_real = True
    is_real = True
    """Angle wrapped to (-pi, pi]: atan2(sin d, cos d). Derivative 1 (away from the cut)."""

    nargs = 1

    def fdiff(self, argindex=1):
        return sp.Integer(1)


class step(sp.Function):  # noqa: N801 - SymPy function name
    is_extended_real = True
    is_real = True
    """1 if d >= 0 else 0 (selects left/right widths); derivative 0 (away from 0)."""

    nargs = 1

    def fdiff(self, argindex=1):
        return sp.Integer(0)


def _np_wrap(d):
    return np.arctan2(np.sin(d), np.cos(d))


def _np_step(d):
    return (np.asarray(d) >= 0).astype(float)


MODULES = [
    {"expwerfc": _np_expwerfc, "wrap": _np_wrap, "step": _np_step},
    "numpy",
    "scipy",
]


@dataclass(frozen=True)
class Compiled:
    expr: object  # sp.Expr or list of sp.Expr
    f: object
    jac: object


def compile_sympy(expr: sp.Expr, psyms: list[sp.Symbol], vsyms: list[sp.Symbol]) -> Compiled:
    args = [*vsyms, *psyms]
    f = sp.lambdify(args, expr, modules=MODULES, cse=True)
    jac = sp.lambdify(args, [sp.diff(expr, s) for s in psyms], modules=MODULES, cse=True)
    return Compiled(expr, f, jac)


@lru_cache(maxsize=512)
def compile_expression(expr: str, params: tuple[str, ...], variables: tuple[str, ...]) -> Compiled:
    psyms = [sp.Symbol(n, real=True) for n in params]
    vsyms = [sp.Symbol(n, real=True) for n in variables]
    local = {s.name: s for s in [*psyms, *vsyms]}
    e = parse_expr(expr, local_dict=local)
    unknown = {s.name for s in e.free_symbols} - set(local)
    if unknown:
        raise ValueError(f"expression {expr!r} uses undeclared symbols {sorted(unknown)}")
    return compile_sympy(e, psyms, vsyms)


class SymbolicModel(Model):
    jacobian_kind = "analytic"

    def __init__(
        self,
        name: str,
        expr,
        params: list[str],
        variables: list[str],
        x: np.ndarray,
        linear: list[str] | None = None,
        compiled: Compiled | None = None,
    ):
        x = np.asarray(x, dtype=float)
        x2 = x.reshape(len(x), -1)
        if x2.shape[1] != len(variables):
            raise ValueError(
                f"{name}: {len(variables)} variables {variables} but x has {x2.shape[1]} columns"
            )
        self.compiled = compiled or compile_expression(expr, tuple(params), tuple(variables))
        super().__init__(name, x2, params)
        self.expr = expr
        self.variables = list(variables)
        self._cols = [x2[:, j] for j in range(x2.shape[1])]
        if linear is None:
            self.set_linear(self.detect_linear())
        else:
            if bad := set(linear) - set(params):
                raise ValueError(f"{name}: linear params {sorted(bad)} not in params")
            self.set_linear([params.index(n) for n in linear])

    def predict(self, theta):
        return np.broadcast_to(self.compiled.f(*self._cols, *theta), (self.m,)).astype(
            float, copy=False
        )

    def jacobian(self, theta):
        cols = self.compiled.jac(*self._cols, *theta)
        return np.column_stack([np.broadcast_to(c, (self.m,)) for c in cols]).astype(
            float, copy=False
        )

    def describe(self) -> str:
        return f"{super().describe()}  f = {self.compiled.expr}"


class VectorResidualModel(Model):
    """predict(theta) = [r_1(theta), ..., r_m(theta)]; pair with y = 0."""

    jacobian_kind = "analytic"

    def __init__(
        self,
        name: str,
        residuals: list[sp.Expr],
        psyms: list[sp.Symbol],
        center: np.ndarray | None = None,
    ):
        self.residuals = residuals
        self.psyms = list(psyms)
        self._f = sp.lambdify(psyms, residuals, modules=MODULES, cse=True)
        J = sp.Matrix(residuals).jacobian(psyms)
        self._J = sp.lambdify(psyms, J, modules=MODULES, cse=True)
        super().__init__(name, np.zeros((len(residuals), 0)), [s.name for s in psyms])
        self.set_linear(self.detect_linear(center))

    @property
    def m(self) -> int:
        return len(self.residuals)

    def predict(self, theta):
        return np.array([np.asarray(v, dtype=float) for v in self._f(*theta)], dtype=float).reshape(
            -1
        )

    def jacobian(self, theta):
        return np.asarray(self._J(*theta), dtype=float).reshape(self.m, self.p)
