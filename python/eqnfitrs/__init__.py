"""Fast nonlinear least-squares fitting of symbolic expressions on millisecond CPU budgets.

model = eqnfitrs.compile("a*exp(-b*x) + c")
result = model.fit(x, y, budget_ms=5)

results = eqnfitrs.Pool(threads=8).fit_many([(expr, x, y), ...], budget_ms=5)
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np

from . import _core

__all__ = ["FitResult", "Model", "Pool", "compile"]
__version__ = version("eqnfitrs")


@dataclass(frozen=True)
class FitResult:
    """Best parameters found and how the fit ended.

    `status` is "converged", "stalled" (no further progress), "deadline" (budget spent; `params`
    is the best so far), "failure" or "error" (the job could not run; see `message`).
    """

    names: tuple[str, ...]
    params: np.ndarray
    loss: float  # mean squared (weighted) residual at params
    status: str
    message: str = ""
    nfev: int = 0
    time_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.status in ("converged", "stalled") and bool(np.isfinite(self.loss))

    def __getitem__(self, name: str) -> float:
        return float(self.params[self.names.index(name)])

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.params.tolist(), strict=True))


def _bytes(a, n: int | None, what: str) -> bytes:
    a = np.ascontiguousarray(a, dtype=np.float64)
    if a.ndim != 1 or (n is not None and a.size != n):
        raise ValueError(f"{what} must be 1-D" + (f" with {n} values" if n is not None else ""))
    return a.tobytes()


def _columns(x, variables: Sequence[str]) -> list:
    """x as one array per variable: an array (one variable), a (k, m) array, a sequence of k
    arrays, or a mapping from variable name to array."""
    if isinstance(x, dict):
        missing = [v for v in variables if v not in x]
        if missing:
            raise ValueError(f"x has no column for {missing}")
        return [x[v] for v in variables]
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1 and len(variables) == 1:
        return [a]
    if a.ndim == 2 and a.shape[0] == len(variables):
        return list(a)
    raise ValueError(f"x must give {len(variables)} column(s) for variables {list(variables)}")


def _args(variables, x, y, weights, p0):
    yb = _bytes(y, None, "y")
    m = len(yb) // 8
    xs = [_bytes(c, m, f"x[{v!r}]") for v, c in zip(variables, _columns(x, variables), strict=True)]
    w = None if weights is None else _bytes(weights, m, "weights")
    return xs, yb, w, None if p0 is None else [float(v) for v in p0]


def _result(raw) -> FitResult:
    names, params, loss, status, message, nfev, time_ms = raw
    return FitResult(tuple(names), np.array(params), loss, status, message, nfev, time_ms)


class Model:
    """A compiled expression: parsed, differentiated and split into linear and nonlinear
    parameters once, then fitted any number of times."""

    def __init__(self, core: _core.Model) -> None:
        self._core = core

    @property
    def params(self) -> list[str]:
        return self._core.params

    @property
    def variables(self) -> list[str]:
        return self._core.variables

    @property
    def linear(self) -> list[str]:
        """Parameters solved exactly at every step (the model is jointly linear in them)."""
        return [self.params[i] for i in self._core.linear]

    def fit(self, x, y, *, p0=None, weights=None, budget_ms: float = 5.0) -> FitResult:
        """Minimise Σ (w (f(x; θ) − y))² from `p0` (default: all ones) within `budget_ms`."""
        raw = self._core.fit(*_args(self.variables, x, y, weights, p0), float(budget_ms))
        return _result(raw)

    def __repr__(self) -> str:
        return f"Model(params={self.params}, variables={self.variables}, linear={self.linear})"


def compile(  # noqa: A001 (the natural name for the API)
    expr: str, variables: Sequence[str] = ("x",), params: Sequence[str] | None = None
) -> Model:
    """Parse `expr`. Names in `variables` are data; every other name is a parameter, in order
    of first appearance unless `params` gives the order. Supports + - * / ^ (or **), exp, log,
    sqrt, sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, abs, sign, erf, erfc, max, min, pow
    and pi. Raises ValueError on a malformed expression."""
    return Model(_core.compile(expr, list(variables), None if params is None else list(params)))


class Pool:
    """Fits many (expression, data) jobs on `threads` threads (default: all cores)."""

    def __init__(self, threads: int | None = None) -> None:
        self.threads = threads or os.cpu_count() or 1

    def fit_many(
        self,
        jobs: Iterable[tuple],
        *,
        variables: Sequence[str] = ("x",),
        budget_ms: float = 5.0,
        progress: bool = True,
    ) -> list[FitResult]:
        """Each job is (expr, x, y) or (expr, x, y, p0). Results come back in job order; a job
        that cannot run gets status "error" instead of stopping the batch. Each job's budget
        starts when a thread picks it up. Repeated expressions are compiled once."""
        variables = list(variables)
        raw, errors = [], {}
        for i, (expr, x, y, *rest) in enumerate(jobs):
            try:
                raw.append(
                    (expr, variables, *_args(variables, x, y, None, rest[0] if rest else None))
                )
            except ValueError as e:
                errors[i] = str(e)
                raw.append((expr, variables, [], b"", None, None))  # placeholder, replaced below
        bar = None
        if progress:
            from tqdm.auto import tqdm

            bar = tqdm(total=len(raw), unit="fit")
        try:
            out = _core.fit_many(raw, float(budget_ms), self.threads, bar.update if bar else None)
        finally:
            if bar is not None:
                bar.close()
        return [
            _result(ok)
            if ok is not None and i not in errors
            else FitResult((), np.array([]), float("nan"), "error", errors.get(i, err))
            for i, (ok, err) in enumerate(out)
        ]
