"""Fast nonlinear least-squares fitting of symbolic expressions on millisecond CPU budgets.

model = symbfitrs.compile("a*exp(-b*x) + c")
result = model.fit(x, y, budget_ms=5)

results = symbfitrs.Pool(threads=8).fit_many([(expr_or_model, x, y), ...], budget_ms=5)
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import version

import numpy as np

from . import _core

__all__ = ["FitResult", "Model", "Pool", "compile"]
__version__ = version("symbfitrs")


@dataclass(frozen=True)
class FitResult:
    """Best parameters found and how the fit ended.

    `status` is "converged", "stalled" (no further progress), "max_steps" (`max_steps` accepted
    updates taken), "deadline" (budget spent; `params`
    is the best so far), "failure" or "error" (the job could not run; see `message`).
    """

    names: tuple[str, ...]
    params: np.ndarray
    loss: float  # mean squared (weighted) residual at params
    status: str
    message: str = ""
    nfev: int = 0
    time_ms: float = 0.0
    n_nonfinite: int = 0  # evaluations with a non-finite loss (skipped)
    # each improvement of the best loss: when (ms since the call) and the new best
    history_ms: np.ndarray = field(default_factory=lambda: np.array([]))
    history_loss: np.ndarray = field(default_factory=lambda: np.array([]))
    # accepted optimizer updates completed; with `trace=True`, the full parameter vector (in
    # `names` order) at step 0 and after each accepted update: (steps + 1, n_params). Step 0 is
    # the start; linear parameters that the solver eliminates are already solved exactly at it.
    steps: int = 0
    trace: np.ndarray | None = None

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


Bounds = Mapping[str, tuple[float | None, float | None]]


def _bounds(params: Sequence[str], bounds: Bounds | None):
    """{name: (lo, hi)} as (lower, upper) lists in `params` order; None = no bounds."""
    if not bounds:
        return None, None
    unknown = [n for n in bounds if n not in params]
    if unknown:
        raise ValueError(f"bounds for unknown parameters {unknown}")
    lo, hi = [], []
    for n in params:
        a, b = bounds.get(n, (None, None))
        lo.append(-np.inf if a is None else float(a))
        hi.append(np.inf if b is None else float(b))
        if np.isnan(lo[-1]) or np.isnan(hi[-1]) or lo[-1] > hi[-1]:
            raise ValueError(f"bounds for {n!r} must have lower <= upper")
    return lo, hi


def _result(raw) -> FitResult:
    (
        names,
        params,
        loss,
        status,
        message,
        nfev,
        n_nonfinite,
        time_ms,
        hist_ms,
        hist_loss,
        steps,
        trace,
    ) = raw
    return FitResult(
        tuple(names),
        np.array(params),
        loss,
        status,
        message,
        nfev,
        time_ms,
        n_nonfinite,
        np.array(hist_ms),
        np.array(hist_loss),
        steps,
        None if trace is None else np.array(trace, dtype=float).reshape(-1, len(names)),
    )


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

    def predict(self, x, params) -> np.ndarray:
        """f(x; params) at every data point."""
        cols = _columns(x, self.variables)
        m = np.asarray(cols[0]).size if cols else 1
        xs = [_bytes(c, m, f"x[{v!r}]") for v, c in zip(self.variables, cols, strict=True)]
        return np.array(self._core.predict(xs, [float(v) for v in params]))

    def fit(
        self,
        x,
        y,
        *,
        p0=None,
        weights=None,
        budget_ms: float = 5.0,
        trace: bool = False,
        max_steps: int | None = None,
        bounds: Bounds | None = None,
    ) -> FitResult:
        """Minimise Σ (w (f(x; θ) − y))² from `p0` (default: all ones) within `budget_ms`.

        `bounds={name: (lo, hi)}` (None = unbounded on that side) keeps every iterate inside the
        box: a parameter on a bound is held there while its gradient points out of the box, and
        the start is first moved into the box. Use it instead of clipping inside the expression,
        where the clipped region has zero gradient and a parameter that lands there cannot come
        back. `lo == hi` fixes a parameter. Bounding a linear parameter turns variable projection
        off.

        `trace=True` also returns θ at step 0 (the start) and after every accepted update;
        `max_steps` stops after that many accepted updates (status "max_steps")."""
        lower, upper = _bounds(self.params, bounds)
        raw = self._core.fit(
            *_args(self.variables, x, y, weights, p0),
            float(budget_ms),
            trace,
            max_steps,
            lower,
            upper,
        )
        return _result(raw)

    def __repr__(self) -> str:
        return f"Model(params={self.params}, variables={self.variables}, linear={self.linear})"


def compile(  # noqa: A001 (the natural name for the API)
    expr: str, variables: Sequence[str] = ("x",), params: Sequence[str] | None = None
) -> Model:
    """Parse `expr`. Names in `variables` are data; every other name is a parameter, in order
    of first appearance unless `params` gives the order. Supports + - * / ^ (or **), exp, log,
    sqrt, sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, atan2, abs, sign, erf, erfc, max,
    min, pow, wrap (angle to (−π, π]), step (1 for x ≥ 0), expwerfc(a, b) = exp(a)·erfc(b) and
    pi. Raises ValueError on a malformed expression."""
    return Model(_core.compile(expr, list(variables), None if params is None else list(params)))


class Pool:
    """Fits many (model, data) jobs on `threads` threads (default: all cores)."""

    def __init__(self, threads: int | None = None) -> None:
        self.threads = threads or os.cpu_count() or 1

    def fit_many(
        self,
        jobs: Iterable[tuple],
        *,
        variables: Sequence[str] = ("x",),
        budget_ms: float = 5.0,
        progress: bool = True,
        trace: bool = False,
        max_steps: int | None = None,
        bounds: Bounds | None = None,
    ) -> list[FitResult]:
        """Each job is (model, x, y), (model, x, y, p0) or (model, x, y, p0, weights), where
        model is a compiled Model or an expression over `variables`. Results come back in job
        order; a job that cannot run gets status "error" instead of stopping the batch. Each
        job's budget starts when a thread picks it up. Repeated expressions are compiled once.
        `trace`, `max_steps` and `bounds` apply to every job, as in `Model.fit`; a job whose model
        has no parameter named in `bounds` is an error."""
        variables = list(variables)
        raw, errors = [], {}
        params_of: dict[tuple, list[str]] = {}
        for i, (model, x, y, *rest) in enumerate(jobs):
            p0, weights = (*rest, None, None)[:2]
            try:
                if isinstance(model, Model):
                    src, names, params = model._core, model.variables, model.params
                else:
                    src, names, params = model, variables, None
                    if bounds:  # bounds are by name, so the expression's parameters are needed here
                        key = (model, tuple(variables))
                        if key not in params_of:
                            params_of[key] = compile(model, variables).params
                        params = params_of[key]
                lower, upper = _bounds(params, bounds) if bounds else (None, None)
                raw.append((src, names, *_args(names, x, y, weights, p0), (lower, upper)))
            except ValueError as e:
                errors[i] = str(e)
                raw.append(("", variables, [], b"", None, None, (None, None)))  # placeholder
        bar = None
        if progress:
            from tqdm.auto import tqdm

            bar = tqdm(total=len(raw), unit="fit")
        try:
            out = _core.fit_many(
                raw, float(budget_ms), self.threads, bar.update if bar else None, trace, max_steps
            )
        finally:
            if bar is not None:
                bar.close()
        return [
            _result(ok)
            if ok is not None and i not in errors
            else FitResult((), np.array([]), float("nan"), "error", errors.get(i, err))
            for i, (ok, err) in enumerate(out)
        ]
