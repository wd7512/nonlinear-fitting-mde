"""Registered baselines. Add one by writing a function of a ``Job`` that returns parameters:

    @baseline
    def my_method(job):
        return scipy.optimize.something(job.residual, job.p0).x

The function name is the label in reports. The harness times the call, records the loss of
every ``job.residual`` evaluation, stops the solver at the deadline and scores the result.

Every baseline runs at its library's defaults and gets only the timed residual, with a
forward-difference Jacobian where the library needs one supplied (SciPy's own 2-point step).
The exception is ``varpropy`` (like the Rust ``varpro`` crate, run from rust/): variable
projection needs the model's separable form, so it also uses ``job.separable``. It fails on
problems without both linear and nonlinear parameters or whose offset depends on the nonlinear
ones, and is credited only with the parameters it returns.

A baseline may carry ``fn.prepare(m, p)``: untimed setup per problem shape, called before the
clock starts (JAX's compile, like the Rust fitter's build, is not charged to the fit).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SAGEFIT = Path(__file__).resolve().parents[1] / "data/external/sagefit"  # `bench.fetch sagefit`


@dataclass(frozen=True)
class Job:
    residual: Callable[[Any], Any]  # p -> weighted residual vector
    p0: Any  # starting parameters
    budget_ms: float
    linear_idx: tuple[int, ...] = ()  # parameters the model is linear in
    # theta -> (r0, Phi) with residual = r0(beta) + Phi(beta) @ theta[linear_idx]; timed, no loss
    separable: Callable[[Any], Any] | None = None


BASELINES: dict[str, Callable[[Job], Any]] = {}


def baseline(fn):
    BASELINES[fn.__name__] = fn
    return fn


def fd_jacobian(f, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(f(x), forward-difference Jacobian) with SciPy's '2-point' step."""
    r0 = f(x)
    h = np.sqrt(np.finfo(float).eps) * np.where(x >= 0, 1.0, -1.0) * np.maximum(1.0, np.abs(x))
    J = np.empty((r0.size, x.size))
    for j in range(x.size):
        xj = x.copy()
        xj[j] += h[j]
        J[:, j] = (f(xj) - r0) / (xj[j] - x[j])
    return r0, J


# ---- SciPy ------------------------------------------------------------------------------


def _scipy(method: str):
    def fit(job):
        from scipy.optimize import least_squares  # here, so the CLI does not pay for the import

        return least_squares(job.residual, job.p0, method=method).x

    fit.__name__ = f"scipy_{method}"
    return baseline(fit)


scipy_trf, scipy_lm, scipy_dogbox = (_scipy(m) for m in ("trf", "lm", "dogbox"))


# ---- Optimistix (JAX) -------------------------------------------------------------------

_OPTX: dict[tuple[str, int, int], Any] = {}  # compiled solve per (solver, m, p)
_ACTIVE: dict[str, Any] = {}  # the running job's residual, and a deadline its callback hit


def _call(f, theta):
    try:
        return f(_ACTIVE["residual"], np.asarray(theta))
    except Exception:
        raise
    except BaseException as exc:  # the deadline: JAX would wrap it in an ordinary error
        _ACTIVE["stop"] = exc
        raise


def _optx(name: str):
    def prepare(m: int, p: int):
        if (name, m, p) in _OPTX:
            return
        import jax
        import jax.numpy as jnp
        import optimistix as optx

        jax.config.update("jax_enable_x64", True)
        logging.getLogger("jax._src.callback").setLevel(logging.CRITICAL)  # deadline tracebacks
        r_shape, j_shape = jax.ShapeDtypeStruct((m,), float), jax.ShapeDtypeStruct((m, p), float)

        @jax.custom_jvp
        def residual(theta):
            return jax.pure_callback(lambda t: _call(lambda f, x: f(x), t), r_shape, theta)

        @residual.defjvp
        def residual_jvp(primals, tangents):
            r0, J = jax.pure_callback(lambda t: _call(fd_jacobian, t), (r_shape, j_shape), *primals)
            return r0, J @ tangents[0]

        solver = {
            "gauss_newton": optx.GaussNewton,
            "lm": optx.LevenbergMarquardt,
            "indirect_lm": optx.IndirectLevenbergMarquardt,
            "dogleg": optx.Dogleg,
        }[name](rtol=1e-8, atol=1e-8)

        def solve(y0):
            fn = lambda y, _: residual(y)  # noqa: E731
            return optx.least_squares(fn, solver, y0, max_steps=None, throw=False).value

        _OPTX[(name, m, p)] = jax.jit(solve).lower(jnp.zeros(p)).compile()

    def fit(job):
        p0 = np.asarray(job.p0, float)
        m = job.residual(p0).size
        prepare(m, p0.size)  # already done by the harness, unless called directly
        _ACTIVE.update(residual=job.residual, stop=None)
        try:
            return np.asarray(_OPTX[(name, m, p0.size)](p0))
        except Exception:
            if _ACTIVE["stop"] is not None:
                raise _ACTIVE["stop"] from None
            raise
        finally:
            _ACTIVE.clear()

    fit.__name__ = f"optx_{name}"
    fit.prepare = prepare
    return baseline(fit)


optx_gauss_newton, optx_lm, optx_indirect_lm, optx_dogleg = (
    _optx(n) for n in ("gauss_newton", "lm", "indirect_lm", "dogleg")
)


# ---- Ceres (pyceres) --------------------------------------------------------------------


@baseline
def ceres_lm(job):
    import pyceres

    theta = np.array(job.p0, dtype=float)
    m = job.residual(theta).size

    class Cost(pyceres.CostFunction):
        def __init__(self):
            super().__init__()
            self.set_num_residuals(m)
            self.set_parameter_block_sizes([theta.size])

        def Evaluate(self, parameters, residuals, jacobians):  # noqa: N802 - Ceres API
            x = np.array(parameters[0])
            if jacobians is not None and jacobians[0] is not None:
                r, J = fd_jacobian(job.residual, x)
                jacobians[0][:] = J.ravel()
            else:
                r = job.residual(x)
            if not np.all(np.isfinite(r)):
                return False
            residuals[:] = r
            return True

    problem = pyceres.Problem()
    problem.add_residual_block(Cost(), None, [theta])
    pyceres.solve(pyceres.SolverOptions(), problem, pyceres.SolverSummary())
    return theta


# ---- SAGE-Fit (research code, fetched with `bench.fetch sagefit`) -----------------------


@baseline
def sagefit(job):
    """Fits the residual to zero, so weights and data need no special handling."""
    evaluation = str(SAGEFIT / "Structure_Progress_Bank/evaluation")
    if not Path(evaluation).is_dir():
        raise FileNotFoundError("SAGE-Fit code missing: uv run python -m bench.fetch sagefit")
    if evaluation not in sys.path:
        sys.path.insert(0, evaluation)
    from optimizers.opt_fsfps import fit_SAGE_Fit

    p0 = np.asarray(job.p0, float)
    m = job.residual(p0).size
    X = np.zeros((m, 1))
    out = fit_SAGE_Fit(
        fn=lambda _x, theta: job.residual(np.asarray(theta, float)),
        n_params=p0.size,
        c_init=p0.copy(),
        X_train=X,
        y_train=np.zeros(m),
        X_val=X,
        y_val=np.zeros(m),
        timeout_sec=job.budget_ms / 1e3,
    )
    return out["c_hat"]


# ---- VarProPy ---------------------------------------------------------------------------


@baseline
def varpropy(job):
    from varpropy import Parameters, VarProSolver

    p0 = np.asarray(job.p0, float)
    lin = list(job.linear_idx)
    nl = [i for i in range(p0.size) if i not in set(lin)]
    if not lin or not nl:
        raise ValueError("needs both linear and nonlinear parameters")
    target = -job.separable(p0)[0]  # residual = Phi(beta) @ alpha - target
    cache: dict[tuple, np.ndarray] = {}  # VarProPy asks for one basis column at a time

    def phi(beta) -> np.ndarray:
        key = tuple(beta)
        if key not in cache:
            th = p0.copy()
            th[nl] = beta
            r0, cache[key] = job.separable(th)
            if not np.array_equal(r0, -target):
                raise ValueError("the offset depends on the nonlinear parameters")
            if len(cache) > 2 * len(nl) + 2:  # room for one finite-difference stencil
                del cache[next(iter(cache))]
        return cache[key]

    names = [f"b{k}" for k in range(len(nl))]
    basis = [
        lambda _t, j=j, **kw: phi(np.array([kw[n] for n in names]))[:, j] for j in range(len(lin))
    ]
    params = Parameters()
    for n, v in zip(names, p0[nl], strict=True):
        params.add(n, value=float(v))
    lin_names = [f"a{k}" for k in range(len(lin))]
    for n in lin_names:
        params.add(n, is_linear=True)
    solver = VarProSolver(
        basis, linear_param_names=lin_names, t=np.arange(target.size), data=target
    )
    beta, alpha, _, _ = solver.fit(params)
    theta = p0.copy()
    theta[nl], theta[lin] = beta, alpha
    return theta
