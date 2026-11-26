"""Exported tapes must reproduce the Python models (values and Jacobians)."""

import numpy as np
import pytest
from scipy.special import erf, erfc

from bench.fitter.export import symbolic_form
from bench.fitter.tape import model_programs
from bench.models import SymbolicModel
from bench.models.families import make_family
from bench.models.symbolic import _np_expwerfc, compile_sympy
from bench.suites import stringer2019

UN = {
    "neg": np.negative, "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "sin": np.sin,
    "cos": np.cos, "tan": np.tan, "tanh": np.tanh, "sinh": np.sinh, "cosh": np.cosh,
    "atan": np.arctan, "asin": np.arcsin, "acos": np.arccos, "abs": np.abs, "sign": np.sign,
    "erf": erf, "erfc": erfc, "wrap": lambda d: np.arctan2(np.sin(d), np.cos(d)),
    "step": lambda d: (d >= 0).astype(float),
}  # fmt: skip
BIN = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide, "powf": np.power,
    "max": np.maximum, "min": np.minimum, "expwerfc": _np_expwerfc,
}  # fmt: skip


def run_tape(prog, cols, theta, m):
    """Reference interpreter for rust/src/tape.rs."""
    reg = []
    for op in prog["ops"]:
        name, *a = op
        if name == "const":
            v = np.float64(a[0])
        elif name == "var":
            v = np.asarray(cols[a[0]], float)
        elif name == "param":
            v = np.float64(theta[a[0]])
        elif name == "powi":
            v = np.power(np.asarray(reg[a[0]], float), float(a[1]))
        elif name in BIN:
            v = BIN[name](reg[a[0]], reg[a[1]])
        else:
            v = UN[name](reg[a[0]])
        reg.append(v)
    out = []
    for o in prog["outputs"]:
        out.append(
            np.broadcast_to(reg[o[0]], (m,)) if len(o) == 1 else np.array([reg[r] for r in o])
        )
    return out


def check(model, theta):
    rows, ps, vs, cols, _ = symbolic_form(model)
    f, fj, sep = model_programs(rows, ps, vs, list(model.linear_idx))
    with np.errstate(all="ignore"):
        (fv,) = run_tape(f, cols, theta, model.m)
        jv = run_tape(fj, cols, theta, model.m)
    np.testing.assert_allclose(fv, model.predict(theta), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.column_stack(jv[1:]), model.jacobian(theta), rtol=1e-10, atol=1e-10
    )
    if len(model.linear_idx):
        assert sep is not None
        c, *phi = run_tape(sep, cols, theta, model.m)
        lin = model.linear_idx
        alpha = theta[lin]
        np.testing.assert_allclose(
            c + np.column_stack(phi) @ alpha, model.predict(theta), rtol=1e-10, atol=1e-10
        )


@pytest.mark.parametrize("eq", list(stringer2019.EQUATIONS))
def test_stringer2019_equation_tapes(eq):
    rng = np.random.default_rng(0)
    theta_x = np.sort(rng.uniform(0, 2 * np.pi, 50))
    pk = stringer2019.Peaks(b=0.2, a1=2.0, a2=0.5, m=1.3)
    th = (
        stringer2019.initial_params(eq, pk)
        * rng.uniform(0.8, 1.2, len(stringer2019.EQUATIONS[eq][0]))
        + 0.05
    )
    names = stringer2019.EQUATIONS[eq][0]
    model = SymbolicModel(
        eq,
        None,
        names,
        ["theta"],
        theta_x,
        linear=[],
        compiled=compile_sympy(*stringer2019.symbolic(eq)),
    )
    model.set_linear(model.detect_linear(th))
    assert {"b", "a1"} & {names[i] for i in model.linear_idx} or eq == "idea_132"
    check(model, th)


@pytest.mark.parametrize("family", ["exp_mixture", "sin_mixture", "coupled_tanh"])
def test_family_tapes(family):
    model = make_family(family, np.linspace(0, 1, 40), 7)
    check(model, np.random.default_rng(1).uniform(0.5, 1.5, 7))


def test_estimate_peaks_finds_preferred_angle():
    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    y = 0.1 + 2 * np.exp(-0.5 * (np.angle(np.exp(1j * (th - 2.0))) / 0.4) ** 2)
    pk = stringer2019.estimate_peaks(th, y)
    assert abs(pk.m - 2.0) < 0.3 and pk.a1 > 1.5
