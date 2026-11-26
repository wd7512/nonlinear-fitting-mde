"""Transcription checks for the external benchmark suites.

MGH and the model kernels need no downloads. NIST, SAGE-Fit and FitBenchmarking tests
are skipped unless ``python -m bench.fetch`` has populated ``data/external``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import least_squares

from bench.models import (
    SymbolicModel,
    VectorResidualModel,
    python_function_model,
)
from bench.models.families import make_family
from bench.scoring import tolerance
from bench.suites import fitbenchmarking, mgh, nist

EXT = Path(__file__).resolve().parents[1] / "data" / "external"


def needs(d):
    return pytest.mark.skipif(not (EXT / d).exists(), reason=f"run python -m bench.fetch ({d})")


def fd(f, theta, h=1e-6):
    return np.column_stack(
        [(f(theta + h * e) - f(theta - h * e)) / (2 * h) for e in np.eye(theta.size)]
    )


# ---- MGH ---------------------------------------------------------------------------------
# Published f* is reached from x0 except at these well-known local minima.
MGH_LOCAL = {"02_freudenstein_roth": 48.9842, "26_trigonometric_n10": 2.79506e-5}


@pytest.fixture(scope="module")
def mgh_problems():
    return mgh.load()


def test_mgh_registry(mgh_problems):
    assert len({p.number for p in mgh_problems}) == 35
    for p in mgh_problems:
        assert p.m >= p.n == len(p.x0) == len(p.symbols)


def test_mgh_fstar(mgh_problems):
    for p in mgh_problems:
        model = VectorResidualModel(p.key, p.residuals, p.symbols, center=p.x0)
        f = min(
            2
            * least_squares(
                model.predict,
                p.x0,
                jac=model.jacobian,
                method=meth,
                xtol=1e-15,
                ftol=1e-15,
                gtol=1e-15,
                max_nfev=20000,
            ).cost
            for meth in ("lm", "trf")
        )
        target = MGH_LOCAL.get(p.key, p.fstar)
        assert f <= target * (1 + 1e-4) + 1e-10, (p.key, f, target)


def test_mgh_jacobians(mgh_problems):
    for p in mgh_problems:
        model = VectorResidualModel(p.key, p.residuals, p.symbols, center=p.x0)
        th = p.x0 + 0.1
        J, Jfd = model.jacobian(th), fd(model.predict, th)
        np.testing.assert_allclose(J, Jfd, rtol=1e-4, atol=1e-5 * max(1.0, np.abs(J).max()))


# ---- models ------------------------------------------------------------------------------
def test_odd_p_mixture_offset():
    x = np.linspace(0, 1, 50)
    m = make_family("exp_mixture", x, 5)
    assert m.n_linear == 3 and m.param_names[-1] == "c"
    th = np.array([1.0, 2.0, 0.5, 3.0, 0.3])
    np.testing.assert_allclose(m.jacobian(th), fd(m.predict, th), rtol=1e-6, atol=1e-8)


def test_numeric_linear_detection():
    m = SymbolicModel(
        "d",
        "A*exp(-k*x)*cos(w*x) + c + b*x**2",
        ["A", "k", "w", "c", "b"],
        ["x"],
        np.linspace(0, 3, 40),
    )
    assert [m.param_names[i] for i in m.linear_idx] == ["A", "c", "b"]


def test_python_function_traced_and_fallback():
    x = np.linspace(0.1, 2, 30)
    traced = python_function_model(
        "t",
        "def equation(t, P, params):\n    return params[0] * P * (1 - P / params[1])",
        ["t", "P"],
        np.column_stack([x, 2 * x]),
    )
    assert traced.jacobian_kind == "analytic" and traced.p == 2 and traced.n_linear == 1
    src = "def equation(t, P, params):\n    return params[0] * np.gradient(P) + params[1] * np.exp(-params[2] * t)"
    fb = python_function_model("f", src, ["t", "P"], np.column_stack([x, x**2]))
    assert fb.jacobian_kind == "finite_difference"
    assert sorted(fb.linear_idx.tolist()) == [0, 1]
    th = np.array([0.5, 1.0, 0.7])
    np.testing.assert_allclose(fb.jacobian(th), fd(fb.predict, th), rtol=1e-6, atol=1e-9)


def test_reduction_tolerance():
    df = pd.DataFrame({"ref": [1.0, 0.0], "loss0": [11.0, None]})
    tol = tolerance(df, 1e-3, "reduction")
    assert tol.iloc[0] == pytest.approx(1e-2) and np.isinf(tol.iloc[1])


# ---- NIST --------------------------------------------------------------------------------
@needs("nist")
def test_nist_certified_rss():
    problems = nist.load(EXT / "nist")
    assert len(problems) == 27
    for p in problems:
        m = SymbolicModel(p.name, p.expr, p.params, p.variables, p.x)
        r = m.predict(p.certified) - p.y
        rss = float(r @ r)
        if p.name == "Lanczos1":  # 1.4e-25 is below what 11-digit parameters reproduce
            assert rss < 1e-20
        else:
            assert rss == pytest.approx(p.certified_rss, rel=1e-9), p.name


def test_lre():
    assert nist.lre([1.000001], [1.0])[0] == pytest.approx(6.0, abs=1e-6)
    assert nist.lre([1.0], [1.0])[0] == 11.0


# ---- FitBenchmarking ---------------------------------------------------------------------
def test_mantid_function_parse():
    expr, params, start, names = fitbenchmarking.parse_function(
        "name=LinearBackground,A0=1,A1=2;name=Gaussian,Height=3,PeakCentre=0.5,Sigma=0.1"
    )
    assert names == ["LinearBackground", "Gaussian"] and start.tolist() == [1, 2, 3, 0.5, 0.1]
    with pytest.raises(fitbenchmarking.Unsupported):
        fitbenchmarking.parse_function("name=BSpline,Order=3")


def test_back_to_back_exponential_stable():
    expr, params, start, _ = fitbenchmarking.parse_function(
        "name=BackToBackExponential,I=100,A=1,B=0.05,X0=1000,S=5"
    )
    x = np.linspace(800, 1600, 4000)  # B = 0.05: long right tail
    from bench.models.symbolic import compile_sympy

    m = SymbolicModel(
        "b",
        None,
        [s.name for s in params],
        ["x"],
        x,
        compiled=compile_sympy(expr, params, [fitbenchmarking.X]),
        linear=[],
    )
    f = m.predict(start)
    assert np.all(np.isfinite(f)) and np.all(f >= 0)
    assert np.trapezoid(f, x) == pytest.approx(100, rel=1e-3)  # I is the integrated intensity
    np.testing.assert_allclose(m.jacobian(start), fd(m.predict, start, 1e-5), rtol=1e-4, atol=1e-8)


@needs("fitbenchmarking")
def test_fitbenchmarking_scan():
    root = EXT / "fitbenchmarking/examples/benchmark_problems"
    ok, skipped = fitbenchmarking.scan(
        root, ["Neutron", "Muon", "simple_tests", "Mantid_System_Test_Data"]
    )
    assert len(ok) == 69 and len(skipped) == 6


@needs("sagefit")
def test_sagefit_candidates_load():
    import json

    from bench.suites.sagefit import BANK

    bank = json.loads((EXT / "sagefit" / BANK).read_text())
    assert len(bank) == 20 and all(len(p["events"]) == 10 for p in bank)
    prob = bank[0]
    data = pd.read_csv(EXT / "sagefit/data" / prob["equation_id"] / "train.csv")
    for ev in prob["events"]:
        m = python_function_model("c", ev["function"], ["t", "P"], data[["t", "P"]].to_numpy())
        assert m.predict(np.ones(m.p)).shape == (len(data),)
