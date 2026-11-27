"""Exported expression strings, compiled by the package, must reproduce the Python models."""

from pathlib import Path

import numpy as np
import pytest
import sympy as sp

import symbfitrs as sf
from bench.config import CONFIG_DIR
from bench.experiment import load_config
from bench.fitter.export import NotExportable, symbolic_form, to_expr
from bench.models import SymbolicModel, VectorResidualModel
from bench.models.families import make_family
from bench.models.symbolic import MODULES, compile_sympy, expwerfc, step, wrap
from bench.suites import build_problems, mgh, stringer2019

NIST = Path(__file__).resolve().parents[1] / "data" / "external" / "nist"


def check(model, theta, rtol=1e-12):
    expr, variables, params, cols = symbolic_form(model)
    m = sf.compile(expr, variables, params)
    assert m.params == params and m.variables == variables
    with np.errstate(all="ignore"):
        want = model.predict(np.asarray(theta, float))
    np.testing.assert_allclose(m.predict(cols, theta), want, rtol=rtol, atol=1e-12)


def test_printer_covers_the_grammar():
    a, b, x = sp.symbols("a b x", real=True)
    cases = {
        sp.Abs(a * x): "abs(a*x)",
        sp.Max(a, b, x): None,
        sp.Rational(1, 3) * x: None,
        sp.E * a: None,
        sp.Float(0.1) * x: None,
        expwerfc(a, x - b) + wrap(x - a) + step(x - b): None,
    }
    th, xs = [0.7, -0.4], np.linspace(-2, 2, 9)
    for e, text in cases.items():
        s = to_expr(e)
        if text is not None:
            assert s == text
        f = sp.lambdify([a, b, x], e, modules=MODULES)
        got = sf.compile(s, ["x"], ["a", "b"]).predict(xs, th)
        np.testing.assert_allclose(got, np.broadcast_to(f(*th, xs), xs.shape), rtol=1e-14)
    with pytest.raises(NotExportable):
        to_expr(sp.oo * a)


@pytest.mark.parametrize("eq", list(stringer2019.EQUATIONS))
def test_stringer2019_equations(eq):
    rng = np.random.default_rng(0)
    theta_x = np.sort(rng.uniform(0, 2 * np.pi, 50))
    pk = stringer2019.Peaks(b=0.2, a1=2.0, a2=0.5, m=1.3)
    names = stringer2019.EQUATIONS[eq][0]
    th = stringer2019.initial_params(eq, pk) * rng.uniform(0.8, 1.2, len(names)) + 0.05
    model = SymbolicModel(
        eq, None, names, ["theta"], theta_x, linear=[],
        compiled=compile_sympy(*stringer2019.symbolic(eq)),
    )  # fmt: skip
    check(model, th, rtol=1e-11)


@pytest.mark.parametrize("family", ["exp_mixture", "sin_mixture", "coupled_tanh"])
def test_families(family):
    model = make_family(family, np.linspace(0, 1, 40), 7)
    check(model, np.random.default_rng(1).uniform(0.5, 1.5, 7))


@pytest.mark.parametrize("mp", mgh.load(), ids=lambda mp: mp.key)
def test_mgh_indicator_encoding(mp):
    model = VectorResidualModel(mp.key, mp.residuals, mp.symbols, center=mp.x0)
    x0 = mp.x0 if np.any(mp.x0) else np.full(mp.x0.shape, 0.3)
    check(model, x0 * 1.1, rtol=1e-11)


@pytest.mark.skipif(not NIST.exists(), reason="run python -m bench.fetch (nist)")
def test_nist_exports_every_problem():
    cfg = load_config(CONFIG_DIR / "nist_strd.toml")
    for pb in build_problems(cfg, np.random.default_rng(cfg.seed)):
        for theta in pb.starts.values():
            check(pb.model, theta, rtol=1e-11)


def test_estimate_peaks_finds_preferred_angle():
    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    y = 0.1 + 2 * np.exp(-0.5 * (np.angle(np.exp(1j * (th - 2.0))) / 0.4) ** 2)
    pk = stringer2019.estimate_peaks(th, y)
    assert abs(pk.m - 2.0) < 0.3 and pk.a1 > 1.5


def test_package_records_match_benchmark_schema(tmp_path):
    """bench.fitter.run.bench_package turns FitResults into the records bench.scoring reads."""
    from bench.fitter.export import export_problems
    from bench.fitter.run import bench_package
    from bench.problem import Problem

    x = np.linspace(0, 5, 60)
    model = SymbolicModel("exp", "a*exp(-b*x) + c", ["a", "b", "c"], ["x"], x)
    pb = Problem(
        "p", "d", "m", "c", model, 2.0 * np.exp(-0.7 * x) + 0.3, {"s": np.ones(3)}, ref_mode="zero"
    )
    export_problems([pb], {"config": {}}, tmp_path / "problems.json")
    recs = bench_package(tmp_path / "problems.json", tmp_path / "r.jsonl", [50], {"sf_t2": 2})
    (rec,) = [r for r in recs if r["type"] == "fit"]
    assert (rec["solver"], rec["problem"], rec["status"]) == ("sf_t2", "p", "converged")
    assert rec["hist_loss"][-1] == pytest.approx(rec["best_loss"])
    assert rec["hist_t_ms"] == sorted(rec["hist_t_ms"]) and len(rec["hist_t_ms"]) > 1
    assert rec["loss0"] == pytest.approx(pb.loss_at(np.ones(3)))
    assert rec["n_linear"] == 2
