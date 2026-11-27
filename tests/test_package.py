"""The installed package: compile, fit and Pool.fit_many."""

from pathlib import Path

import numpy as np
import pytest

import symbfitrs as sf
from bench.suites import nist

NIST = Path(__file__).resolve().parents[1] / "data" / "external" / "nist"
X = np.linspace(0, 5, 60)
Y = 2.0 * np.exp(-0.7 * X) + 0.3


def test_compile_names_params_and_linear_set():
    m = sf.compile("a*exp(-b*x) + c")
    assert (m.params, m.variables, m.linear) == (["a", "b", "c"], ["x"], ["a", "c"])
    assert sf.compile("a*exp(-b*x) + c", params=["c", "b", "a"]).params == ["c", "b", "a"]
    for bad in ("a*exp(-b*x", "a*foo(x)", "2*x", "a $ x"):
        with pytest.raises(ValueError):
            sf.compile(bad)


def test_fit_recovers_parameters():
    r = sf.compile("a*exp(-b*x) + c").fit(X, Y, budget_ms=50)
    assert r.success and r.status == "converged"
    np.testing.assert_allclose(r.params, [2.0, 0.7, 0.3], rtol=1e-10)
    assert r["b"] == pytest.approx(0.7) and set(r.as_dict()) == {"a", "b", "c"}
    assert r.loss < 1e-25 and r.nfev > 0 and 0 < r.time_ms < 50


def test_fit_input_forms():
    m = sf.compile("a*x + b*sin(k*t)", variables=["x", "t"])
    x, t = np.linspace(0, 1, 40), np.linspace(0, 3, 40)
    y = 1.5 * x - 0.5 * np.sin(2.0 * t)
    for cols in ({"x": x, "t": t}, np.vstack([x, t]), [x, t]):
        r = m.fit(cols, y, p0=[1, 1, 1.8], budget_ms=50)
        np.testing.assert_allclose(r.params, [1.5, -0.5, 2.0], rtol=1e-8)
    w = np.r_[np.ones(39), 0.0]  # a zero weight hides the outlier
    r = m.fit([x, t], np.r_[y[:-1], 99.0], p0=[1, 1, 1.8], weights=w, budget_ms=50)
    np.testing.assert_allclose(r.params, [1.5, -0.5, 2.0], rtol=1e-8)
    with pytest.raises(ValueError):
        m.fit(x, y)  # one column for two variables
    with pytest.raises(ValueError):
        m.fit([x, t], y, p0=[1, 2])
    with pytest.raises(ValueError):
        m.fit([x, t[:-1]], y)


def test_deadline_returns_best_so_far():
    m = sf.compile(" + ".join(f"a{k}*exp(-b{k}*x)" for k in range(12)))
    r = m.fit(X, Y, p0=np.linspace(0.1, 3, 24), budget_ms=0.01)
    assert r.status == "deadline" and r.params.shape == (24,)


def test_fit_many_keeps_order_and_isolates_errors():
    jobs = [
        ("a*exp(-b*x) + c", X, Y),
        ("a*exp(-b*x", X, Y),
        ("a*x + c", X, 3 * X - 1),
        ("a*exp(-b*x) + c", X, Y[:-1]),
        ("a*exp(-b*x) + c", X, -Y, [-1, 1, 0]),
    ] * 50
    out = sf.Pool(threads=4).fit_many(jobs, budget_ms=20, progress=False)
    assert len(out) == len(jobs)
    for i in range(0, len(jobs), 5):
        np.testing.assert_allclose(out[i].params, [2.0, 0.7, 0.3], rtol=1e-8)
        assert out[i + 1].status == "error" and "expected" in out[i + 1].message
        assert out[i + 2].as_dict() == pytest.approx({"a": 3.0, "c": -1.0})
        assert out[i + 3].status == "error" and not out[i + 3].success
        np.testing.assert_allclose(out[i + 4].params, [-2.0, 0.7, -0.3], rtol=1e-8)


@pytest.mark.skipif(not NIST.exists(), reason="run python -m bench.fetch (nist)")
def test_nist_certified_rss_from_both_starts():
    for nb in nist.load(NIST):
        m = sf.compile(nb.expr, nb.variables, nb.params)
        x = nb.x if nb.x.ndim == 1 else nb.x.T
        for start in ("start1", "start2"):
            r = m.fit(x, nb.y, p0=nb.starts[start], budget_ms=500)
            rss = r.loss * nb.y.size
            assert rss == pytest.approx(nb.certified_rss, rel=1e-6, abs=1e-20), (nb.name, start)


def test_thread_record_matches_benchmark_schema():
    """bench.fitter.package_run turns a FitResult into the record bench.scoring reads."""
    from bench.fitter.package_run import expression, record
    from bench.models import SymbolicModel
    from bench.problem import Problem

    model = SymbolicModel("exp", "a*exp(-b*x) + c", ["a", "b", "c"], ["x"], X)
    e, variables, params = expression(model)
    r = sf.compile(e, variables, params).fit(X, Y, p0=[1, 1, 0], budget_ms=50)
    pb = Problem("p", "d", "m", "c", model, Y, {"s": np.ones(3)}, ref_mode="zero")
    rec = record(pb, "s", [1, 1, 0], r, "symbfitrs_t2", 1.0)
    assert (rec["solver"], rec["problem"], rec["status"]) == ("symbfitrs_t2", "p", "converged")
    # the package reports mean squared residual; the record divides by the problem's denominator
    assert rec["best_loss"] == pytest.approx(r.loss / pb.denom)
    assert rec["hist_loss"][-1] == pytest.approx(rec["best_loss"])
    assert rec["hist_t_ms"] == sorted(rec["hist_t_ms"]) and len(rec["hist_t_ms"]) > 1
    assert rec["loss0"] == pytest.approx(pb.loss_at(np.array([1.0, 1.0, 0.0])))
    assert rec["theta_at"] is None and rec["n_linear"] == 2
