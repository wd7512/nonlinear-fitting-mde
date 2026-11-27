"""The installed package: compile, fit and Pool.fit_many."""

from pathlib import Path

import numpy as np
import pytest
from scipy.special import erfc

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


def test_predict_and_history():
    m = sf.compile("a*exp(-b*x) + c")
    np.testing.assert_allclose(m.predict(X, [2.0, 0.7, 0.3]), Y, rtol=1e-15)
    with pytest.raises(ValueError):
        m.predict(X, [1.0, 2.0])
    r = m.fit(X, Y, budget_ms=50)
    assert r.history_loss.size > 0 and r.history_loss[-1] == r.loss
    assert np.all(np.diff(r.history_loss) < 0) and np.all(np.diff(r.history_ms) >= 0)
    assert r.history_ms[-1] <= r.time_ms and r.n_nonfinite == 0


def test_wrap_step_expwerfc_and_declared_pi():
    x = np.linspace(-6, 6, 80)
    m = sf.compile("a*wrap(x - b) + c*step(x - 1) + expwerfc(d*x, 2 - x)")
    np.testing.assert_allclose(
        m.predict(x, [1.5, 0.5, 2.0, 0.3]),
        1.5 * np.angle(np.exp(1j * (x - 0.5))) + 2.0 * (x >= 1) + np.exp(0.3 * x) * erfc(2 - x),
        rtol=1e-13,
    )
    assert sf.compile("pi*x", params=["pi"]).predict(X, [2.0])[1] == 2.0 * X[1]


def test_fit_many_takes_models_with_their_own_variables_and_weights():
    m = sf.compile("a*u + b", variables=["u"])
    w = np.r_[np.ones(59), 0.0]
    jobs = [(m, X, np.r_[3 * X[:-1] + 1, 99.0], None, w), ("a*exp(-b*x) + c", X, Y)]
    out = sf.Pool(threads=2).fit_many(jobs, budget_ms=50, progress=False)
    np.testing.assert_allclose(out[0].params, [3.0, 1.0], rtol=1e-10)
    np.testing.assert_allclose(out[1].params, [2.0, 0.7, 0.3], rtol=1e-8)


def test_atan2_matches_numpy():
    x = np.linspace(-3, 3, 41)
    m = sf.compile("atan2(a*x - 1, b)")
    np.testing.assert_allclose(m.predict(x, [1.2, -0.5]), np.arctan2(1.2 * x - 1, -0.5), rtol=1e-15)


def test_package_records_match_benchmark_schema(tmp_path):
    """bench.fitter.run.bench_package turns FitResults into the records bench.scoring reads."""
    from bench.fitter.export import export_problems
    from bench.fitter.run import bench_package
    from bench.models import SymbolicModel
    from bench.problem import Problem

    model = SymbolicModel("exp", "a*exp(-b*x) + c", ["a", "b", "c"], ["x"], X)
    pb = Problem("p", "d", "m", "c", model, Y, {"s": np.ones(3)}, ref_mode="zero")
    export_problems([pb], {"config": {}}, tmp_path / "problems.json")
    recs = bench_package(tmp_path / "problems.json", tmp_path / "r.jsonl", [50], solver="sf_t2")
    (rec,) = [r for r in recs if r["type"] == "fit"]
    assert (rec["solver"], rec["problem"], rec["status"]) == ("sf_t2", "p", "converged")
    assert rec["hist_loss"][-1] == pytest.approx(rec["best_loss"])
    assert rec["hist_t_ms"] == sorted(rec["hist_t_ms"]) and len(rec["hist_t_ms"]) > 1
    assert rec["loss0"] == pytest.approx(pb.loss_at(np.ones(3)))
    assert rec["n_linear"] == 2
