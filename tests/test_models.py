import numpy as np
import pytest

from bench import config
from bench.models import SymbolicModel
from bench.suites.synthetic import make_problem


def fd_jacobian(model, theta, h=1e-6):
    cols = []
    for e in np.eye(model.p):
        cols.append((model.predict(theta + h * e) - model.predict(theta - h * e)) / (2 * h))
    return np.column_stack(cols)


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(config.SEED)
    return [make_problem(f, p, rng) for f in config.FAMILIES for p in config.PARAM_COUNTS]


def test_grid_shape(synthetic):
    assert len(synthetic) == 12
    for pb in synthetic:
        assert pb.m == max(64, 4 * pb.p)
        assert set(pb.starts) == {"near", "far"}
        assert pb.ref_loss == 0.0


def test_family_jacobians(synthetic):
    for pb in synthetic:
        th = pb.starts["near"]
        np.testing.assert_allclose(
            pb.model.jacobian(th), fd_jacobian(pb.model, th), rtol=1e-5, atol=1e-6
        )


def test_true_parameters_give_zero_loss(synthetic):
    for pb in synthetic:
        assert pb.loss_at(pb.ref_params) == 0.0


def test_symbolic_model_jacobian_and_linear_detection():
    x = np.linspace(0, 3, 40)
    m = SymbolicModel("d", "A*exp(-k*x)*cos(w*x) + c", ["A", "k", "w", "c"], ["x"], x)
    assert [m.param_names[i] for i in m.linear_idx] == ["A", "c"]
    th = np.array([1.3, 0.4, 2.0, 0.1])
    np.testing.assert_allclose(m.jacobian(th), fd_jacobian(m, th), rtol=1e-6, atol=1e-8)
    # constant derivative columns are broadcast
    assert m.jacobian(th)[:, 3].tolist() == [1.0] * 40


def test_symbolic_multivariate_and_override():
    X = np.random.default_rng(0).uniform(size=(30, 2))
    m = SymbolicModel("mv", "a*u + b*exp(-c*v)", ["a", "b", "c"], ["u", "v"], X, linear=[])
    assert m.n_linear == 0
    th = np.array([0.5, 2.0, 1.5])
    np.testing.assert_allclose(m.jacobian(th), fd_jacobian(m, th), rtol=1e-6, atol=1e-8)


def test_symbolic_rejects_unknown_symbols():
    with pytest.raises(ValueError, match="undeclared"):
        SymbolicModel("bad", "a*exp(-q*x)", ["a"], ["x"], np.arange(5.0))
