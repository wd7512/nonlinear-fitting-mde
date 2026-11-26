import numpy as np
import pytest

from nlfit_mde import config
from nlfit_mde.datasets.synthetic import make_problem
from nlfit_mde.models import SymbolicModel
from nlfit_mde.problem import Problem
from nlfit_mde.runner import run_job


@pytest.mark.parametrize("solver", config.SOLVER_NAMES)
def test_solvers_fit_easy_problem_from_near_start(solver):
    pb = make_problem("exp_mixture", 2, np.random.default_rng(0))
    rec = run_job(pb, "near", solver, deadline_ms=200)
    assert rec["status"] != "failure", rec["error"]
    assert rec["best_loss"] < 1e-6
    assert rec["hist_t_ms"] == sorted(rec["hist_t_ms"])


def test_varpro_labels_fallback():
    pb = make_problem("coupled_tanh", 2, np.random.default_rng(0))
    assert run_job(pb, "near", "varpro_lm")["solver_label"] == "varpro_lm (LM fallback)"


def _symbolic_problem(weights=None):
    x = np.linspace(0, 3, 50)
    model = SymbolicModel("d", "A*exp(-k*x)*cos(w*x) + c", ["A", "k", "w", "c"], ["x"], x)
    y = model.predict(np.array([1.2, 0.5, 2.5, 0.2]))
    start = {"s": np.array([1.0, 0.4, 2.3, 0.0])}
    return Problem("t/d/0", "t", "d", "0", model, y, start, np.ones(4), weights=weights)


@pytest.mark.parametrize("solver", ["varpro_lm", "lm", "gauss_newton"])
@pytest.mark.parametrize("weighted", [False, True])
def test_symbolic_affine_separable_fit(solver, weighted):
    w = np.linspace(1, 3, 50) if weighted else None
    rec = run_job(_symbolic_problem(w), "s", solver, deadline_ms=500)
    assert rec["status"] != "failure", rec["error"]
    assert rec["best_loss"] < 1e-12


def test_varpro_fully_linear_model():
    x = np.linspace(0, 1, 20)
    model = SymbolicModel("lin", "a + b*x", ["a", "b"], ["x"], x)
    pb = Problem("t/lin/0", "t", "lin", "0", model, 1 + 2 * x, {"s": np.zeros(2)}, np.ones(2))
    rec = run_job(pb, "s", "varpro_lm")
    assert rec["solver_label"] == "varpro_lm (linear solve)"
    assert rec["best_loss"] < 1e-28
