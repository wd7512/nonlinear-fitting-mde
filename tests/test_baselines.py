import contextlib

import numpy as np
import pytest

from bench.baselines import BASELINES, baseline
from bench.models import SymbolicModel
from bench.problem import Problem
from bench.runner import run_job
from bench.suites.synthetic import make_problem


@pytest.mark.parametrize("name", list(BASELINES))
@pytest.mark.parametrize("weighted", [False, True])
def test_baselines_fit_an_easy_problem(name, weighted):
    x = np.linspace(0, 3, 50)
    model = SymbolicModel("d", "A*exp(-k*x) + c", ["A", "k", "c"], ["x"], x)
    y = model.predict(np.array([1.2, 0.5, 0.2]))
    w = np.linspace(1, 3, 50) if weighted else None
    pb = Problem("t/d/0", "t", "d", "0", model, y, {"s": np.array([1.0, 0.4, 0.0])}, weights=w)
    rec = run_job(pb, "s", name, deadline_ms=500, budgets_ms=[500])
    assert rec["status"] == "returned", rec["error"]
    assert rec["best_loss"] < 1e-12
    assert rec["hist_t_ms"] == sorted(rec["hist_t_ms"])


def _problem():
    return make_problem("exp_mixture", 2, np.random.default_rng(0))


@pytest.fixture
def register():
    added = []

    def add(fn):
        added.append(fn.__name__)
        return baseline(fn)

    yield add
    for name in added:
        del BASELINES[name]


def test_registered_function_receives_the_job(register):
    seen = {}

    @register
    def stay_put(job):
        seen.update(p0=job.p0, budget=job.budget_ms)
        return job.p0

    rec = run_job(_problem(), "near", "stay_put", deadline_ms=50)
    assert rec["status"] == "returned" and rec["solver"] == "stay_put"
    assert seen["budget"] == 50 and seen["p0"].shape == (2,)


def test_deadline_stops_a_baseline(register):
    @register
    def forever(job):
        while True:
            job.residual(job.p0)

    rec = run_job(_problem(), "near", "forever", deadline_ms=5)
    assert rec["status"] == "deadline" and rec["error"] is None


@pytest.mark.parametrize("bad", ["raise", "nan"])
def test_failures_are_recorded(register, bad):
    @register
    def broken(job):
        if bad == "raise":
            raise RuntimeError("boom")
        return np.full_like(job.p0, np.nan)

    rec = run_job(_problem(), "near", "broken", deadline_ms=50)
    assert rec["status"] == "failure" and rec["error"]


def test_except_exception_cannot_swallow_the_deadline(register):
    @register
    def stubborn(job):
        while True:
            with contextlib.suppress(Exception):  # the pattern under test
                job.residual(job.p0)

    rec = run_job(_problem(), "near", "stubborn", deadline_ms=5)
    assert rec["status"] == "deadline"


def test_harness_errors_are_not_baseline_failures(register, monkeypatch):
    @register
    def stay_put(job):
        return job.p0

    pb = _problem()
    calls = []
    real = pb.model.predict

    def predict(theta):  # fine for the start point, broken in the post-return check
        calls.append(1)
        return real(theta) if len(calls) == 1 else 1 / 0

    monkeypatch.setattr(pb.model, "predict", predict)
    with pytest.raises(ZeroDivisionError):
        run_job(pb, "near", "stay_put", deadline_ms=50)
