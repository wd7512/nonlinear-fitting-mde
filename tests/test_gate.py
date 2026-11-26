import numpy as np

from bench.gate import _sample, _side


def test_side_is_deterministic_per_group():
    assert _side("mgh/Rosenbrock") == _side("mgh/Rosenbrock")
    sides = {_side(f"suite/model{i}") for i in range(50)}
    assert sides == {"dev", "test"}


def test_sample_spreads_over_groups():
    cases = [(i, 0) for i in range(30)]
    groups = ["a"] * 20 + ["b"] * 5 + ["c"] * 5
    picked = _sample(cases, groups, 6, np.random.default_rng(0))
    got = [groups[i] for i, _ in picked]
    assert sorted(got) == ["a", "a", "b", "b", "c", "c"]
