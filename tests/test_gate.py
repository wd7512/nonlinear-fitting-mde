import numpy as np
import pandas as pd

from nlfit_mde.rust_bridge.gate import _sample, _side, best_threshold, pair


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


def test_compare_finds_a_planted_threshold():
    rows = []
    for i in range(40):
        p = i % 20
        common = {"problem": f"p{i}", "start": "a", "suite": "s", "feat_p": p, "ok_5": True}
        rows.append({**common, "solver": "base", "ttt": 2.0})
        # the challenger is faster for small p and fails for large p
        rows.append({**common, "solver": "chal", "ok_5": p < 8, "ttt": 1.0})
    t = pair(pd.DataFrame(rows), "base", "chal")
    assert (t.outcome == "win").sum() == 16
    rule = best_threshold(t, "feat_p")
    assert rule["rule"] == "<= 7.5"
    assert rule["success"] == 1.0
