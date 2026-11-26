import json

import numpy as np

from bench.gate import RULE_COLS, _sample, _side, score_gate


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


def _fit(ref):
    return {
        "type": "fit", "problem": "p", "dataset": "d", "model": "m", "start": "s",
        "solver": "symbfitrs", "n_nonfinite": 0, "loss0": 1.0, "best_loss": 0.5,
        "hist_t_ms": [0.1], "hist_loss": [0.5], "ref_loss": ref, "ref_mode": "given",
        "gate_mode": "absolute", "gate_target": 0.1, "gate_floor": 1e-20,
    }  # fmt: skip


def test_rules_replace_the_stored_reference(tmp_path):
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps(_fit(0.0)) + "\n")  # loss 0.5 misses a reference of 0
    assert not score_gate(path, [1.0])["ok_1"].iloc[0]
    rules = score_gate(path, [1.0]).assign(ref_loss=0.45).groupby("problem")[RULE_COLS].first()
    assert score_gate(path, [1.0], rules)["ok_1"].iloc[0]
