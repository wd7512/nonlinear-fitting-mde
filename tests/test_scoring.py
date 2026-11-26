import numpy as np
import pandas as pd

from bench.scoring import (
    best_at,
    best_so_far_curve,
    fill_reference,
    time_to_target,
)


def test_late_results_not_credited_early():
    t = np.array([0.1, 3.0, 30.0])
    loss = np.array([1.0, 1e-3, 1e-12])
    assert best_at(t, loss, 1) == 1.0
    assert best_at(t, loss, 5) == 1e-3
    assert best_at(t, loss, 25) == 1e-3  # 30 ms overrun is not credited at 25 ms
    assert np.isnan(time_to_target(t, loss, 1e-10, 25))


def test_best_so_far_curve():
    out = best_so_far_curve(
        np.array([1.0, 2.0, 3.0]), np.array([5.0, 7.0, 1.0]), np.array([0.5, 1.0, 2.5, 10.0])
    )
    assert np.isinf(out[0])
    np.testing.assert_array_equal(out[1:], [5.0, 5.0, 1.0])


def test_reference_modes():
    df = pd.DataFrame(
        {
            "problem": ["a", "a", "b", "c"],
            "ref_mode": ["best_found", "best_found", "given", "none"],
            "ref_loss": [None, None, 0.5, None],
            "best_loss": [0.3, 0.2, 0.7, 0.9],
        }
    )
    assert fill_reference(df).ref.tolist() == [0.2, 0.2, 0.5, 0.0]
