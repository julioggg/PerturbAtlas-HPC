import pandas as pd

from perturbatlas.nulls import interpolate_empirical_thresholds


def test_interpolate_empirical_thresholds():
    effects = pd.DataFrame(
        {
            "n_cells": [10, 20, 30],
            "l2_effect": [5.0, 3.0, 1.0],
            "expected_null_l2": [4.0, 2.0, 1.0],
        },
        index=["A", "B", "C"],
    )
    null = pd.DataFrame(
        {
            "n_cells": [10, 30],
            "null_median_l2": [3.0, 1.0],
            "null_q90_l2": [4.0, 1.5],
            "null_q95_l2": [4.5, 2.0],
            "null_q99_l2": [5.5, 2.5],
        }
    )
    out = interpolate_empirical_thresholds(effects, null)
    assert "above_empirical_q95" in out.columns
    assert "empirical_l2_over_q95" in out.columns
    assert bool(out.loc["A", "above_empirical_q95"])
