import pandas as pd

from perturbatlas.association import reliability_associations


def test_reliability_associations():
    df = pd.DataFrame(
        {
            "n_cells": [30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
            "l2_effect": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
            "noise_adjusted_l2": [0.5, 1, 1.2, 2, 2.2, 3, 3.1, 4, 4.2, 5],
            "l2_to_null_ratio": [1.0, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.1, 2.3, 2.5],
            "pearson_median": [0.0, 0.1, 0.15, 0.25, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7],
        }
    )
    assoc, within, summary = reliability_associations(df)
    assert "partial_spearman_controlling_log_n" in assoc.columns
    assert summary["n_perturbations"] == 10
    assert within.shape[0] >= 1
