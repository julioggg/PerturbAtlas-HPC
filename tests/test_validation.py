import pandas as pd

from perturbatlas.validation import ranking_diagnostics


def test_ranking_diagnostics():
    df = pd.DataFrame(
        {
            "n_cells": [10, 20, 30],
            "l2_effect": [3.0, 2.0, 1.0],
            "noise_adjusted_l2": [1.0, 3.0, 2.0],
            "l2_to_null_ratio": [1.2, 2.0, 1.5],
        },
        index=["A", "B", "C"],
    )
    out, summary = ranking_diagnostics(df)
    assert "rank_naive_l2" in out.columns
    assert "rank_adjusted_l2" in out.columns
    assert summary["n_perturbations"] == 3
    assert 0.0 <= summary["top100_jaccard"] <= 1.0
