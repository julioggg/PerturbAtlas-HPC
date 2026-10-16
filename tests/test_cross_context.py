import pandas as pd

from perturbatlas.cross_context import cross_context_summary, merge_context_tables


def test_merge_and_summary():
    idx = ["A", "B", "C"]
    effects_a = pd.DataFrame({
        "n_cells": [50, 60, 70],
        "l2_effect": [1.0, 2.0, 3.0],
        "expected_null_l2": [0.5, 0.5, 0.5],
        "noise_adjusted_l2": [0.8, 1.8, 2.8],
        "l2_to_null_ratio": [2.0, 4.0, 6.0],
    }, index=idx)
    repro_a = pd.DataFrame({
        "pearson_median": [0.1, 0.2, 0.3],
        "cosine_median": [0.1, 0.2, 0.3],
        "nrmse_median": [2.0, 1.5, 1.0],
    }, index=idx)

    effects_b = effects_a.iloc[::-1].copy()
    repro_b = repro_a.iloc[::-1].copy()

    merged = merge_context_tables(
        effects_a, repro_a, effects_b, repro_b,
        suffix_a="a", suffix_b="b"
    )
    assert list(merged.index) == idx
    assert "pearson_median_a" in merged
    assert "pearson_median_b" in merged

    summary = cross_context_summary(merged, suffix_a="a", suffix_b="b")
    assert summary["n_shared_perturbations"] == 3
    assert -1.0 <= summary["spearman_pearson_reliability"] <= 1.0
