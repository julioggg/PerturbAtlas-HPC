import pandas as pd

from perturbatlas.reliability import build_reliability_table


def test_build_reliability_table():
    effects = pd.DataFrame(
        {
            "l2_effect": [3.0, 1.0],
            "noise_adjusted_l2": [2.5, 0.5],
            "l2_to_null_ratio": [2.0, 1.1],
        },
        index=["A", "B"],
    )
    repro = pd.DataFrame(
        {
            "n_cells": [100, 80],
            "pearson_median": [0.9, 0.2],
            "nrmse_median": [0.3, 1.3],
        },
        index=["A", "B"],
    )
    out, summary = build_reliability_table(effects, repro)
    assert out.loc["A", "reproducible_r_gt_0_8"]
    assert summary["n_perturbations"] == 2
