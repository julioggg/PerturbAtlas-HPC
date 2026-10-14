import pandas as pd

from perturbatlas.reproducibility_biology import summarize_split_half


def test_summarize_split_half():
    df = pd.DataFrame(
        {
            "perturbation": ["A", "A", "B", "B"],
            "repeat": [0, 1, 0, 1],
            "n_cells": [10, 10, 20, 20],
            "pearson": [0.8, 0.6, 0.2, 0.4],
            "cosine": [0.9, 0.7, 0.3, 0.5],
            "rmse": [1.0, 1.2, 2.0, 1.8],
            "nrmse": [0.5, 0.6, 1.0, 0.9],
        }
    )
    out = summarize_split_half(df)
    assert out.loc["A", "pearson_median"] == 0.7
    assert out.loc["B", "n_cells"] == 20
