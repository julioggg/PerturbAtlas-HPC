import numpy as np
import pandas as pd

from perturbatlas.uncertainty import perturbation_uncertainty, uncertainty_diagnostics


def test_perturbation_uncertainty_and_diagnostics():
    preds = np.array([
        [[0.0, 1.0], [1.0, 1.0]],
        [[0.2, 0.8], [2.0, 0.0]],
        [[-0.1, 1.1], [3.0, -1.0]],
    ])
    u = perturbation_uncertainty(preds)
    assert u.shape == (2,)
    df = pd.DataFrame({
        "ensemble_variance": [u[0], u[1], 0.2, 0.4],
        "pearson": [0.9, 0.2, 0.7, 0.4],
        "rmse": [0.1, 0.8, 0.3, 0.6],
        "intrinsic_reproducibility": [0.9, 0.2, 0.7, 0.4],
    })
    out = uncertainty_diagnostics(df)
    assert out["n_test"] == 4
    assert "failure_detection_auc_below_median_pearson" in out
