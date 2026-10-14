"""Uncertainty calibration utilities for perturbation-response prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def perturbation_uncertainty(predictions: np.ndarray) -> np.ndarray:
    """Mean predictive variance across genes for each perturbation.

    Parameters
    ----------
    predictions
        Array with shape (n_members, n_perturbations, n_genes).
    """
    if predictions.ndim != 3:
        raise ValueError("predictions must have shape (members, perturbations, genes)")
    return predictions.var(axis=0, ddof=1).mean(axis=1)


def uncertainty_diagnostics(
    table: pd.DataFrame,
    uncertainty_col: str = "ensemble_variance",
    pearson_col: str = "pearson",
    rmse_col: str = "rmse",
    reliability_col: str = "intrinsic_reproducibility",
) -> dict[str, float | int]:
    """Quantify whether predictive uncertainty identifies model failures."""
    cols = [uncertainty_col, pearson_col, rmse_col]
    clean = table[cols + ([reliability_col] if reliability_col in table.columns else [])]
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna().copy()
    clean["correlation_error"] = 1.0 - clean[pearson_col]

    bad = clean[pearson_col] < clean[pearson_col].median()
    auc = roc_auc_score(bad.astype(int), clean[uncertainty_col]) if bad.nunique() == 2 else float("nan")

    out = {
        "n_test": int(clean.shape[0]),
        "spearman_uncertainty_vs_correlation_error": float(
            clean[uncertainty_col].corr(clean["correlation_error"], method="spearman")
        ),
        "spearman_uncertainty_vs_rmse": float(
            clean[uncertainty_col].corr(clean[rmse_col], method="spearman")
        ),
        "failure_detection_auc_below_median_pearson": float(auc),
        "median_pearson_lowest_uncertainty_quartile": float(
            clean.nsmallest(max(1, clean.shape[0] // 4), uncertainty_col)[pearson_col].median()
        ),
        "median_pearson_highest_uncertainty_quartile": float(
            clean.nlargest(max(1, clean.shape[0] // 4), uncertainty_col)[pearson_col].median()
        ),
    }
    if reliability_col in clean.columns:
        out["spearman_uncertainty_vs_intrinsic_reliability"] = float(
            clean[uncertainty_col].corr(clean[reliability_col], method="spearman")
        )
    return out
