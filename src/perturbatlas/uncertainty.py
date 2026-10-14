"""Uncertainty calibration utilities for perturbation-response prediction."""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def perturbation_uncertainty(predictions: np.ndarray) -> np.ndarray:
    """Mean predictive variance across genes for each perturbation."""
    if predictions.ndim != 3:
        raise ValueError("predictions must have shape (members, perturbations, genes)")
    return predictions.var(axis=0, ddof=1).mean(axis=1)


def relative_ensemble_variance(predictions: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale-normalized ensemble variance."""
    mean_pred = predictions.mean(axis=0)
    var = perturbation_uncertainty(predictions)
    signal_power = np.mean(mean_pred.astype(np.float64) ** 2, axis=1)
    return var / (signal_power + eps)


def pairwise_correlation_disagreement(predictions: np.ndarray) -> np.ndarray:
    """One minus mean pairwise Pearson correlation among ensemble members."""
    if predictions.ndim != 3:
        raise ValueError("predictions must have shape (members, perturbations, genes)")
    m, n, _ = predictions.shape
    vals = np.zeros((m * (m - 1) // 2, n), dtype=np.float64)
    k = 0
    for i, j in itertools.combinations(range(m), 2):
        a = predictions[i].astype(np.float64)
        b = predictions[j].astype(np.float64)
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        corr = np.full(n, np.nan, dtype=float)
        ok = denom > 0
        corr[ok] = np.sum(a[ok] * b[ok], axis=1) / denom[ok]
        vals[k] = 1.0 - corr
        k += 1
    return np.nanmean(vals, axis=0)


def uncertainty_diagnostics(
    table: pd.DataFrame,
    uncertainty_col: str = "ensemble_variance",
    pearson_col: str = "pearson",
    rmse_col: str = "rmse",
    reliability_col: str = "intrinsic_reproducibility",
) -> dict[str, float | int]:
    """Quantify whether a score ranks model failures as more uncertain."""
    cols = [uncertainty_col, pearson_col, rmse_col]
    if reliability_col in table.columns:
        cols.append(reliability_col)
    clean = table[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    clean["correlation_error"] = 1.0 - clean[pearson_col]

    bad = clean[pearson_col] < clean[pearson_col].median()
    auc = roc_auc_score(bad.astype(int), clean[uncertainty_col]) if bad.nunique() == 2 else float("nan")
    q = max(1, clean.shape[0] // 4)

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
            clean.nsmallest(q, uncertainty_col)[pearson_col].median()
        ),
        "median_pearson_highest_uncertainty_quartile": float(
            clean.nlargest(q, uncertainty_col)[pearson_col].median()
        ),
    }
    if reliability_col in clean.columns:
        out["spearman_uncertainty_vs_intrinsic_reliability"] = float(
            clean[uncertainty_col].corr(clean[reliability_col], method="spearman")
        )
    return out


def selective_prediction_curve(
    table: pd.DataFrame,
    uncertainty_col: str,
    pearson_col: str = "pearson",
    coverages: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.2),
) -> pd.DataFrame:
    """Evaluate performance when retaining only the least-uncertain cases."""
    clean = table[[uncertainty_col, pearson_col]].replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean.sort_values(uncertainty_col, ascending=True)
    rows = []
    for coverage in coverages:
        n_keep = max(1, int(np.ceil(coverage * len(clean))))
        kept = clean.iloc[:n_keep]
        rows.append({
            "coverage": float(coverage),
            "n_kept": int(n_keep),
            "median_pearson": float(kept[pearson_col].median()),
            "mean_pearson": float(kept[pearson_col].mean()),
        })
    return pd.DataFrame(rows)
