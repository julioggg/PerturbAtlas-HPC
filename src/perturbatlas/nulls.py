"""Empirical null models for perturbation effect magnitudes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def _row_sum(x):
    if sparse.issparse(x):
        return np.asarray(x.sum(axis=0)).ravel().astype(np.float64)
    return np.asarray(x, dtype=np.float64).sum(axis=0)


def empirical_control_null(
    raw_path: str | Path,
    perturbation_sizes: np.ndarray,
    perturbation_col: str = "perturbation",
    control_label: str = "control",
    n_replicates: int = 100,
    max_sizes: int = 40,
    seed: int = 20261013,
) -> pd.DataFrame:
    """Estimate a matched-size null using disjoint control-vs-control splits.

    For each selected pseudo-perturbation size n, repeatedly sample n control
    cells without replacement. Their mean is compared with the mean of all
    remaining control cells. This produces an empirical null distribution for
    the Euclidean distance expected solely from finite sampling.
    """
    adata = sc.read_h5ad(raw_path)
    labels = adata.obs[perturbation_col].astype(str)
    control_mask = (labels == control_label).to_numpy()
    control_x = adata.X[control_mask]
    n_controls = int(control_x.shape[0])

    sizes = np.asarray(perturbation_sizes, dtype=int)
    sizes = np.unique(sizes[(sizes > 1) & (sizes < n_controls - 1)])
    if sizes.size == 0:
        raise ValueError("No valid perturbation sizes supplied.")

    if sizes.size > max_sizes:
        q = np.linspace(0.0, 1.0, max_sizes)
        pos = np.unique(np.round(q * (sizes.size - 1)).astype(int))
        sizes = sizes[pos]

    total_sum = _row_sum(control_x)
    rng = np.random.default_rng(seed)
    rows = []

    for n in sizes:
        l2_values = np.empty(n_replicates, dtype=np.float64)
        for b in range(n_replicates):
            idx = rng.choice(n_controls, size=int(n), replace=False)
            sample_sum = _row_sum(control_x[idx])
            sample_mean = sample_sum / n
            remainder_mean = (total_sum - sample_sum) / (n_controls - n)
            delta = sample_mean - remainder_mean
            l2_values[b] = np.linalg.norm(delta)

        rows.append(
            {
                "n_cells": int(n),
                "n_replicates": int(n_replicates),
                "null_mean_l2": float(np.mean(l2_values)),
                "null_median_l2": float(np.median(l2_values)),
                "null_sd_l2": float(np.std(l2_values, ddof=1)),
                "null_q90_l2": float(np.quantile(l2_values, 0.90)),
                "null_q95_l2": float(np.quantile(l2_values, 0.95)),
                "null_q99_l2": float(np.quantile(l2_values, 0.99)),
            }
        )

    return pd.DataFrame(rows).sort_values("n_cells").reset_index(drop=True)


def interpolate_empirical_thresholds(
    effect_table: pd.DataFrame,
    null_table: pd.DataFrame,
) -> pd.DataFrame:
    """Interpolate empirical null quantiles onto every perturbation size."""
    out = effect_table.copy()
    x = null_table["n_cells"].to_numpy(dtype=float)
    for col in ["null_median_l2", "null_q90_l2", "null_q95_l2", "null_q99_l2"]:
        y = null_table[col].to_numpy(dtype=float)
        out[col] = np.interp(out["n_cells"].to_numpy(dtype=float), x, y)

    out["above_empirical_q95"] = out["l2_effect"] > out["null_q95_l2"]
    out["above_empirical_q99"] = out["l2_effect"] > out["null_q99_l2"]
    out["empirical_l2_over_q95"] = out["l2_effect"] / out["null_q95_l2"]
    return out
