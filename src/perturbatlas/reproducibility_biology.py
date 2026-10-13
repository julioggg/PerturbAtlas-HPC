"""Intrinsic reproducibility of perturbation response signatures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def _mean_rows(x, idx: np.ndarray) -> np.ndarray:
    sub = x[idx]
    if sparse.issparse(sub):
        return np.asarray(sub.mean(axis=0)).ravel().astype(np.float32)
    return np.asarray(sub, dtype=np.float32).mean(axis=0)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def split_half_reproducibility(
    raw_path: str | Path,
    perturbation_col: str = "perturbation",
    control_label: str = "control",
    repeats: int = 10,
    seed: int = 20261013,
) -> pd.DataFrame:
    """Estimate perturbation-specific response reproducibility.

    Each repeat splits perturbation cells into two disjoint halves and also
    splits control cells into two disjoint halves. Response signatures are
    computed independently in each half, avoiding a shared control mean that
    would otherwise inflate agreement.
    """
    adata = sc.read_h5ad(raw_path)
    labels = adata.obs[perturbation_col].astype(str).to_numpy()
    control_idx = np.flatnonzero(labels == control_label)
    perturbations = pd.Index(pd.unique(labels[labels != control_label]))
    rng = np.random.default_rng(seed)

    rows = []
    for perturbation in perturbations:
        p_idx = np.flatnonzero(labels == perturbation)
        n = p_idx.size
        if n < 4:
            continue

        for rep in range(repeats):
            p_perm = rng.permutation(p_idx)
            p_cut = n // 2
            p_a, p_b = p_perm[:p_cut], p_perm[p_cut:]

            c_perm = rng.permutation(control_idx)
            c_cut = control_idx.size // 2
            c_a, c_b = c_perm[:c_cut], c_perm[c_cut:]

            delta_a = _mean_rows(adata.X, p_a) - _mean_rows(adata.X, c_a)
            delta_b = _mean_rows(adata.X, p_b) - _mean_rows(adata.X, c_b)

            diff = delta_a.astype(np.float64) - delta_b.astype(np.float64)
            rmse = float(np.sqrt(np.mean(diff**2)))
            signal_scale = float(
                0.5 * (np.sqrt(np.mean(delta_a.astype(np.float64)**2)) +
                       np.sqrt(np.mean(delta_b.astype(np.float64)**2)))
            )
            nrmse = rmse / signal_scale if signal_scale > 0 else float("nan")

            rows.append(
                {
                    "perturbation": perturbation,
                    "repeat": rep,
                    "n_cells": int(n),
                    "pearson": _pearson(delta_a, delta_b),
                    "cosine": _cosine(delta_a, delta_b),
                    "rmse": rmse,
                    "nrmse": nrmse,
                }
            )

    return pd.DataFrame(rows)


def summarize_split_half(replicates: pd.DataFrame) -> pd.DataFrame:
    """Aggregate split-half metrics to one row per perturbation."""
    return (
        replicates.groupby("perturbation", as_index=True)
        .agg(
            n_cells=("n_cells", "first"),
            pearson_median=("pearson", "median"),
            pearson_mean=("pearson", "mean"),
            cosine_median=("cosine", "median"),
            rmse_median=("rmse", "median"),
            nrmse_median=("nrmse", "median"),
        )
        .sort_index()
    )
