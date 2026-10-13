"""Sample-size-aware calibration of perturbation effect magnitudes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


@dataclass(frozen=True)
class EffectCalibrationSummary:
    n_perturbations: int
    n_controls: int
    spearman_ncells_l2: float
    spearman_ncells_mean_abs: float


def _rowwise_control_variance(x) -> np.ndarray:
    """Return per-gene sample variance for a control-cell matrix."""
    if sparse.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel()
        mean_sq = np.asarray(x.power(2).mean(axis=0)).ravel()
        n = x.shape[0]
        pop_var = np.maximum(mean_sq - mean**2, 0.0)
        if n > 1:
            return pop_var * n / (n - 1)
        return pop_var

    arr = np.asarray(x)
    return arr.var(axis=0, ddof=1)


def calibrate_effects(
    raw_path: str | Path,
    atlas_path: str | Path,
    perturbation_col: str = "perturbation",
    control_label: str = "control",
) -> tuple[pd.DataFrame, EffectCalibrationSummary]:
    """Calibrate naive L2 effect sizes for unequal perturbation sample sizes.

    Under a null where perturbation cells are distributed like controls, the
    expected squared Euclidean distance between a perturbation mean based on n
    cells and the control mean based on N_c cells is approximately:

        sum_j Var_control(gene_j) * (1/n + 1/N_c)

    The returned table includes this sample-size-dependent noise floor and a
    noise-adjusted L2 magnitude.
    """
    raw = sc.read_h5ad(raw_path)
    atlas = ad.read_h5ad(atlas_path)

    labels = raw.obs[perturbation_col].astype(str)
    control_mask = labels == control_label
    n_controls = int(control_mask.sum())
    if n_controls < 2:
        raise ValueError("At least two control cells are required.")

    control_x = raw.X[control_mask.to_numpy()]
    control_var = _rowwise_control_variance(control_x)
    total_control_variance = float(np.sum(control_var, dtype=np.float64))

    out = atlas.obs.copy()
    n = out["n_cells"].to_numpy(dtype=np.float64)
    naive_l2 = out["l2_effect"].to_numpy(dtype=np.float64)

    expected_null_l2_sq = total_control_variance * (1.0 / n + 1.0 / n_controls)
    expected_null_l2 = np.sqrt(expected_null_l2_sq)
    excess_l2_sq = np.maximum(naive_l2**2 - expected_null_l2_sq, 0.0)
    adjusted_l2 = np.sqrt(excess_l2_sq)

    out["expected_null_l2"] = expected_null_l2.astype(np.float32)
    out["noise_adjusted_l2"] = adjusted_l2.astype(np.float32)
    out["l2_to_null_ratio"] = (
        naive_l2 / np.maximum(expected_null_l2, np.finfo(np.float64).eps)
    ).astype(np.float32)

    spearman_ncells_l2 = float(
        pd.Series(out["n_cells"]).corr(pd.Series(out["l2_effect"]), method="spearman")
    )
    spearman_ncells_mean_abs = float(
        pd.Series(out["n_cells"]).corr(
            pd.Series(out["mean_abs_effect"]), method="spearman"
        )
    )

    summary = EffectCalibrationSummary(
        n_perturbations=int(out.shape[0]),
        n_controls=n_controls,
        spearman_ncells_l2=spearman_ncells_l2,
        spearman_ncells_mean_abs=spearman_ncells_mean_abs,
    )
    return out, summary
