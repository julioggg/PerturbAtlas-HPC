"""Dataset inspection utilities for perturbational single-cell data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


@dataclass(frozen=True)
class DatasetProfile:
    n_cells: int
    n_genes: int
    n_perturbations: int
    n_controls: int
    min_cells_per_perturbation: int
    median_cells_per_perturbation: float
    max_cells_per_perturbation: int
    target_gene_coverage: int
    target_gene_coverage_fraction: float


def profile_h5ad(
    path: str | Path,
    perturbation_col: str = "perturbation",
    control_label: str = "control",
) -> tuple[DatasetProfile, pd.DataFrame]:
    """Return high-level metadata and perturbation counts for a .h5ad file."""
    adata = sc.read_h5ad(path, backed="r")

    if perturbation_col not in adata.obs.columns:
        raise KeyError(
            f"Missing required obs column {perturbation_col!r}; "
            f"found {list(adata.obs.columns)!r}"
        )

    counts = adata.obs[perturbation_col].value_counts().rename("n_cells").to_frame()
    counts.index.name = perturbation_col

    targets = counts.index.astype(str)
    targets_no_control = targets[targets != control_label]
    genes = set(map(str, adata.var_names))
    covered = sum(target in genes for target in targets_no_control)

    profile = DatasetProfile(
        n_cells=int(adata.n_obs),
        n_genes=int(adata.n_vars),
        n_perturbations=int(counts.shape[0]),
        n_controls=int(counts.loc[control_label, "n_cells"]) if control_label in counts.index else 0,
        min_cells_per_perturbation=int(counts["n_cells"].min()),
        median_cells_per_perturbation=float(counts["n_cells"].median()),
        max_cells_per_perturbation=int(counts["n_cells"].max()),
        target_gene_coverage=int(covered),
        target_gene_coverage_fraction=float(covered / len(targets_no_control))
        if len(targets_no_control)
        else 0.0,
    )
    return profile, counts


def sample_expression_summary(
    path: str | Path,
    sample_size: int = 5000,
    seed: int = 20261013,
) -> dict[str, float | int | str]:
    """Estimate expression-scale properties from a reproducible cell sample."""
    adata = sc.read_h5ad(path, backed="r")
    rng = np.random.default_rng(seed)
    n = min(sample_size, adata.n_obs)
    idx = np.sort(rng.choice(adata.n_obs, size=n, replace=False))
    x = adata[idx, :].X

    if sparse.issparse(x):
        values = x.data
        nnz = int(x.nnz)
        total = int(x.shape[0] * x.shape[1])
        mean = float(x.mean())
        min_value = float(values.min()) if values.size else 0.0
        max_value = float(values.max()) if values.size else 0.0
        dtype = str(x.dtype)
        matrix_type = type(x).__name__
    else:
        arr = np.asarray(x)
        nnz = int(np.count_nonzero(arr))
        total = int(arr.size)
        mean = float(arr.mean())
        min_value = float(arr.min())
        max_value = float(arr.max())
        dtype = str(arr.dtype)
        matrix_type = type(x).__name__

    return {
        "sample_cells": int(n),
        "matrix_type": matrix_type,
        "dtype": dtype,
        "mean_expression": mean,
        "min_expression": min_value,
        "max_expression": max_value,
        "approx_zero_fraction": float(1.0 - nnz / total),
    }
