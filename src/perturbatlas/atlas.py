"""Build perturbation-level transcriptomic response atlases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


@dataclass(frozen=True)
class AtlasSummary:
    n_perturbations: int
    n_genes: int
    n_controls: int
    matrix_type: str


def _group_means(x, labels: pd.Series):
    """Compute one mean-expression vector per label using sparse algebra."""
    categories = pd.Index(pd.unique(labels.astype(str)))
    cat = pd.Categorical(labels.astype(str), categories=categories)
    codes = cat.codes
    if np.any(codes < 0):
        raise ValueError("Unexpected missing perturbation labels.")

    n_cells = len(labels)
    n_groups = len(categories)
    indicator = sparse.csr_matrix(
        (np.ones(n_cells, dtype=np.float32),
         (np.arange(n_cells, dtype=np.int64), codes)),
        shape=(n_cells, n_groups),
    )
    counts = np.asarray(indicator.sum(axis=0)).ravel().astype(np.float64)

    if sparse.issparse(x):
        sums = indicator.T @ x
        means = sums.multiply((1.0 / counts)[:, None]).toarray()
        matrix_type = type(x).__name__
    else:
        arr = np.asarray(x)
        sums = indicator.T @ arr
        means = np.asarray(sums) / counts[:, None]
        matrix_type = type(x).__name__

    return means.astype(np.float32, copy=False), categories, counts, matrix_type


def build_response_atlas(
    path: str | Path,
    perturbation_col: str = "perturbation",
    control_label: str = "control",
):
    """Aggregate cells to perturbation means and subtract the control mean."""
    adata = sc.read_h5ad(path)

    if perturbation_col not in adata.obs:
        raise KeyError(f"Missing obs column {perturbation_col!r}")
    if control_label not in set(adata.obs[perturbation_col].astype(str)):
        raise ValueError(f"Control label {control_label!r} not found.")

    means, categories, counts, matrix_type = _group_means(
        adata.X, adata.obs[perturbation_col]
    )

    control_idx = categories.get_loc(control_label)
    control_mean = means[control_idx].astype(np.float32, copy=False)
    keep = np.arange(len(categories)) != control_idx
    perturbations = categories[keep]
    perturbation_means = means[keep]
    deltas = perturbation_means - control_mean[None, :]

    obs = pd.DataFrame(
        {
            "n_cells": counts[keep].astype(np.int64),
            "l2_effect": np.linalg.norm(deltas, axis=1).astype(np.float32),
            "mean_abs_effect": np.mean(np.abs(deltas), axis=1).astype(np.float32),
        },
        index=pd.Index(perturbations, name="perturbation"),
    )
    var = pd.DataFrame(
        {"control_mean": control_mean},
        index=adata.var_names.copy(),
    )

    atlas = ad.AnnData(X=deltas, obs=obs, var=var)
    atlas.layers["mean_expression"] = perturbation_means.astype(np.float32, copy=False)

    summary = AtlasSummary(
        n_perturbations=int(atlas.n_obs),
        n_genes=int(atlas.n_vars),
        n_controls=int(counts[control_idx]),
        matrix_type=matrix_type,
    )
    return atlas, summary
