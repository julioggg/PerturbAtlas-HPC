"""Held-out perturbation denoising benchmark utilities."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA


def _mean_rows(x, idx: np.ndarray) -> np.ndarray:
    sub = x[idx]
    if sparse.issparse(sub):
        return np.asarray(sub.mean(axis=0)).ravel().astype(np.float32)
    return np.asarray(sub, dtype=np.float32).mean(axis=0)


def make_half_signature_pair(
    raw_path: str | Path,
    perturbation_col: str = "perturbation",
    control_label: str = "control",
    seed: int = 20261013,
) -> tuple[ad.AnnData, ad.AnnData]:
    """Create one deterministic pair of independent half-sample signatures.

    Perturbation cells and controls are both split into disjoint halves. The
    returned AnnData objects have matching perturbations and genes; X stores
    perturbation-minus-control signatures for half A and half B respectively.
    """
    adata = sc.read_h5ad(raw_path)
    labels = adata.obs[perturbation_col].astype(str).to_numpy()
    controls = np.flatnonzero(labels == control_label)
    perturbations = pd.Index(pd.unique(labels[labels != control_label]))
    rng = np.random.default_rng(seed)

    c_perm = rng.permutation(controls)
    c_cut = controls.size // 2
    c_a, c_b = c_perm[:c_cut], c_perm[c_cut:]
    c_mean_a = _mean_rows(adata.X, c_a)
    c_mean_b = _mean_rows(adata.X, c_b)

    sig_a, sig_b, names, sizes = [], [], [], []
    for perturbation in perturbations:
        p_idx = np.flatnonzero(labels == perturbation)
        if p_idx.size < 4:
            continue
        p_perm = rng.permutation(p_idx)
        cut = p_idx.size // 2
        p_a, p_b = p_perm[:cut], p_perm[cut:]
        sig_a.append(_mean_rows(adata.X, p_a) - c_mean_a)
        sig_b.append(_mean_rows(adata.X, p_b) - c_mean_b)
        names.append(perturbation)
        sizes.append(int(p_idx.size))

    obs = pd.DataFrame({"n_cells": sizes}, index=pd.Index(names, name="perturbation"))
    var = pd.DataFrame(index=adata.var_names.copy())
    a = ad.AnnData(X=np.vstack(sig_a).astype(np.float32), obs=obs.copy(), var=var.copy())
    b = ad.AnnData(X=np.vstack(sig_b).astype(np.float32), obs=obs.copy(), var=var.copy())
    return a, b


def split_perturbations(names, seed: int = 20261013, train_fraction: float = 0.7, val_fraction: float = 0.15):
    """Deterministically split perturbation identities into train/val/test."""
    names = np.asarray(names, dtype=str)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    n_train = int(round(train_fraction * len(names)))
    n_val = int(round(val_fraction * len(names)))
    train = order[:n_train]
    val = order[n_train:n_train+n_val]
    test = order[n_train+n_val:]
    return train, val, test


def row_pearson(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pearson correlation across genes for each perturbation row."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pred = pred - pred.mean(axis=1, keepdims=True)
    target = target - target.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1)
    num = np.sum(pred * target, axis=1)
    out = np.full(pred.shape[0], np.nan, dtype=float)
    mask = denom > 0
    out[mask] = num[mask] / denom[mask]
    return out


def pca_denoise_baseline(x_train: np.ndarray, x_apply: np.ndarray, n_components: int) -> np.ndarray:
    """Fit PCA on training perturbations and reconstruct held-out signatures."""
    n_components = min(n_components, x_train.shape[0] - 1, x_train.shape[1])
    model = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    model.fit(x_train)
    return model.inverse_transform(model.transform(x_apply)).astype(np.float32)
