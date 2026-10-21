#!/usr/bin/env python3
"""Validation-locked shrinkage between identity and PCA denoising.

Selects both PCA dimensionality and shrinkage alpha using validation data only,
then evaluates the locked configuration once on test data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from perturbatlas.benchmark import pca_denoise_baseline, row_pearson
from perturbatlas.reproducibility import run_metadata, save_metadata


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    r = row_pearson(pred, target)
    rmse = np.sqrt(np.mean((pred - target) ** 2, axis=1))
    return {
        "median_pearson": float(np.nanmedian(r)),
        "mean_pearson": float(np.nanmean(r)),
        "median_rmse": float(np.median(rmse)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("half_a", type=Path)
    p.add_argument("half_b", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pca-components", nargs="+", type=int, default=[8,16,32,64,128])
    p.add_argument(
        "--alphas", nargs="+", type=float,
        default=[0.0,0.05,0.1,0.2,0.3,0.5,0.75,1.0],
        help="Blend weight: prediction=(1-alpha)*identity + alpha*PCA.",
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    a, b = ad.read_h5ad(args.half_a), ad.read_h5ad(args.half_b)
    xa, xb = np.asarray(a.X, dtype=np.float32), np.asarray(b.X, dtype=np.float32)
    split = a.obs["split"].astype(str).to_numpy()
    tr, va, te = split == "train", split == "val", split == "test"

    val_rows = []
    val_preds = {}
    test_preds = {}

    for k in args.pca_components:
        pv = pca_denoise_baseline(xa[tr], xa[va], k)
        pt = pca_denoise_baseline(xa[tr], xa[te], k)
        val_preds[k] = pv
        test_preds[k] = pt
        for alpha in args.alphas:
            pred = (1.0 - alpha) * xa[va] + alpha * pv
            m = metrics(pred, xb[va])
            val_rows.append({
                "pca_components": int(k),
                "alpha": float(alpha),
                **m,
            })

    selection = pd.DataFrame(val_rows).sort_values(
        ["median_pearson", "median_rmse"],
        ascending=[False, True],
    ).reset_index(drop=True)
    best = selection.iloc[0]
    k = int(best["pca_components"])
    alpha = float(best["alpha"])

    pred_test = (1.0 - alpha) * xa[te] + alpha * test_preds[k]
    test_m = metrics(pred_test, xb[te])
    identity_m = metrics(xa[te], xb[te])
    r_test = row_pearson(pred_test, xb[te])
    r_identity = row_pearson(xa[te], xb[te])

    summary = {
        "model": "validation_locked_identity_pca_shrinkage",
        "selected_pca_components": k,
        "selected_alpha": alpha,
        "selection_metric": "validation_median_pearson",
        "validation_median_pearson": float(best["median_pearson"]),
        "validation_median_rmse": float(best["median_rmse"]),
        "test_median_pearson": test_m["median_pearson"],
        "test_mean_pearson": test_m["mean_pearson"],
        "test_median_rmse": test_m["median_rmse"],
        "identity_test_median_pearson": identity_m["median_pearson"],
        "median_gain_over_identity": float(np.nanmedian(r_test - r_identity)),
        "n_test": int(te.sum()),
    }

    selection.to_csv(args.output_dir / "validation_grid.csv", index=False)
    pd.DataFrame({
        "perturbation": a.obs_names[te],
        "pearson": r_test,
        "identity_pearson": r_identity,
        "gain_over_identity": r_test - r_identity,
    }).to_csv(args.output_dir / "per_perturbation.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    save_metadata(run_metadata(vars(args)), args.output_dir / "run_metadata.json")

    print("\n=== VALIDATION-LOCKED SHRINKAGE ===")
    print(selection.head(15).to_string(index=False))
    print("\n=== LOCKED TEST RESULT ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
