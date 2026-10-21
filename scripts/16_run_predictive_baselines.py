#!/usr/bin/env python3
"""Evaluate held-out perturbation denoising baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from perturbatlas.benchmark import pca_denoise_baseline, row_pearson
from perturbatlas.reproducibility import run_metadata, save_metadata


def summarize(name, pred, target):
    r = row_pearson(pred, target)
    rmse = np.sqrt(np.mean((pred - target) ** 2, axis=1))
    return {
        "model": name,
        "median_pearson": float(np.nanmedian(r)),
        "mean_pearson": float(np.nanmean(r)),
        "median_rmse": float(np.median(rmse)),
    }, r, rmse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("half_a", type=Path)
    p.add_argument("half_b", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/models/replogle_k562_baselines"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    a, b = ad.read_h5ad(args.half_a), ad.read_h5ad(args.half_b)
    x_a, x_b = np.asarray(a.X, dtype=np.float32), np.asarray(b.X, dtype=np.float32)
    train = (a.obs["split"].astype(str) == "train").to_numpy()
    val = (a.obs["split"].astype(str) == "val").to_numpy()
    test = (a.obs["split"].astype(str) == "test").to_numpy()

    rows, per = [], []
    pca_validation = []
    s, r, e = summarize("identity_half_A", x_a[test], x_b[test])
    rows.append(s)
    per.append(pd.DataFrame({"perturbation": a.obs_names[test], "model": s["model"], "pearson": r, "rmse": e}))

    zero = np.zeros_like(x_b[test])
    s, r, e = summarize("zero_response", zero, x_b[test])
    rows.append(s)
    per.append(pd.DataFrame({"perturbation": a.obs_names[test], "model": s["model"], "pearson": r, "rmse": e}))

    for k in [8, 16, 32, 64, 128]:
        pred_val = pca_denoise_baseline(x_a[train], x_a[val], k)
        val_s, _, _ = summarize(f"pca_{k}", pred_val, x_b[val])
        pca_validation.append({
            "model": f"pca_{k}",
            "validation_median_pearson": val_s["median_pearson"],
            "validation_mean_pearson": val_s["mean_pearson"],
            "validation_median_rmse": val_s["median_rmse"],
        })

        pred = pca_denoise_baseline(x_a[train], x_a[test], k)
        s, r, e = summarize(f"pca_{k}", pred, x_b[test])
        rows.append(s)
        per.append(pd.DataFrame({"perturbation": a.obs_names[test], "model": s["model"], "pearson": r, "rmse": e}))

    validation = pd.DataFrame(pca_validation).sort_values(
        "validation_median_pearson", ascending=False
    )
    selected_pca = str(validation.iloc[0]["model"])

    summary = pd.DataFrame(rows).sort_values("median_pearson", ascending=False)
    summary["selected_by_validation"] = summary["model"].eq(selected_pca)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    validation.to_csv(args.output_dir / "pca_validation_selection.csv", index=False)
    pd.concat(per, ignore_index=True).to_csv(args.output_dir / "per_perturbation.csv", index=False)

    selection = {
        "selected_pca": selected_pca,
        "selection_metric": "validation_median_pearson",
        "validation_median_pearson": float(
            validation.iloc[0]["validation_median_pearson"]
        ),
        "test_median_pearson": float(
            summary.loc[summary["model"].eq(selected_pca), "median_pearson"].iloc[0]
        ),
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    save_metadata(run_metadata({"half_a": str(args.half_a), "half_b": str(args.half_b)}), args.output_dir / "run_metadata.json")
    print(summary.to_string(index=False))
    print("\n=== PCA SELECTED ON VALIDATION ===")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
