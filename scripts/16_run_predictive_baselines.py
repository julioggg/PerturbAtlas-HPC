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
    test = (a.obs["split"].astype(str) == "test").to_numpy()

    rows, per = [], []
    s, r, e = summarize("identity_half_A", x_a[test], x_b[test])
    rows.append(s)
    per.append(pd.DataFrame({"perturbation": a.obs_names[test], "model": s["model"], "pearson": r, "rmse": e}))

    zero = np.zeros_like(x_b[test])
    s, r, e = summarize("zero_response", zero, x_b[test])
    rows.append(s)
    per.append(pd.DataFrame({"perturbation": a.obs_names[test], "model": s["model"], "pearson": r, "rmse": e}))

    for k in [8, 16, 32, 64, 128]:
        pred = pca_denoise_baseline(x_a[train], x_a[test], k)
        s, r, e = summarize(f"pca_{k}", pred, x_b[test])
        rows.append(s)
        per.append(pd.DataFrame({"perturbation": a.obs_names[test], "model": s["model"], "pearson": r, "rmse": e}))

    summary = pd.DataFrame(rows).sort_values("median_pearson", ascending=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.concat(per, ignore_index=True).to_csv(args.output_dir / "per_perturbation.csv", index=False)
    save_metadata(run_metadata({"half_a": str(args.half_a), "half_b": str(args.half_b)}), args.output_dir / "run_metadata.json")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
