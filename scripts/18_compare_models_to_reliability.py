#!/usr/bin/env python3
"""Relate held-out model performance to intrinsic perturbation reliability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from perturbatlas.reproducibility import run_metadata, save_metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("baselines", type=Path)
    p.add_argument("pytorch", type=Path)
    p.add_argument("reliability", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/model_reliability/replogle_k562"))
    p.add_argument(
        "--pca-model",
        default=None,
        help="Locked PCA model chosen on validation data (e.g. pca_64). "
             "If omitted, legacy behavior selects the best PCA on test data.",
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.baselines)
    pt = pd.read_csv(args.pytorch)
    rel = pd.read_csv(args.reliability, index_col=0)

    wide = base.pivot(index="perturbation", columns="model", values="pearson")
    pt = pt.set_index("perturbation").rename(columns={"pearson": "pytorch"})[["pytorch"]]
    df = rel.join(wide, how="inner").join(pt, how="inner")

    if args.pca_model is not None:
        if args.pca_model not in wide.columns:
            raise ValueError(
                f"Requested --pca-model {args.pca_model!r} not found. "
                f"Available PCA models: {[c for c in wide.columns if str(c).startswith('pca_')]}"
            )
        best_pca = args.pca_model
        pca_selection_source = "validation_locked"
    else:
        best_pca = None
        best_median = -np.inf
        for col in [c for c in wide.columns if str(c).startswith("pca_")]:
            med = float(df[col].median(skipna=True))
            if med > best_median:
                best_median = med
                best_pca = col
        pca_selection_source = "legacy_test_selected"

    df["pca_gain_over_identity"] = df[best_pca] - df["identity_half_A"]
    df["pytorch_gain_over_identity"] = df["pytorch"] - df["identity_half_A"]
    df["pytorch_minus_best_pca"] = df["pytorch"] - df[best_pca]

    reliable = df["pearson_median"] > 0.5
    high_reliable = df["pearson_median"] > 0.8

    summary = {
        "n_test_matched": int(df.shape[0]),
        "best_pca": str(best_pca),
        "pca_selection_source": pca_selection_source,
        "median_identity_pearson": float(df["identity_half_A"].median()),
        "median_best_pca_pearson": float(df[best_pca].median()),
        "median_pytorch_pearson": float(df["pytorch"].median()),
        "spearman_best_pca_vs_intrinsic_repro": float(df[best_pca].corr(df["pearson_median"], method="spearman")),
        "spearman_pytorch_vs_intrinsic_repro": float(df["pytorch"].corr(df["pearson_median"], method="spearman")),
        "median_pca_gain_over_identity": float(df["pca_gain_over_identity"].median()),
        "median_pytorch_gain_over_identity": float(df["pytorch_gain_over_identity"].median()),
        "median_pytorch_minus_best_pca": float(df["pytorch_minus_best_pca"].median()),
        "n_intrinsic_r_gt_0_5": int(reliable.sum()),
        "median_best_pca_if_intrinsic_r_gt_0_5": float(df.loc[reliable, best_pca].median()) if reliable.any() else float("nan"),
        "median_pytorch_if_intrinsic_r_gt_0_5": float(df.loc[reliable, "pytorch"].median()) if reliable.any() else float("nan"),
        "n_intrinsic_r_gt_0_8": int(high_reliable.sum()),
        "median_best_pca_if_intrinsic_r_gt_0_8": float(df.loc[high_reliable, best_pca].median()) if high_reliable.any() else float("nan"),
        "median_pytorch_if_intrinsic_r_gt_0_8": float(df.loc[high_reliable, "pytorch"].median()) if high_reliable.any() else float("nan"),
    }

    df.to_csv(args.output_dir / "model_reliability_table.csv")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_metadata(run_metadata(vars(args)), args.output_dir / "run_metadata.json")

    print("\n=== MODEL PERFORMANCE VS INTRINSIC RELIABILITY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
