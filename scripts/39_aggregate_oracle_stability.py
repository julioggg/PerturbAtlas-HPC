#!/usr/bin/env python3
"""Aggregate perturbation-specific oracle correction targets across repeated splits.

Outputs per perturbation:
- number of test appearances
- modal/median oracle alpha
- modal oracle PCA dimensionality
- median oracle gain
- fraction of appearances improved over identity
- alpha stability / entropy
- PCA stability / entropy

These summaries characterize target stability before any adaptive policy is trained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import math

import numpy as np
import pandas as pd


def entropy_norm(values: pd.Series) -> float:
    x = values.dropna()
    if len(x) == 0:
        return np.nan
    p = x.value_counts(normalize=True).to_numpy(float)
    if len(p) <= 1:
        return 0.0
    h = -np.sum(p * np.log(p))
    return float(h / np.log(len(p)))


def mode_smallest(values: pd.Series):
    x = values.dropna()
    if len(x) == 0:
        return np.nan
    vc = x.value_counts()
    top = vc[vc == vc.max()].index
    try:
        return float(np.min(np.asarray(top, dtype=float)))
    except Exception:
        return sorted(map(str, top))[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_root", type=Path,
                   help="e.g. results/correction_landscapes/wessels23")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.dataset_root.glob("seed_*/oracle_targets.csv"))
    if not files:
        raise SystemExit(f"No seed_*/oracle_targets.csv files found in {args.dataset_root}")

    rows = []
    for f in files:
        d = pd.read_csv(f)
        if "seed" not in d:
            seed = int(f.parent.name.split("_", 1)[1])
            d["seed"] = seed
        rows.append(d)

    allx = pd.concat(rows, ignore_index=True)
    allx["perturbation"] = allx["perturbation"].astype(str)

    records = []
    for pert, g in allx.groupby("perturbation", sort=True):
        alpha = g["oracle_alpha"]
        kpca = g["oracle_pca_components"]
        alpha_mode = mode_smallest(alpha)
        pca_mode = mode_smallest(kpca)
        alpha_mode_frac = float(np.mean(alpha == alpha_mode)) if np.isfinite(alpha_mode) else np.nan
        pca_mode_frac = float(np.mean(kpca == pca_mode)) if np.isfinite(pca_mode) else np.nan

        records.append({
            "perturbation": pert,
            "n_test_appearances": int(g["seed"].nunique()),
            "oracle_alpha_median": float(np.nanmedian(alpha)),
            "oracle_alpha_mode": alpha_mode,
            "oracle_alpha_mode_fraction": alpha_mode_frac,
            "oracle_alpha_entropy_norm": entropy_norm(alpha),
            "oracle_pca_median": float(np.nanmedian(kpca)),
            "oracle_pca_mode": pca_mode,
            "oracle_pca_mode_fraction": pca_mode_frac,
            "oracle_pca_entropy_norm": entropy_norm(kpca),
            "oracle_gain_median": float(np.nanmedian(g["oracle_gain_over_identity"])),
            "oracle_gain_mean": float(np.nanmean(g["oracle_gain_over_identity"])),
            "oracle_gain_q25": float(np.nanquantile(g["oracle_gain_over_identity"], .25)),
            "oracle_gain_q75": float(np.nanquantile(g["oracle_gain_over_identity"], .75)),
            "fraction_oracle_gain_positive": float(np.mean(g["oracle_gain_over_identity"] > 0)),
            "fraction_oracle_identity": float(np.mean(g["oracle_alpha"] == 0)),
            "identity_pearson_median": float(np.nanmedian(g["identity_pearson"])),
            "oracle_pearson_median": float(np.nanmedian(g["oracle_pearson"])),
        })

    agg = pd.DataFrame(records)
    agg.to_csv(args.output_dir / "perturbation_oracle_stability.csv", index=False)
    allx.to_csv(args.output_dir / "all_oracle_targets.csv", index=False)

    summary = {
        "n_seed_files": len(files),
        "n_unique_perturbations": int(agg.shape[0]),
        "median_test_appearances": float(np.median(agg["n_test_appearances"])),
        "median_oracle_gain": float(np.nanmedian(agg["oracle_gain_median"])),
        "median_alpha_mode_fraction": float(np.nanmedian(agg["oracle_alpha_mode_fraction"])),
        "median_alpha_entropy_norm": float(np.nanmedian(agg["oracle_alpha_entropy_norm"])),
        "fraction_perturbations_alpha_mode_fraction_ge_0_5": float(
            np.mean(agg["oracle_alpha_mode_fraction"] >= 0.5)
        ),
        "fraction_perturbations_gain_positive_majority": float(
            np.mean(agg["fraction_oracle_gain_positive"] > 0.5)
        ),
        "note": (
            "A low normalized alpha entropy and high modal fraction indicate a stable "
            "perturbation-specific correction regime across repeated splits."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== ORACLE STABILITY SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n=== ALPHA MODE DISTRIBUTION ===")
    print(agg["oracle_alpha_mode"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
