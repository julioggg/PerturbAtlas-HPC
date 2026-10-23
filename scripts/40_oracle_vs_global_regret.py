#!/usr/bin/env python3
"""Quantify the value of perturbation-specific adaptation over global shrinkage.

For each repeated split:
- read the validation-locked global (k, alpha)
- read the full correction landscape
- compare each test perturbation's oracle Pearson to the Pearson obtained by
  the globally selected configuration
- compute oracle regret = oracle - global

This is the direct ceiling on how much a perfect adaptive selector could recover
over the already-strong validation-locked global shrinkage policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "landscape_root",
        type=Path,
        help="e.g. results/correction_landscapes/replogle_k562",
    )
    p.add_argument(
        "repeated_root",
        type=Path,
        help="e.g. results/repeated_shrinkage/replogle_k562",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seed_summaries = []

    seed_dirs = sorted(args.landscape_root.glob("seed_*"))
    if not seed_dirs:
        raise SystemExit("No landscape seed_* directories found.")

    for sdir in seed_dirs:
        seed = int(sdir.name.split("_", 1)[1])
        landscape_path = sdir / "correction_landscape.csv"
        oracle_path = sdir / "oracle_targets.csv"
        global_summary_path = (
            args.repeated_root / sdir.name / "shrinkage" / "summary.json"
        )

        if not (landscape_path.exists() and oracle_path.exists() and global_summary_path.exists()):
            print(f"SKIP seed={seed}: missing required file")
            continue

        landscape = pd.read_csv(landscape_path)
        oracle = pd.read_csv(oracle_path)
        gs = json.loads(global_summary_path.read_text())

        k = int(gs["selected_pca_components"])
        alpha = float(gs["selected_alpha"])

        if alpha == 0:
            sel = landscape[
                (landscape["pca_components"] == 0)
                & np.isclose(landscape["alpha"], 0.0)
            ].copy()
        else:
            sel = landscape[
                (landscape["pca_components"] == k)
                & np.isclose(landscape["alpha"], alpha)
            ].copy()

        if sel.empty:
            raise RuntimeError(
                f"seed={seed}: global config k={k}, alpha={alpha} not found"
            )

        g = sel[
            ["perturbation", "pearson", "rmse", "gain_over_identity"]
        ].rename(columns={
            "pearson": "global_pearson",
            "rmse": "global_rmse",
            "gain_over_identity": "global_gain_over_identity",
        })

        o = oracle[
            [
                "perturbation",
                "oracle_pca_components",
                "oracle_alpha",
                "oracle_pearson",
                "oracle_rmse",
                "oracle_gain_over_identity",
                "identity_pearson",
            ]
        ]

        m = o.merge(g, on="perturbation", how="inner")
        m.insert(0, "seed", seed)
        m["global_pca_components"] = k
        m["global_alpha"] = alpha
        m["oracle_regret_pearson"] = m["oracle_pearson"] - m["global_pearson"]
        m["oracle_gain_increment_over_global"] = (
            m["oracle_gain_over_identity"] - m["global_gain_over_identity"]
        )
        m["same_alpha_as_global"] = np.isclose(m["oracle_alpha"], alpha)
        m["same_config_as_global"] = (
            np.isclose(m["oracle_alpha"], alpha)
            & (m["oracle_pca_components"] == k)
        )
        rows.append(m)

        r = m["oracle_regret_pearson"].to_numpy(float)
        seed_summaries.append({
            "seed": seed,
            "global_k": k,
            "global_alpha": alpha,
            "n_test": len(m),
            "median_global_pearson": float(np.nanmedian(m["global_pearson"])),
            "median_oracle_pearson": float(np.nanmedian(m["oracle_pearson"])),
            "median_oracle_regret": float(np.nanmedian(r)),
            "mean_oracle_regret": float(np.nanmean(r)),
            "q75_oracle_regret": float(np.nanquantile(r, .75)),
            "q90_oracle_regret": float(np.nanquantile(r, .90)),
            "fraction_regret_gt_0_01": float(np.mean(r > .01)),
            "fraction_regret_gt_0_02": float(np.mean(r > .02)),
            "fraction_regret_gt_0_05": float(np.mean(r > .05)),
            "fraction_same_alpha_as_global": float(m["same_alpha_as_global"].mean()),
            "fraction_same_config_as_global": float(m["same_config_as_global"].mean()),
        })

    if not rows:
        raise SystemExit("No seeds could be analyzed.")

    allx = pd.concat(rows, ignore_index=True)
    per_seed = pd.DataFrame(seed_summaries)

    allx.to_csv(args.output_dir / "per_perturbation_regret.csv", index=False)
    per_seed.to_csv(args.output_dir / "per_seed_summary.csv", index=False)

    r = allx["oracle_regret_pearson"].to_numpy(float)
    overall = {
        "n_rows": int(len(allx)),
        "n_seeds": int(allx["seed"].nunique()),
        "median_oracle_regret": float(np.nanmedian(r)),
        "mean_oracle_regret": float(np.nanmean(r)),
        "q75_oracle_regret": float(np.nanquantile(r, .75)),
        "q90_oracle_regret": float(np.nanquantile(r, .90)),
        "q95_oracle_regret": float(np.nanquantile(r, .95)),
        "fraction_regret_gt_0_01": float(np.mean(r > .01)),
        "fraction_regret_gt_0_02": float(np.mean(r > .02)),
        "fraction_regret_gt_0_05": float(np.mean(r > .05)),
        "fraction_oracle_same_alpha_as_global": float(
            allx["same_alpha_as_global"].mean()
        ),
        "fraction_oracle_same_config_as_global": float(
            allx["same_config_as_global"].mean()
        ),
        "interpretation": (
            "oracle_regret_pearson is the per-perturbation performance gap between "
            "a perfect perturbation-specific selector and the validation-locked "
            "global shrinkage configuration. It is an upper bound on the value "
            "available to an adaptive policy under this expert grid."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(overall, indent=2))

    print("\n=== ORACLE VS GLOBAL REGRET ===")
    print(json.dumps(overall, indent=2))
    print("\n=== PER-SEED ===")
    print(per_seed.to_string(index=False))


if __name__ == "__main__":
    main()
