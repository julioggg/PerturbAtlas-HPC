#!/usr/bin/env python3
"""Build a unified adaptive-regime atlas across datasets.

Joins:
- multidataset master predictors
- oracle stability summaries
- per-perturbation regret aggregated across repeated splits

The resulting table is the development set for a leakage-safe adaptive gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate_regret(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    g = d.groupby("perturbation", as_index=False).agg(
        regret_n_appearances=("seed", "nunique"),
        oracle_regret_median=("oracle_regret_pearson", "median"),
        oracle_regret_mean=("oracle_regret_pearson", "mean"),
        oracle_regret_q75=("oracle_regret_pearson", lambda x: x.quantile(.75)),
        oracle_regret_q90=("oracle_regret_pearson", lambda x: x.quantile(.90)),
        fraction_regret_gt_0_01=("oracle_regret_pearson", lambda x: float(np.mean(x > .01))),
        fraction_regret_gt_0_02=("oracle_regret_pearson", lambda x: float(np.mean(x > .02))),
        fraction_regret_gt_0_05=("oracle_regret_pearson", lambda x: float(np.mean(x > .05))),
        fraction_same_alpha_as_global=("same_alpha_as_global", "mean"),
        fraction_same_config_as_global=("same_config_as_global", "mean"),
    )
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--master",
        type=Path,
        default=Path("results/multidataset_master/perturbation_master_table.csv"),
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "arch1",
            "wessels23",
            "replogle_k562",
            "replogle_rpe1",
            "nadig25hepg2",
            "nadig25jurkat",
            "kaden25rpe1",
        ],
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/adaptive_regime_atlas"),
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(args.master)
    all_rows = []
    audit = []

    for ds in args.datasets:
        m = master[master["dataset"] == ds].copy()

        stab_path = Path(f"results/oracle_stability/{ds}/perturbation_oracle_stability.csv")
        reg_path = Path(f"results/oracle_regret/{ds}/per_perturbation_regret.csv")

        if not stab_path.exists() or not reg_path.exists():
            audit.append({"dataset": ds, "status": "missing"})
            continue

        stab = pd.read_csv(stab_path)
        reg = aggregate_regret(reg_path)

        x = (
            m.merge(stab, on="perturbation", how="left", suffixes=("", "_oracle"))
             .merge(reg, on="perturbation", how="left")
        )

        # Development targets. These are labels only, never deployment features.
        x["target_personalize_regret_gt_0_01"] = (
            x["oracle_regret_median"] > .01
        ).astype("Int64")
        x["target_personalize_regret_gt_0_02"] = (
            x["oracle_regret_median"] > .02
        ).astype("Int64")

        all_rows.append(x)
        audit.append({
            "dataset": ds,
            "status": "ok",
            "n_master": len(m),
            "n_with_stability": int(x["oracle_alpha_mode"].notna().sum()),
            "n_with_regret": int(x["oracle_regret_median"].notna().sum()),
            "fraction_target_gt_0_01": float(
                x["target_personalize_regret_gt_0_01"].dropna().mean()
            ),
            "median_regret": float(np.nanmedian(x["oracle_regret_median"])),
            "median_oracle_gain": float(np.nanmedian(x["oracle_gain_median"])),
        })

    if not all_rows:
        raise SystemExit("No datasets assembled.")

    atlas = pd.concat(all_rows, ignore_index=True)
    atlas.to_csv(args.output_dir / "adaptive_regime_atlas.csv", index=False)
    pd.DataFrame(audit).to_csv(args.output_dir / "assembly_audit.csv", index=False)

    usable = atlas["oracle_regret_median"].notna()
    summary = {
        "n_rows": int(len(atlas)),
        "n_datasets": int(atlas["dataset"].nunique()),
        "n_with_regret": int(usable.sum()),
        "fraction_regret_median_gt_0_01": float(
            np.mean(atlas.loc[usable, "oracle_regret_median"] > .01)
        ),
        "fraction_regret_median_gt_0_02": float(
            np.mean(atlas.loc[usable, "oracle_regret_median"] > .02)
        ),
        "median_regret": float(np.nanmedian(atlas["oracle_regret_median"])),
        "note": (
            "Oracle/regret columns are development labels and must never be used "
            "as deployment features in outer-held-out evaluation."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== ASSEMBLY AUDIT ===")
    print(pd.DataFrame(audit).to_string(index=False))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
