#!/usr/bin/env python3
"""Summarize the strictly split-safe Gate v3 by transfer regime.

This analysis is descriptive only. It does not refit, tune, or select models.
Its purpose is to separate:
- in-regime CRISPRi essentiality transfer,
- CRISPRi developmental/context shift,
- Cas13 combinatorial modality shift,
- CRISPRa reprogramming modality shift.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

GROUPS = {
    "replogle_k562": "core_crispri_essentiality",
    "replogle_rpe1": "core_crispri_essentiality",
    "nadig25hepg2": "core_crispri_essentiality",
    "nadig25jurkat": "core_crispri_essentiality",
    "arch1": "context_shift_crispri_hesc",
    "wessels23": "modality_shift_cas13_combinatorial",
    "kaden25rpe1": "modality_shift_crispra_reprogramming",
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "outer_metrics", type=Path, nargs="?",
        default=Path("results/split_safe_gate_v3/outer_metrics.csv")
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=Path("results/split_safe_gate_v3_regimes")
    )
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df=pd.read_csv(args.outer_metrics)
    df["transfer_group"]=df["held_out_dataset"].map(GROUPS)
    if df["transfer_group"].isna().any():
        missing=sorted(df.loc[df["transfer_group"].isna(),"held_out_dataset"].unique())
        raise SystemExit(f"Unmapped datasets: {missing}")

    df.to_csv(args.output_dir/"outer_metrics_with_regimes.csv", index=False)

    core=df[df["transfer_group"]=="core_crispri_essentiality"].copy()
    core_summary=pd.DataFrame([{
        "n_core_datasets":core["held_out_dataset"].nunique(),
        "median_core_auroc":core["auroc"].median(),
        "min_core_auroc":core["auroc"].min(),
        "max_core_auroc":core["auroc"].max(),
        "median_core_auprc":core["auprc"].median(),
        "median_core_balanced_accuracy":core["balanced_accuracy"].median(),
        "median_core_regret_capture":core["regret_capture_fraction"].median(),
        "median_core_selected_fraction":core["selected_fraction"].median(),
    }])
    core_summary.to_csv(args.output_dir/"core_summary.csv", index=False)

    by_group=(
        df.groupby("transfer_group", as_index=False)
        .agg(
            n_datasets=("held_out_dataset","nunique"),
            median_auroc=("auroc","median"),
            min_auroc=("auroc","min"),
            median_auprc=("auprc","median"),
            median_balanced_accuracy=("balanced_accuracy","median"),
            median_regret_capture=("regret_capture_fraction","median"),
            median_selected_fraction=("selected_fraction","median"),
        )
    )
    by_group.to_csv(args.output_dir/"by_transfer_group.csv", index=False)

    print("\n=== SPLIT-SAFE CORE CRISPRi SUMMARY ===")
    print(core_summary.to_string(index=False))
    print("\n=== SPLIT-SAFE TRANSFER REGIMES ===")
    print(by_group.to_string(index=False))

if __name__=="__main__":
    main()
