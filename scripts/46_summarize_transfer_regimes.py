#!/usr/bin/env python3
"""Summarize adaptive-gate performance by biologically defined transfer regime.

This analysis separates:
- core CRISPRi essentiality transfer (Replogle/Nadig),
- CRISPRi developmental context shift (ARCH1),
- Cas13 combinatorial modality shift (Wessels),
- CRISPRa reprogramming modality shift (Kaden).

It does not tune or refit any model; it only summarizes previously frozen
outer-LODO metrics, preventing post-hoc leakage into model selection.
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
        default=Path("results/adaptive_gate_v2/outer_metrics.csv")
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=Path("results/adaptive_gate_v2_regimes")
    )
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(args.outer_metrics)
    df["transfer_group"]=df["held_out_dataset"].map(GROUPS)
    if df["transfer_group"].isna().any():
        missing=sorted(df.loc[df["transfer_group"].isna(),"held_out_dataset"].unique())
        raise SystemExit(f"Unmapped datasets: {missing}")

    detailed=df.copy()
    detailed.to_csv(args.output_dir/"outer_metrics_with_regimes.csv",index=False)

    rows=[]
    for (rep,grp),g in df.groupby(["representation","transfer_group"]):
        rows.append({
            "representation":rep,
            "transfer_group":grp,
            "n_datasets":g["held_out_dataset"].nunique(),
            "median_auroc":g["auroc"].median(),
            "min_auroc":g["auroc"].min(),
            "median_auprc":g["auprc"].median(),
            "median_balanced_accuracy":g["balanced_accuracy"].median(),
            "median_regret_capture":g["regret_capture_fraction"].median(),
            "median_selected_fraction":g["selected_fraction"].median(),
        })
    by_group=pd.DataFrame(rows)
    by_group.to_csv(args.output_dir/"by_transfer_group.csv",index=False)

    core=df[df["transfer_group"]=="core_crispri_essentiality"].copy()
    core_rows=[]
    for rep,g in core.groupby("representation"):
        core_rows.append({
            "representation":rep,
            "n_core_datasets":g["held_out_dataset"].nunique(),
            "median_core_auroc":g["auroc"].median(),
            "min_core_auroc":g["auroc"].min(),
            "median_core_auprc":g["auprc"].median(),
            "median_core_balanced_accuracy":g["balanced_accuracy"].median(),
            "median_core_regret_capture":g["regret_capture_fraction"].median(),
        })
    core_summary=pd.DataFrame(core_rows).sort_values(
        ["median_core_auprc","median_core_auroc"], ascending=False
    )
    core_summary.to_csv(args.output_dir/"core_crispri_summary.csv",index=False)

    print("\n=== CORE CRISPRi ESSENTIALITY ===")
    print(core_summary.to_string(index=False))
    print("\n=== ALL TRANSFER REGIMES ===")
    print(by_group.to_string(index=False))


if __name__=="__main__":
    main()
