#!/usr/bin/env python3
"""End-to-end adaptive policy v1: Gate v3 + Correction Selector v1.

Policy
------
For supported core CRISPRi contexts:
- if Gate v3 selects a perturbation, use Selector v1's chosen (k, alpha);
- otherwise retain the validation-locked global shrinkage configuration.

This script does not retrain or tune anything. It combines outer-held-out
predictions from already nested-LODO components and evaluates realized Pearson
gain over the validation-locked global policy.

Evaluation is aggregated at the biological perturbation level across repeated
seed appearances, matching Gate v3's evaluation unit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CORE = [
    "replogle_k562",
    "replogle_rpe1",
    "nadig25hepg2",
    "nadig25jurkat",
]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--gate-predictions", type=Path,
        default=Path("results/split_safe_gate_v3/outer_predictions_by_perturbation.csv")
    )
    ap.add_argument(
        "--selector-configs", type=Path,
        default=Path("results/correction_selector_v1/outer_selected_configs.csv")
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=Path("results/end_to_end_adaptive_v1")
    )
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)

    gate=pd.read_csv(args.gate_predictions)
    sel=pd.read_csv(args.selector_configs)

    gate=gate[gate["dataset"].isin(CORE)].copy()
    sel=sel[sel["dataset"].isin(CORE)].copy()

    required_gate={"dataset","perturbation","predicted_probability","selected_threshold"}
    required_sel={
        "dataset","seed","perturbation","pearson","global_pearson",
        "oracle_pearson","oracle_regret_pearson","pca_components","alpha"
    }
    if not required_gate.issubset(gate.columns):
        raise SystemExit(f"Missing gate columns: {sorted(required_gate-set(gate.columns))}")
    if not required_sel.issubset(sel.columns):
        raise SystemExit(f"Missing selector columns: {sorted(required_sel-set(sel.columns))}")

    gate["gate_select"]=gate["predicted_probability"]>=gate["selected_threshold"]

    m=sel.merge(
        gate[["dataset","perturbation","predicted_probability",
              "selected_threshold","gate_select"]],
        on=["dataset","perturbation"],
        how="left",
        validate="many_to_one",
    )
    if m["gate_select"].isna().any():
        bad=m.loc[m["gate_select"].isna(),["dataset","perturbation"]].drop_duplicates()
        raise SystemExit(f"Missing gate predictions for {len(bad)} perturbations")

    # Realized policy score per seed appearance.
    m["adaptive_pearson"]=np.where(
        m["gate_select"], m["pearson"], m["global_pearson"]
    )
    m["adaptive_gain_over_global"]=m["adaptive_pearson"]-m["global_pearson"]
    m["adaptive_remaining_oracle_regret"]=m["oracle_pearson"]-m["adaptive_pearson"]
    m["adaptive_harmed"]=m["adaptive_gain_over_global"]<0
    m["adaptive_helped"]=m["adaptive_gain_over_global"]>0
    m.to_csv(args.output_dir/"per_seed_policy_results.csv",index=False)

    # Aggregate seed appearances to one biological perturbation.
    agg=(
        m.groupby(["dataset","perturbation"],as_index=False)
        .agg(
            gate_probability=("predicted_probability","first"),
            gate_threshold=("selected_threshold","first"),
            gate_select=("gate_select","first"),
            n_appearances=("seed","nunique"),
            global_pearson=("global_pearson","median"),
            selector_pearson=("pearson","median"),
            adaptive_pearson=("adaptive_pearson","median"),
            oracle_pearson=("oracle_pearson","median"),
            oracle_regret=("oracle_regret_pearson","median"),
            adaptive_gain_over_global=("adaptive_gain_over_global","median"),
            adaptive_remaining_oracle_regret=("adaptive_remaining_oracle_regret","median"),
            selector_k=("pca_components","median"),
            selector_alpha=("alpha","median"),
        )
    )
    agg["helped"]=agg["adaptive_gain_over_global"]>0
    agg["harmed"]=agg["adaptive_gain_over_global"]<0
    denom=agg["oracle_regret"].clip(lower=0)
    agg["oracle_value_recovered_fraction"]=np.where(
        denom>0,
        np.clip(
            (denom-agg["adaptive_remaining_oracle_regret"].clip(lower=0))/denom,
            -10,1
        ),
        np.nan,
    )
    agg.to_csv(args.output_dir/"per_perturbation_results.csv",index=False)

    rows=[]
    for ds,g in agg.groupby("dataset"):
        rows.append({
            "dataset":ds,
            "n_perturbations":len(g),
            "gate_selected_fraction":float(g["gate_select"].mean()),
            "median_gain_over_global":float(g["adaptive_gain_over_global"].median()),
            "mean_gain_over_global":float(g["adaptive_gain_over_global"].mean()),
            "q10_gain_over_global":float(g["adaptive_gain_over_global"].quantile(.10)),
            "q90_gain_over_global":float(g["adaptive_gain_over_global"].quantile(.90)),
            "fraction_helped":float(g["helped"].mean()),
            "fraction_harmed":float(g["harmed"].mean()),
            "median_oracle_regret":float(g["oracle_regret"].median()),
            "median_remaining_oracle_regret":float(
                g["adaptive_remaining_oracle_regret"].median()
            ),
            "median_oracle_value_recovered_fraction":float(
                g["oracle_value_recovered_fraction"].median(skipna=True)
            ),
        })
    ds=pd.DataFrame(rows)
    ds.to_csv(args.output_dir/"dataset_metrics.csv",index=False)

    summary={
        "policy":"Gate v3 decides whether to personalize; Selector v1 chooses correction; otherwise use validation-locked global shrinkage",
        "datasets":CORE,
        "n_unique_perturbations":int(len(agg)),
        "median_dataset_gain_over_global":float(ds["median_gain_over_global"].median()),
        "mean_dataset_gain_over_global":float(ds["mean_gain_over_global"].mean()),
        "median_dataset_fraction_helped":float(ds["fraction_helped"].median()),
        "median_dataset_fraction_harmed":float(ds["fraction_harmed"].median()),
        "median_dataset_remaining_oracle_regret":float(
            ds["median_remaining_oracle_regret"].median()
        ),
        "median_dataset_oracle_value_recovered_fraction":float(
            ds["median_oracle_value_recovered_fraction"].median()
        ),
        "evaluation_unit":"unique dataset-perturbation after median aggregation across seed appearances",
        "uses_outer_test_for_training_or_selection":False,
        "note":"Development benchmark only; frozen external validation still required."
    }
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2))

    print("\n=== END-TO-END ADAPTIVE V1 SUMMARY ===")
    print(json.dumps(summary,indent=2))
    print("\n=== DATASET METRICS ===")
    print(ds.to_string(index=False))

if __name__=="__main__":
    main()
