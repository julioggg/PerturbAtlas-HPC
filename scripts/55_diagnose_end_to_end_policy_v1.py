#!/usr/bin/env python3
"""Diagnose end-to-end adaptive policy v1 without tuning.

Purpose
-------
Determine WHY Gate + Selector still harms some perturbations and whether a
selector-confidence guardrail is justified.

No model is refit and no threshold is selected here. This is descriptive only.

Outputs:
- dataset-level total oracle-regret recovery
- metrics conditional on Gate-selected perturbations
- realized gain by predicted selector-utility bins
- realized gain by Gate-probability bins
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def wquantile_bins(x, q=10):
    x=pd.Series(x)
    try:
        return pd.qcut(x,q=q,duplicates="drop")
    except Exception:
        return pd.cut(x,bins=min(q,max(2,x.nunique())),duplicates="drop")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--selector-configs",type=Path,
        default=Path("results/correction_selector_v1/outer_selected_configs.csv")
    )
    ap.add_argument(
        "--gate-predictions",type=Path,
        default=Path("results/split_safe_gate_v3/outer_predictions_by_perturbation.csv")
    )
    ap.add_argument(
        "--output-dir",type=Path,
        default=Path("results/end_to_end_diagnostics_v1")
    )
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)

    s=pd.read_csv(args.selector_configs)
    g=pd.read_csv(args.gate_predictions)

    g=g[["dataset","perturbation","predicted_probability","selected_threshold"]].copy()
    g["gate_select"]=g["predicted_probability"]>=g["selected_threshold"]

    m=s.merge(g,on=["dataset","perturbation"],how="inner",validate="many_to_one")
    if "predicted_delta_vs_global" not in m.columns:
        raise SystemExit("selector configs missing predicted_delta_vs_global")

    m["realized_gain"]=m["pearson"]-m["global_pearson"]
    m["oracle_regret_pos"]=m["oracle_regret_pearson"].clip(lower=0)
    m["recoverable_gain"]=np.minimum(
        np.maximum(m["realized_gain"],0),
        m["oracle_regret_pos"]
    )
    m["policy_gain"]=np.where(m["gate_select"],m["realized_gain"],0.0)
    m["policy_recoverable_gain"]=np.where(
        m["gate_select"],m["recoverable_gain"],0.0
    )

    # Perturbation-level aggregation across repeated appearances.
    p=(
        m.groupby(["dataset","perturbation"],as_index=False)
        .agg(
            gate_probability=("predicted_probability","first"),
            gate_threshold=("selected_threshold","first"),
            gate_select=("gate_select","first"),
            predicted_delta=("predicted_delta_vs_global","median"),
            realized_gain=("realized_gain","median"),
            oracle_regret=("oracle_regret_pearson","median"),
            policy_gain=("policy_gain","median"),
        )
    )
    p["oracle_regret_pos"]=p["oracle_regret"].clip(lower=0)
    p["policy_recoverable_gain"]=np.minimum(
        np.maximum(p["policy_gain"],0),
        p["oracle_regret_pos"]
    )
    p.to_csv(args.output_dir/"per_perturbation_diagnostics.csv",index=False)

    ds_rows=[]
    for ds,d in p.groupby("dataset"):
        selected=d[d.gate_select].copy()
        total_oracle=d["oracle_regret_pos"].sum()
        recovered=d["policy_recoverable_gain"].sum()
        ds_rows.append({
            "dataset":ds,
            "n":len(d),
            "gate_selected_fraction":float(d.gate_select.mean()),
            "mean_policy_gain":float(d.policy_gain.mean()),
            "median_policy_gain":float(d.policy_gain.median()),
            "fraction_helped_all":float((d.policy_gain>0).mean()),
            "fraction_harmed_all":float((d.policy_gain<0).mean()),
            "selected_mean_realized_gain":float(selected.realized_gain.mean()) if len(selected) else np.nan,
            "selected_median_realized_gain":float(selected.realized_gain.median()) if len(selected) else np.nan,
            "selected_fraction_helped":float((selected.realized_gain>0).mean()) if len(selected) else np.nan,
            "selected_fraction_harmed":float((selected.realized_gain<0).mean()) if len(selected) else np.nan,
            "total_positive_oracle_regret":float(total_oracle),
            "total_recoverable_gain_realized":float(recovered),
            "fraction_total_oracle_value_recovered":float(recovered/total_oracle) if total_oracle>0 else np.nan,
            "median_predicted_delta_selected":float(selected.predicted_delta.median()) if len(selected) else np.nan,
        })
    ds=pd.DataFrame(ds_rows)
    ds.to_csv(args.output_dir/"dataset_diagnostics.csv",index=False)

    # Utility bins across all gate-selected perturbations.
    z=p[p.gate_select].copy()
    z["predicted_delta_bin"]=wquantile_bins(z["predicted_delta"],10)
    util=(
        z.groupby("predicted_delta_bin",observed=True)
        .agg(
            n=("realized_gain","size"),
            predicted_delta_median=("predicted_delta","median"),
            realized_gain_mean=("realized_gain","mean"),
            realized_gain_median=("realized_gain","median"),
            fraction_helped=("realized_gain",lambda x:float((x>0).mean())),
            fraction_harmed=("realized_gain",lambda x:float((x<0).mean())),
            oracle_regret_median=("oracle_regret","median"),
        )
        .reset_index()
    )
    util.to_csv(args.output_dir/"selector_utility_bins.csv",index=False)

    z["gate_probability_bin"]=wquantile_bins(z["gate_probability"],10)
    gatebins=(
        z.groupby("gate_probability_bin",observed=True)
        .agg(
            n=("realized_gain","size"),
            gate_probability_median=("gate_probability","median"),
            realized_gain_mean=("realized_gain","mean"),
            realized_gain_median=("realized_gain","median"),
            fraction_helped=("realized_gain",lambda x:float((x>0).mean())),
            fraction_harmed=("realized_gain",lambda x:float((x<0).mean())),
        )
        .reset_index()
    )
    gatebins.to_csv(args.output_dir/"gate_probability_bins.csv",index=False)

    summary={
        "n_perturbations":int(len(p)),
        "median_dataset_fraction_total_oracle_value_recovered":float(
            ds["fraction_total_oracle_value_recovered"].median()
        ),
        "mean_dataset_policy_gain":float(ds["mean_policy_gain"].mean()),
        "purpose":"diagnose whether selector predicted utility can safely guard personalization",
        "tuning_performed":False,
        "note":"Use these diagnostics to design a nested-LODO utility threshold; do not choose a final threshold from outer results."
    }
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2))

    print("\n=== END-TO-END DIAGNOSTICS ===")
    print(ds.to_string(index=False))
    print("\n=== SELECTOR UTILITY BINS ===")
    print(util.to_string(index=False))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
