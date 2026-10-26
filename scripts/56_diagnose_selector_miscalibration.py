#!/usr/bin/env python3
"""Diagnose why Correction Selector v1 is miscalibrated in its high-utility tail.

This is descriptive only. It does not tune or refit models.

Analyses:
- predicted vs realized utility association by dataset;
- linear calibration slope/intercept;
- utility quantiles within each dataset;
- which (k, alpha) experts dominate high predicted-utility bins;
- realized gain by selected expert configuration.

Goal: determine whether failure is due to calibration, expert-choice bias,
or cross-dataset ranking inversion before designing Selector v2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression


def safe_spearman(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3:
        return np.nan
    return float(spearmanr(x[ok],y[ok]).statistic)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--selector-configs", type=Path,
        default=Path("results/correction_selector_v1/outer_selected_configs.csv")
    )
    ap.add_argument(
        "--gate-predictions", type=Path,
        default=Path("results/split_safe_gate_v3/outer_predictions_by_perturbation.csv")
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=Path("results/correction_selector_v1_diagnostics")
    )
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)

    s=pd.read_csv(args.selector_configs)
    g=pd.read_csv(args.gate_predictions)[
        ["dataset","perturbation","predicted_probability","selected_threshold"]
    ].copy()
    g["gate_select"]=g["predicted_probability"]>=g["selected_threshold"]

    m=s.merge(g,on=["dataset","perturbation"],how="left",validate="many_to_one")
    m["realized_gain"]=m["pearson"]-m["global_pearson"]

    # Aggregate repeated appearances to biological perturbation level.
    p=(
        m.groupby(["dataset","perturbation"],as_index=False)
        .agg(
            gate_select=("gate_select","first"),
            predicted_delta=("predicted_delta_vs_global","median"),
            realized_gain=("realized_gain","median"),
            selected_alpha=("alpha","median"),
            selected_k=("pca_components","median"),
            oracle_regret=("oracle_regret_pearson","median"),
        )
    )
    p.to_csv(args.output_dir/"per_perturbation.csv",index=False)

    assoc=[]
    for ds,d in p.groupby("dataset"):
        lr=LinearRegression().fit(
            d[["predicted_delta"]].to_numpy(float),
            d["realized_gain"].to_numpy(float),
        )
        assoc.append({
            "dataset":ds,
            "n":len(d),
            "spearman_predicted_vs_realized":safe_spearman(
                d["predicted_delta"],d["realized_gain"]
            ),
            "calibration_slope":float(lr.coef_[0]),
            "calibration_intercept":float(lr.intercept_),
            "predicted_delta_median":float(d["predicted_delta"].median()),
            "realized_gain_mean":float(d["realized_gain"].mean()),
            "realized_gain_median":float(d["realized_gain"].median()),
        })
    assoc=pd.DataFrame(assoc)
    assoc.to_csv(args.output_dir/"association_by_dataset.csv",index=False)

    # Within-dataset utility deciles avoid Simpson/cross-context effects.
    bins=[]
    for ds,d in p.groupby("dataset"):
        d=d.copy()
        d["utility_bin"]=pd.qcut(
            d["predicted_delta"], q=10, duplicates="drop"
        )
        q=(
            d.groupby("utility_bin",observed=True)
             .agg(
                 n=("realized_gain","size"),
                 predicted_delta_median=("predicted_delta","median"),
                 realized_gain_mean=("realized_gain","mean"),
                 realized_gain_median=("realized_gain","median"),
                 fraction_helped=("realized_gain",lambda x:float((x>0).mean())),
                 fraction_harmed=("realized_gain",lambda x:float((x<0).mean())),
                 selected_alpha_median=("selected_alpha","median"),
                 selected_k_median=("selected_k","median"),
             )
             .reset_index()
        )
        q.insert(0,"dataset",ds)
        bins.append(q)
    bins=pd.concat(bins,ignore_index=True)
    bins.to_csv(args.output_dir/"utility_bins_by_dataset.csv",index=False)

    # Which experts are selected, and how do they actually perform?
    ex=(
        p.groupby(["dataset","selected_k","selected_alpha"],as_index=False)
         .agg(
             n=("realized_gain","size"),
             predicted_delta_median=("predicted_delta","median"),
             realized_gain_mean=("realized_gain","mean"),
             realized_gain_median=("realized_gain","median"),
             fraction_helped=("realized_gain",lambda x:float((x>0).mean())),
             fraction_harmed=("realized_gain",lambda x:float((x<0).mean())),
         )
         .sort_values(["dataset","n"],ascending=[True,False])
    )
    ex.to_csv(args.output_dir/"selected_expert_performance.csv",index=False)

    # High predicted utility tail composition.
    high=[]
    for ds,d in p.groupby("dataset"):
        q90=d["predicted_delta"].quantile(.90)
        h=d[d["predicted_delta"]>=q90].copy()
        top=(
            h.groupby(["selected_k","selected_alpha"],as_index=False)
             .agg(
                 n=("realized_gain","size"),
                 realized_gain_mean=("realized_gain","mean"),
                 realized_gain_median=("realized_gain","median"),
                 fraction_harmed=("realized_gain",lambda x:float((x<0).mean())),
             )
             .sort_values("n",ascending=False)
        )
        top.insert(0,"dataset",ds)
        top["high_tail_threshold"]=q90
        high.append(top)
    high=pd.concat(high,ignore_index=True)
    high.to_csv(args.output_dir/"high_predicted_tail_experts.csv",index=False)

    summary={
        "median_spearman_predicted_vs_realized":float(
            assoc["spearman_predicted_vs_realized"].median()
        ),
        "min_spearman_predicted_vs_realized":float(
            assoc["spearman_predicted_vs_realized"].min()
        ),
        "median_calibration_slope":float(assoc["calibration_slope"].median()),
        "purpose":"identify selector-v1 miscalibration before designing selector-v2",
        "tuning_performed":False,
    }
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2))

    print("\n=== SELECTOR ASSOCIATION BY DATASET ===")
    print(assoc.to_string(index=False))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary,indent=2))
    print("\n=== TOP HIGH-UTILITY-TAIL EXPERTS ===")
    print(high.groupby("dataset").head(5).to_string(index=False))


if __name__=="__main__":
    main()
