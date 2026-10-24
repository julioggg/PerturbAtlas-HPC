#!/usr/bin/env python3
"""Diagnose cross-context sign flips and feature regimes for adaptive gating.

This script does NOT train a final gate. It characterizes:
1. per-dataset univariate associations with oracle regret and binary gate target;
2. whether raw feature effects change sign across datasets;
3. whether within-context percentile / robust-z normalization stabilizes those
   associations using only unlabeled predictor distributions.

Outputs guide the architecture of a context-normalized adaptive gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


FEATURES = [
    "reliability_n_cells",
    "effect_l2",
    "effect_mean_abs",
    "expected_null_l2",
    "noise_adjusted_l2",
    "l2_to_null_ratio",
    "intrinsic_reliability",
]


def safe_rho(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<5:
        return np.nan
    return float(spearmanr(x[m],y[m]).statistic)


def safe_auc(y, x):
    y=np.asarray(y,int); x=np.asarray(x,float)
    m=np.isfinite(x)
    y=y[m]; x=x[m]
    if len(y)<5 or len(np.unique(y))<2:
        return np.nan
    a=float(roc_auc_score(y,x))
    return a


def robust_z(s: pd.Series) -> pd.Series:
    x=pd.to_numeric(s,errors="coerce")
    med=x.median()
    q25=x.quantile(.25); q75=x.quantile(.75)
    iqr=q75-q25
    if not np.isfinite(iqr) or iqr==0:
        return x*0.0
    return (x-med)/iqr


def pct_rank(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s,errors="coerce").rank(method="average",pct=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "atlas",type=Path,nargs="?",
        default=Path("results/adaptive_regime_atlas/adaptive_regime_atlas.csv")
    )
    ap.add_argument("--output-dir",type=Path,default=Path("results/adaptive_gate_diagnostics"))
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(args.atlas)
    target="target_personalize_regret_gt_0_01"
    df=df.dropna(subset=[target,"oracle_regret_median"]).copy()
    df[target]=df[target].astype(int)

    feats=[f for f in FEATURES if f in df.columns]

    for f in feats:
        df[f"{f}__pct"]=df.groupby("dataset")[f].transform(pct_rank)
        df[f"{f}__rz"]=df.groupby("dataset")[f].transform(robust_z)

    rows=[]
    for ds,g in df.groupby("dataset"):
        for f in feats:
            for rep,col in [
                ("raw",f),
                ("context_percentile",f"{f}__pct"),
                ("context_robust_z",f"{f}__rz"),
            ]:
                rho=safe_rho(g[col],g["oracle_regret_median"])
                auc=safe_auc(g[target],g[col])
                # orientation-free AUC measures discriminative strength while
                # preserving raw AUC separately to reveal sign reversals.
                auc_oriented=max(auc,1-auc) if np.isfinite(auc) else np.nan
                rows.append({
                    "dataset":ds,
                    "feature":f,
                    "representation":rep,
                    "n":len(g),
                    "positive_fraction":float(g[target].mean()),
                    "spearman_vs_regret":rho,
                    "auc_raw_direction":auc,
                    "auc_orientation_free":auc_oriented,
                })

    assoc=pd.DataFrame(rows)
    assoc.to_csv(args.output_dir/"per_dataset_feature_associations.csv",index=False)

    # Summarize sign consistency across datasets.
    summaries=[]
    for (f,rep),g in assoc.groupby(["feature","representation"]):
        rr=g["spearman_vs_regret"].dropna()
        signs=np.sign(rr.to_numpy())
        summaries.append({
            "feature":f,
            "representation":rep,
            "median_spearman":float(rr.median()) if len(rr) else np.nan,
            "min_spearman":float(rr.min()) if len(rr) else np.nan,
            "max_spearman":float(rr.max()) if len(rr) else np.nan,
            "n_positive_sign":int((signs>0).sum()),
            "n_negative_sign":int((signs<0).sum()),
            "sign_consistency":float(max((signs>0).mean(),(signs<0).mean())) if len(signs) else np.nan,
            "median_orientation_free_auc":float(g["auc_orientation_free"].median()),
        })
    stab=pd.DataFrame(summaries).sort_values(
        ["representation","sign_consistency","median_orientation_free_auc"],
        ascending=[True,False,False]
    )
    stab.to_csv(args.output_dir/"feature_stability_summary.csv",index=False)

    # Dataset-level target/regret context summary.
    context=df.groupby("dataset").agg(
        n=("perturbation","size"),
        positive_fraction=(target,"mean"),
        median_regret=("oracle_regret_median","median"),
        q90_regret=("oracle_regret_median",lambda x:x.quantile(.90)),
        median_reliability=("intrinsic_reliability","median"),
        median_noise_adjusted_l2=("noise_adjusted_l2","median"),
        median_effect_l2=("effect_l2","median"),
    ).reset_index()
    context.to_csv(args.output_dir/"dataset_regime_summary.csv",index=False)

    summary={
        "n_rows":int(len(df)),
        "n_datasets":int(df["dataset"].nunique()),
        "features":feats,
        "representations":["raw","context_percentile","context_robust_z"],
        "purpose":(
            "Detect cross-context sign reversals and assess whether unlabeled "
            "within-context normalization stabilizes gate predictors."
        ),
        "warning":(
            "This is diagnostic analysis on development benchmark datasets; "
            "do not treat repeated inspection as pristine external validation."
        ),
    }
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2))

    print("\n=== DATASET REGIMES ===")
    print(context.to_string(index=False))
    print("\n=== FEATURE STABILITY ===")
    print(stab.to_string(index=False))


if __name__=="__main__":
    main()
