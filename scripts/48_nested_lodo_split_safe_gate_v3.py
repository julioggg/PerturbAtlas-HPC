#!/usr/bin/env python3
"""Nested-LODO benchmark for a strictly split-safe adaptive gate.

Predictors:
- half-A only;
- PCA geometry fitted only on half-A TRAIN perturbations within each seed;
- no full-data effect calibration;
- no intrinsic split-half reliability;
- no half-B/oracle information.

Repeated seed appearances are used for training with inverse-appearance weights,
but OUTER and INNER evaluation is performed once per biological perturbation
after aggregating probabilities across seeds.

This is the first deployment-oriented gate benchmark in the pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DROP = {
    "dataset","seed","perturbation",
    "n_cells_total",  # stricter: use the actual observed half-A count only
    "oracle_regret_pearson","global_pearson","oracle_pearson",
    "target_regret_gt_0_01","target_regret_gt_0_02","appearance_weight",
}
THRESHOLDS=np.linspace(.10,.90,33)


def model_specs(seed):
    return {
        "logistic":lambda:Pipeline([
            ("imp",SimpleImputer(strategy="median")),
            ("sc",StandardScaler()),
            ("m",LogisticRegression(C=1.0,class_weight="balanced",max_iter=5000,random_state=seed)),
        ]),
        "rf":lambda:Pipeline([
            ("imp",SimpleImputer(strategy="median")),
            ("m",RandomForestClassifier(
                n_estimators=500,min_samples_leaf=10,max_features="sqrt",
                class_weight="balanced_subsample",n_jobs=-1,random_state=seed)),
        ]),
        "extra":lambda:Pipeline([
            ("imp",SimpleImputer(strategy="median")),
            ("m",ExtraTreesClassifier(
                n_estimators=500,min_samples_leaf=10,max_features="sqrt",
                class_weight="balanced",n_jobs=-1,random_state=seed)),
        ]),
        "hgb":lambda:Pipeline([
            ("imp",SimpleImputer(strategy="median")),
            ("m",HistGradientBoostingClassifier(
                learning_rate=.05,max_iter=300,max_leaf_nodes=15,
                l2_regularization=1.0,random_state=seed)),
        ]),
    }


def prep_x(df, feats):
    x=df[feats].copy()
    for c in feats:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    return x


def add_perturbation_targets(df):
    """Attach perturbation-level median-regret targets without using them as features."""
    med=df.groupby(["dataset","perturbation"])["oracle_regret_pearson"].transform("median")
    out=df.copy()
    out["perturbation_median_regret"]=med
    out["perturbation_target_gt_0_01"]=(med>.01).astype(int)
    return out


def training_weights(df):
    """Equal dataset contribution and equal biological-perturbation contribution."""
    npert=df.groupby("dataset")["perturbation"].nunique()
    ds_factor=df["dataset"].map(lambda d:1.0/npert[d]).to_numpy(float)
    app=pd.to_numeric(df["appearance_weight"],errors="coerce").fillna(1.0).to_numpy(float)
    w=ds_factor*app
    return w*len(w)/w.sum()


def fit_model(model,x,y,w):
    model.fit(x,y,m__sample_weight=w)
    return model


def aggregate_predictions(df,p):
    q=df[["dataset","perturbation","oracle_regret_pearson"]].copy()
    q["p"]=p
    g=q.groupby(["dataset","perturbation"],as_index=False).agg(
        predicted_probability=("p","mean"),
        probability_sd=("p","std"),
        n_seed_appearances=("p","size"),
        median_regret=("oracle_regret_pearson","median"),
        mean_regret=("oracle_regret_pearson","mean"),
    )
    g["target"]=(g["median_regret"]>.01).astype(int)
    return g


def safe_auc(y,p):
    return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan


def safe_ap(y,p):
    return float(average_precision_score(y,p)) if np.sum(y)>0 else np.nan


def metrics(g,threshold):
    y=g["target"].to_numpy(int)
    p=g["predicted_probability"].to_numpy(float)
    z=p>=threshold
    r=np.clip(g["median_regret"].to_numpy(float),0,None)
    total=r.sum()
    captured=r[z].sum()
    return {
        "n_perturbations":int(len(g)),
        "positive_fraction":float(y.mean()),
        "auroc":safe_auc(y,p),
        "auprc":safe_ap(y,p),
        "brier":float(brier_score_loss(y,p)),
        "balanced_accuracy":float(balanced_accuracy_score(y,z)),
        "f1":float(f1_score(y,z,zero_division=0)),
        "recall":float(recall_score(y,z,zero_division=0)),
        "precision":float(precision_score(y,z,zero_division=0)),
        "selected_fraction":float(z.mean()),
        "regret_capture_fraction":float(captured/total) if total>0 else np.nan,
        "median_regret_selected":float(np.median(r[z])) if z.any() else np.nan,
        "median_regret_not_selected":float(np.median(r[~z])) if (~z).any() else np.nan,
    }


def select_threshold(inner_parts):
    rows=[]
    for t in THRESHOLDS:
        per=[]
        for g in inner_parts:
            m=metrics(g,t)
            per.append(m)
        d=pd.DataFrame(per)
        rows.append({
            "threshold":float(t),
            "median_balanced_accuracy":float(d["balanced_accuracy"].median()),
            "median_f1":float(d["f1"].median()),
            "median_regret_capture":float(d["regret_capture_fraction"].median()),
            "median_selected_fraction":float(d["selected_fraction"].median()),
        })
    tab=pd.DataFrame(rows).sort_values(
        ["median_balanced_accuracy","median_f1","median_regret_capture"],
        ascending=[False,False,False],
    ).reset_index(drop=True)
    return float(tab.iloc[0]["threshold"]),tab


def inner_select(train,feats,seed):
    rows=[]
    threshold_tables=[]
    for name,make in model_specs(seed).items():
        parts=[]
        fold_metrics=[]
        for held in sorted(train["dataset"].unique()):
            tr=train[train.dataset!=held].copy()
            va=train[train.dataset==held].copy()
            model=make()
            ytr=tr["perturbation_target_gt_0_01"].to_numpy(int)
            fit_model(model,prep_x(tr,feats),ytr,training_weights(tr))
            p=model.predict_proba(prep_x(va,feats))[:,1]
            agg=aggregate_predictions(va,p)
            parts.append(agg)

        threshold,tt=select_threshold(parts)
        tt.insert(0,"model",name)
        threshold_tables.append(tt)

        for agg in parts:
            mm=metrics(agg,threshold)
            fold_metrics.append(mm)
        fm=pd.DataFrame(fold_metrics)
        rows.append({
            "model":name,
            "threshold":threshold,
            "median_inner_auprc":float(fm.auprc.median()),
            "median_inner_auroc":float(fm.auroc.median()),
            "min_inner_auroc":float(fm.auroc.min()),
            "median_inner_balanced_accuracy":float(fm.balanced_accuracy.median()),
            "median_inner_regret_capture":float(fm.regret_capture_fraction.median()),
        })

    tab=pd.DataFrame(rows).sort_values(
        ["median_inner_auprc","median_inner_auroc","min_inner_auroc"],
        ascending=[False,False,False],
    ).reset_index(drop=True)
    best=str(tab.iloc[0].model)
    threshold=float(tab.iloc[0].threshold)
    return best,threshold,tab,pd.concat(threshold_tables,ignore_index=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "table",type=Path,nargs="?",
        default=Path("results/split_safe_gate_features/split_safe_gate_table.csv")
    )
    ap.add_argument("--output-dir",type=Path,default=Path("results/split_safe_gate_v3"))
    ap.add_argument("--seed",type=int,default=20261026)
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(args.table)
    df=add_perturbation_targets(df)

    feats=[c for c in df.columns if c not in DROP and c not in {
        "perturbation_median_regret","perturbation_target_gt_0_01"
    }]
    # Explicit provenance sanity check.
    forbidden=[c for c in feats if any(k in c.lower() for k in ["oracle","regret","global_pearson","target"])]
    if forbidden:
        raise SystemExit(f"Forbidden predictor columns detected: {forbidden}")

    outer=[]
    preds=[]
    selections=[]
    threshold_tables=[]

    for held in sorted(df.dataset.unique()):
        tr=df[df.dataset!=held].copy()
        te=df[df.dataset==held].copy()

        best,t,stab,tt=inner_select(tr,feats,args.seed)
        stab.insert(0,"outer_held_out_dataset",held)
        selections.append(stab)
        tt.insert(0,"outer_held_out_dataset",held)
        threshold_tables.append(tt)

        model=model_specs(args.seed)[best]()
        fit_model(
            model,prep_x(tr,feats),
            tr["perturbation_target_gt_0_01"].to_numpy(int),
            training_weights(tr)
        )
        p=model.predict_proba(prep_x(te,feats))[:,1]
        agg=aggregate_predictions(te,p)
        mm=metrics(agg,t)

        outer.append({
            "held_out_dataset":held,
            "selected_model":best,
            "selected_threshold":t,
            **mm,
        })
        agg["held_out_dataset"]=held
        agg["selected_model"]=best
        agg["selected_threshold"]=t
        preds.append(agg)

    out=pd.DataFrame(outer)
    out.to_csv(args.output_dir/"outer_metrics.csv",index=False)
    pd.concat(preds,ignore_index=True).to_csv(args.output_dir/"outer_predictions_by_perturbation.csv",index=False)
    pd.concat(selections,ignore_index=True).to_csv(args.output_dir/"inner_model_selection.csv",index=False)
    pd.concat(threshold_tables,ignore_index=True).to_csv(args.output_dir/"inner_threshold_search.csv",index=False)

    summary={
        "n_predictors":len(feats),
        "predictors":feats,
        "median_outer_auroc":float(out.auroc.median()),
        "min_outer_auroc":float(out.auroc.min()),
        "median_outer_auprc":float(out.auprc.median()),
        "median_outer_balanced_accuracy":float(out.balanced_accuracy.median()),
        "median_regret_capture_fraction":float(out.regret_capture_fraction.median()),
        "median_selected_fraction":float(out.selected_fraction.median()),
        "datasets_auroc_gt_0_7":int((out.auroc>.7).sum()),
        "evaluation_unit":"unique dataset-perturbation after averaging predicted probability across seed appearances",
        "feature_provenance":"half-A only; PCA models fit on half-A train perturbations within seed",
        "excluded_feature":"n_cells_total excluded; n_cells_halfA retained",
        "selection_protocol":"nested leave-one-dataset-out; model and threshold selected only on inner held-out training datasets",
    }
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2))

    print("\n=== SPLIT-SAFE GATE V3 SUMMARY ===")
    print(json.dumps(summary,indent=2))
    print("\n=== OUTER METRICS ===")
    print(out.to_string(index=False))


if __name__=="__main__":
    main()
