#!/usr/bin/env python3
"""Nested-LODO adaptive gate v2 with regime-relative representations.

This benchmark tests whether cross-context transfer improves when each
perturbation is represented jointly by:
- raw deployable predictors;
- unlabeled within-context percentile / robust-z features;
- context-level predictor summaries;
- leakage-safe geometric ratios derived from deployable predictors.

No oracle regret labels from the outer held-out dataset enter training,
model selection, feature construction, or threshold selection.
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
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE = [
    "reliability_n_cells",
    "effect_l2",
    "effect_mean_abs",
    "expected_null_l2",
    "noise_adjusted_l2",
    "l2_to_null_ratio",
]
LOG_FEATURES = {
    "reliability_n_cells","effect_l2","effect_mean_abs",
    "expected_null_l2","l2_to_null_ratio"
}
THRESHOLDS=np.linspace(.10,.90,33)


def robust_z(s):
    x=pd.to_numeric(s,errors="coerce")
    med=x.median(); q25=x.quantile(.25); q75=x.quantile(.75); iqr=q75-q25
    if not np.isfinite(iqr) or iqr==0: return x*0.0
    return (x-med)/iqr


def pct(s):
    return pd.to_numeric(s,errors="coerce").rank(method="average",pct=True)


def engineer(df):
    out=df.copy()
    eps=1e-8
    # Per-row geometry, all derived from deployable predictors.
    out["signal_fraction_noise_adjusted"] = (
        out["noise_adjusted_l2"] / (out["effect_l2"] + eps)
    )
    out["null_fraction_of_effect"] = (
        out["expected_null_l2"] / (out["effect_l2"] + eps)
    )
    out["effect_excess_over_null"] = (
        out["effect_l2"] - out["expected_null_l2"]
    )
    geom=[
        "signal_fraction_noise_adjusted",
        "null_fraction_of_effect",
        "effect_excess_over_null",
    ]
    allbase=BASE+geom

    relative=[]
    context=[]
    for c in allbase:
        out[f"rel_pct__{c}"]=out.groupby("dataset")[c].transform(pct)
        out[f"rel_rz__{c}"]=out.groupby("dataset")[c].transform(robust_z)
        relative += [f"rel_pct__{c}",f"rel_rz__{c}"]

        g=out.groupby("dataset")[c]
        out[f"ctx_med__{c}"]=g.transform("median")
        out[f"ctx_iqr__{c}"]=g.transform(lambda x:x.quantile(.75)-x.quantile(.25))
        context += [f"ctx_med__{c}",f"ctx_iqr__{c}"]

    return out, allbase, relative, context


def prep_x(df, feats):
    x=df[feats].copy()
    for c in feats:
        v=pd.to_numeric(x[c],errors="coerce")
        root=c
        for prefix in ["rel_pct__","rel_rz__","ctx_med__","ctx_iqr__"]:
            if root.startswith(prefix):
                root=root[len(prefix):]
        if root in LOG_FEATURES:
            v=np.where(v>=0,np.log1p(v),np.nan)
        x[c]=v
    return x


def weights(df):
    counts=df.dataset.value_counts()
    w=df.dataset.map(lambda d:1.0/counts[d]).to_numpy(float)
    return w*len(w)/w.sum()


def specs(seed):
    return {
      "logistic":lambda:Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("sc",StandardScaler()),
        ("m",LogisticRegression(C=1,class_weight="balanced",max_iter=5000,random_state=seed))
      ]),
      "rf":lambda:Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("m",RandomForestClassifier(n_estimators=400,min_samples_leaf=10,max_features="sqrt",
             class_weight="balanced_subsample",n_jobs=-1,random_state=seed))
      ]),
      "extra":lambda:Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("m",ExtraTreesClassifier(n_estimators=400,min_samples_leaf=10,max_features="sqrt",
             class_weight="balanced",n_jobs=-1,random_state=seed))
      ]),
      "hgb":lambda:Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("m",HistGradientBoostingClassifier(learning_rate=.05,max_iter=300,max_leaf_nodes=15,
             l2_regularization=1,random_state=seed))
      ])
    }


def fit(m,x,y,w):
    m.fit(x,y,m__sample_weight=w); return m


def auc(y,p):
    return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan


def ap(y,p):
    return float(average_precision_score(y,p)) if np.sum(y)>0 else np.nan


def choose_threshold(y,p):
    rows=[]
    for t in THRESHOLDS:
        z=p>=t
        rows.append((t,balanced_accuracy_score(y,z),f1_score(y,z,zero_division=0)))
    rows=sorted(rows,key=lambda q:(-q[1],-q[2]))
    return float(rows[0][0])


def utility(regret,z):
    r=np.asarray(regret,float); z=np.asarray(z,bool)
    total=np.nansum(np.clip(r,0,None))
    cap=np.nansum(np.clip(r[z],0,None))
    return float(cap/total) if total>0 else np.nan


def metrics(y,p,t,regret):
    z=p>=t
    return {
      "auroc":auc(y,p),
      "auprc":ap(y,p),
      "brier":float(brier_score_loss(y,p)),
      "balanced_accuracy":float(balanced_accuracy_score(y,z)),
      "f1":float(f1_score(y,z,zero_division=0)),
      "recall":float(recall_score(y,z,zero_division=0)),
      "precision":float(precision_score(y,z,zero_division=0)),
      "selected_fraction":float(np.mean(z)),
      "regret_capture_fraction":utility(regret,z),
      "median_regret_selected":float(np.nanmedian(np.asarray(regret)[z])) if z.any() else np.nan,
      "median_regret_not_selected":float(np.nanmedian(np.asarray(regret)[~z])) if (~z).any() else np.nan,
    }


def inner_select(train,feats,target,seed):
    rows=[]
    for name,make in specs(seed).items():
        ys=[]; ps=[]; rs=[]
        for held in sorted(train.dataset.unique()):
            tr=train[train.dataset!=held]; va=train[train.dataset==held]
            m=make()
            fit(m,prep_x(tr,feats),tr[target].astype(int).to_numpy(),weights(tr))
            p=m.predict_proba(prep_x(va,feats))[:,1]
            ys.append(va[target].astype(int).to_numpy())
            ps.append(p)
            rs.append(va.oracle_regret_median.to_numpy(float))
        y=np.concatenate(ys); p=np.concatenate(ps); r=np.concatenate(rs)
        t=choose_threshold(y,p)
        mm=metrics(y,p,t,r)
        rows.append({"model":name,"threshold":t,**mm})
    tab=pd.DataFrame(rows).sort_values(["auprc","auroc","balanced_accuracy"],ascending=False)
    q=tab.iloc[0]
    return str(q.model),float(q.threshold),tab


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("atlas",type=Path,nargs="?",default=Path("results/adaptive_regime_atlas/adaptive_regime_atlas.csv"))
    pa.add_argument("--output-dir",type=Path,default=Path("results/adaptive_gate_v2"))
    pa.add_argument("--seed",type=int,default=20261023)
    args=pa.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)

    target="target_personalize_regret_gt_0_01"
    df=pd.read_csv(args.atlas).dropna(subset=[target,"oracle_regret_median"]).copy()
    df[target]=df[target].astype(int)
    df, raw, rel, ctx=engineer(df)

    configs={
      "raw":raw,
      "relative_only":rel,
      "raw_relative":raw+rel,
      "raw_relative_context":raw+rel+ctx,
    }

    outer=[]; sels=[]
    for rep,feats in configs.items():
      for held in sorted(df.dataset.unique()):
        tr=df[df.dataset!=held].copy(); te=df[df.dataset==held].copy()
        best,t,tab=inner_select(tr,feats,target,args.seed)
        tab.insert(0,"representation",rep)
        tab.insert(1,"outer_held_out_dataset",held)
        sels.append(tab)

        m=specs(args.seed)[best]()
        fit(m,prep_x(tr,feats),tr[target].astype(int).to_numpy(),weights(tr))
        p=m.predict_proba(prep_x(te,feats))[:,1]
        met=metrics(te[target].astype(int).to_numpy(),p,t,te.oracle_regret_median.to_numpy(float))
        outer.append({
          "representation":rep,"held_out_dataset":held,"selected_model":best,
          "selected_threshold":t,"n_test":len(te),"positive_fraction":float(te[target].mean()),**met
        })

    out=pd.DataFrame(outer)
    sel=pd.concat(sels,ignore_index=True)
    out.to_csv(args.output_dir/"outer_metrics.csv",index=False)
    sel.to_csv(args.output_dir/"inner_model_selection.csv",index=False)

    summary=[]
    for rep,g in out.groupby("representation"):
        summary.append({
          "representation":rep,
          "median_auroc":float(g.auroc.median()),
          "min_auroc":float(g.auroc.min()),
          "median_auprc":float(g.auprc.median()),
          "median_balanced_accuracy":float(g.balanced_accuracy.median()),
          "median_regret_capture":float(g.regret_capture_fraction.median()),
          "median_selected_fraction":float(g.selected_fraction.median()),
          "datasets_auroc_gt_0_7":int((g.auroc>.7).sum()),
        })
    s=pd.DataFrame(summary).sort_values(["median_auprc","median_auroc"],ascending=False)
    s.to_csv(args.output_dir/"summary.csv",index=False)

    meta={
      "protocol":"nested LODO; outer dataset labels excluded from model/threshold selection",
      "representations":list(configs),
      "transductive_context_note":"relative/context features use only unlabeled predictor distributions of each context",
      "purpose":"test regime-relative representation before introducing a custom gate architecture",
    }
    (args.output_dir/"run_summary.json").write_text(json.dumps(meta,indent=2))
    print("\n=== GATE V2 SUMMARY ===")
    print(s.to_string(index=False))
    print("\n=== OUTER ===")
    print(out.to_string(index=False))


if __name__=="__main__":
    main()
