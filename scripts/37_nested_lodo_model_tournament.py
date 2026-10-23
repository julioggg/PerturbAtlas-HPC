#!/usr/bin/env python3
"""Nested leave-one-dataset-out model tournament for cross-context gain prediction.

Goals
-----
1. Improve zero-shot cross-dataset generalization without tuning on the outer
   held-out dataset.
2. Separate two tasks:
   - RANKING: predict which perturbations benefit more within a new context.
   - ABSOLUTE: predict the magnitude of gain.
3. Compare linear and nonlinear models under nested dataset-level validation.
4. Test unlabeled context summaries as zero-shot adaptation features.

Important
---------
The outer held-out dataset is never used for model/hyperparameter selection.
All selection happens by leave-one-training-dataset-out validation inside the
remaining datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEPLOYABLE = [
    "reliability_n_cells",
    "effect_l2",
    "effect_mean_abs",
    "expected_null_l2",
    "noise_adjusted_l2",
    "l2_to_null_ratio",
]
ORACLE = ["intrinsic_reliability", *DEPLOYABLE]

LOG_FEATURES = {
    "reliability_n_cells",
    "effect_l2",
    "effect_mean_abs",
    "expected_null_l2",
    "l2_to_null_ratio",
}


def safe_rho(y, p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    m=np.isfinite(y)&np.isfinite(p)
    if m.sum()<3: return np.nan
    return float(spearmanr(y[m],p[m]).statistic)


def rank01(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True)


def add_context_features(df: pd.DataFrame, base: list[str]) -> tuple[pd.DataFrame,list[str]]:
    out=df.copy()
    new=[]
    for c in base:
        if c not in out: continue
        g=out.groupby("dataset")[c]
        med=g.transform("median")
        q25=g.transform(lambda x:x.quantile(.25))
        q75=g.transform(lambda x:x.quantile(.75))
        for name,val in [
            (f"ctx_{c}_median",med),
            (f"ctx_{c}_iqr",q75-q25),
        ]:
            out[name]=val
            new.append(name)
    return out, base+new


def prep_x(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    x=df[feats].copy()
    for c in feats:
        v=pd.to_numeric(x[c],errors="coerce")
        if any(c==lf or c.startswith(f"ctx_{lf}_") for lf in LOG_FEATURES):
            v=np.where(v>=0,np.log1p(v),np.nan)
        x[c]=v
    return x


def model_specs(seed:int):
    return {
        "ridge_1": lambda: Pipeline([
            ("imp",SimpleImputer(strategy="median")),
            ("sc",StandardScaler()),
            ("m",Ridge(alpha=1.0)),
        ]),
        "ridge_10": lambda: Pipeline([
            ("imp",SimpleImputer(strategy="median")),
            ("sc",StandardScaler()),
            ("m",Ridge(alpha=10.0)),
        ]),
        "elastic": lambda: Pipeline([
            ("imp",SimpleImputer(strategy="median")),
            ("sc",StandardScaler()),
            ("m",ElasticNet(alpha=0.001,l1_ratio=0.25,max_iter=10000,random_state=seed)),
        ]),
        "rf": lambda: Pipeline([
            ("imp",SimpleImputer(strategy="median")),
            ("m",RandomForestRegressor(
                n_estimators=300,min_samples_leaf=10,max_features="sqrt",
                n_jobs=-1,random_state=seed
            )),
        ]),
        "extra": lambda: Pipeline([
            ("imp",SimpleImputer(strategy="median")),
            ("m",ExtraTreesRegressor(
                n_estimators=300,min_samples_leaf=10,max_features="sqrt",
                n_jobs=-1,random_state=seed
            )),
        ]),
        "hgb": lambda: Pipeline([
            ("imp",SimpleImputer(strategy="median")),
            ("m",HistGradientBoostingRegressor(
                learning_rate=0.05,max_iter=250,max_leaf_nodes=15,
                l2_regularization=1.0,random_state=seed
            )),
        ]),
    }


def target(df:pd.DataFrame, task:str):
    y=df["gain_over_identity_median"].astype(float)
    if task=="ranking":
        return df.assign(_y=df.groupby("dataset")["gain_over_identity_median"].transform(rank01))["_y"].to_numpy(float)
    return y.to_numpy(float)


def metrics(y,p):
    return {
        "rho":safe_rho(y,p),
        "rmse":float(np.sqrt(mean_squared_error(y,p))),
        "mae":float(mean_absolute_error(y,p)),
    }


def inner_select(train:pd.DataFrame, feats:list[str], task:str, seed:int):
    specs=model_specs(seed)
    rows=[]
    dsets=sorted(train.dataset.unique())
    for name,make in specs.items():
        fold=[]
        for v in dsets:
            tr=train[train.dataset!=v]
            va=train[train.dataset==v]
            if len(va)<3: continue
            m=make()
            m.fit(prep_x(tr,feats),target(tr,task))
            p=m.predict(prep_x(va,feats))
            # For ranking task, outer/inner evaluation is still against raw gain ranking.
            yy=va["gain_over_identity_median"].to_numpy(float) if task=="ranking" else target(va,task)
            fold.append(metrics(yy,p))
        if not fold: continue
        fr=pd.DataFrame(fold)
        rows.append({
            "model":name,
            "median_inner_rho":float(fr.rho.median()),
            "median_inner_rmse":float(fr.rmse.median()),
            "median_inner_mae":float(fr.mae.median()),
        })
    tab=pd.DataFrame(rows).sort_values(
        ["median_inner_rho","median_inner_rmse"],
        ascending=[False,True]
    ).reset_index(drop=True)
    return str(tab.iloc[0].model),tab


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("master",type=Path,nargs="?",default=Path("results/multidataset_master/perturbation_master_table.csv"))
    ap.add_argument("--output-dir",type=Path,default=Path("results/nested_lodo_tournament"))
    ap.add_argument("--seed",type=int,default=20261023)
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(args.master).dropna(subset=["gain_over_identity_median"]).copy()

    configs=[]
    for fs_name,base in [("deployable",DEPLOYABLE),("oracle",ORACLE)]:
        configs.append((fs_name,base,False))
        configs.append((fs_name+"_context",base,True))

    outer=[]
    selections=[]

    for task in ["ranking","absolute"]:
      for fs_name,base,use_ctx in configs:
        work=df.copy()
        feats=base
        if use_ctx:
            work,feats=add_context_features(work,base)

        for held in sorted(work.dataset.unique()):
            tr=work[work.dataset!=held].copy()
            te=work[work.dataset==held].copy()
            best,tab=inner_select(tr,feats,task,args.seed)

            tab.insert(0,"task",task)
            tab.insert(1,"feature_set",fs_name)
            tab.insert(2,"outer_held_out_dataset",held)
            selections.append(tab)

            m=model_specs(args.seed)[best]()
            m.fit(prep_x(tr,feats),target(tr,task))
            pred=m.predict(prep_x(te,feats))
            yraw=te["gain_over_identity_median"].to_numpy(float)

            rho=safe_rho(yraw,pred)
            if task=="absolute":
                rmse=float(np.sqrt(mean_squared_error(yraw,pred)))
                mae=float(mean_absolute_error(yraw,pred))
                base_pred=np.full_like(yraw,np.median(tr["gain_over_identity_median"]))
                base_rmse=float(np.sqrt(mean_squared_error(yraw,base_pred)))
                base_mae=float(mean_absolute_error(yraw,base_pred))
            else:
                # Ranking scores have no meaningful raw-gain RMSE.
                rmse=mae=base_rmse=base_mae=np.nan

            outer.append({
                "task":task,
                "feature_set":fs_name,
                "held_out_dataset":held,
                "selected_model":best,
                "n_test":len(te),
                "spearman_pred_vs_gain":rho,
                "rmse":rmse,
                "mae":mae,
                "baseline_rmse":base_rmse,
                "baseline_mae":base_mae,
                "rmse_improvement_vs_baseline":base_rmse-rmse if task=="absolute" else np.nan,
                "mae_improvement_vs_baseline":base_mae-mae if task=="absolute" else np.nan,
            })

    out=pd.DataFrame(outer)
    sel=pd.concat(selections,ignore_index=True)
    out.to_csv(args.output_dir/"outer_metrics.csv",index=False)
    sel.to_csv(args.output_dir/"inner_model_selection.csv",index=False)

    summary=[]
    for (task,fs),g in out.groupby(["task","feature_set"]):
        summary.append({
            "task":task,
            "feature_set":fs,
            "median_outer_spearman":float(np.nanmedian(g.spearman_pred_vs_gain)),
            "min_outer_spearman":float(np.nanmin(g.spearman_pred_vs_gain)),
            "datasets_rho_gt_0_4":int((g.spearman_pred_vs_gain>0.4).sum()),
            "median_rmse_improvement_vs_baseline":float(np.nanmedian(g.rmse_improvement_vs_baseline)) if task=="absolute" else np.nan,
            "datasets_rmse_better_than_baseline":int((g.rmse_improvement_vs_baseline>0).sum()) if task=="absolute" else np.nan,
        })
    s=pd.DataFrame(summary).sort_values(["task","median_outer_spearman"],ascending=[True,False])
    s.to_csv(args.output_dir/"summary.csv",index=False)

    meta={
        "selection_protocol":"nested leave-one-training-dataset-out; outer held-out dataset never used for selection",
        "ranking_target":"within-training-dataset percentile gain",
        "absolute_target":"raw median gain over identity",
        "context_features":"unlabeled per-dataset medians and IQRs of available predictors",
        "warning":"Do not select the final method by looking repeatedly at outer-test performance without freezing a protocol."
    }
    (args.output_dir/"run_summary.json").write_text(json.dumps(meta,indent=2))

    print("\n=== SUMMARY ===")
    print(s.to_string(index=False))
    print("\n=== OUTER METRICS ===")
    print(out.to_string(index=False))


if __name__=="__main__":
    main()
