#!/usr/bin/env python3
"""Nested-LODO correction selector v1 for supported CRISPRi contexts.

Goal
----
For each held-out perturbation, predict which correction expert (k, alpha)
will improve Pearson correlation relative to the validation-locked GLOBAL
shrinkage configuration.

Training target
---------------
delta_vs_global = pearson(config) - global_pearson

Predictors
----------
- strictly split-safe half-A perturbation features;
- candidate expert descriptors (alpha, PCA dimension);
- no half-B/oracle quantities as predictors.

Protocol
--------
- core CRISPRi datasets only;
- outer leave-one-dataset-out evaluation;
- inner leave-one-training-dataset-out regressor selection;
- selection criterion is realized median adaptive gain over global on inner
  held-out datasets, not regression RMSE;
- outer dataset is never used for model/hyperparameter selection.

This is a selector benchmark only. Gate/Context Trust are integrated later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


CORE = [
    "replogle_k562",
    "replogle_rpe1",
    "nadig25hepg2",
    "nadig25jurkat",
]

NON_FEATURE = {
    "dataset","seed","perturbation","n_cells_total",
    "oracle_regret_pearson","global_pearson","oracle_pearson",
    "target_regret_gt_0_01","target_regret_gt_0_02","appearance_weight",
}

def feature_cols(df):
    return [
        c for c in df.columns
        if c not in NON_FEATURE
        and not any(x in c.lower() for x in ["oracle","regret","target"])
    ]

def model_specs(seed):
    return {
        "extra": lambda: Pipeline([
            ("imp",SimpleImputer(strategy="median")),
            ("m",ExtraTreesRegressor(
                n_estimators=400,
                min_samples_leaf=8,
                max_features="sqrt",
                n_jobs=-1,
                random_state=seed,
            )),
        ]),
        "rf": lambda: Pipeline([
            ("imp",SimpleImputer(strategy="median")),
            ("m",RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=8,
                max_features="sqrt",
                n_jobs=-1,
                random_state=seed,
            )),
        ]),
        "hgb": lambda: Pipeline([
            ("imp",SimpleImputer(strategy="median")),
            ("m",HistGradientBoostingRegressor(
                learning_rate=.05,
                max_iter=300,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=seed,
            )),
        ]),
    }

def load_long_table(features_path: Path, landscape_root: Path, regret_root: Path):
    feat=pd.read_csv(features_path)
    feat=feat[feat["dataset"].isin(CORE)].copy()
    base_features=feature_cols(feat)

    rows=[]
    for ds in CORE:
        fds=feat[feat.dataset==ds].copy()
        for seed in sorted(fds.seed.unique()):
            fseed=fds[fds.seed==seed].copy()
            lp=landscape_root/ds/f"seed_{int(seed)}"/"correction_landscape.csv"
            rp=regret_root/ds/"per_perturbation_regret.csv"
            if not lp.exists() or not rp.exists():
                continue

            land=pd.read_csv(lp)
            reg=pd.read_csv(rp)
            reg=reg[reg.seed==seed][[
                "seed","perturbation","global_pearson",
                "global_pca_components","global_alpha",
                "oracle_pearson","oracle_regret_pearson",
            ]]

            x=(
                land.merge(fseed,on=["seed","perturbation"],how="inner")
                    .merge(reg,on=["seed","perturbation"],how="inner")
            )
            x["delta_vs_global"]=x["pearson"]-x["global_pearson"]
            x["candidate_alpha"]=x["alpha"].astype(float)
            x["candidate_pca_components"]=x["pca_components"].astype(float)
            x["candidate_log2_k"]=np.where(
                x["pca_components"]>0,
                np.log2(x["pca_components"].clip(lower=1)),
                0.0,
            )
            x["candidate_is_identity"]=(x["alpha"]==0).astype(int)
            rows.append(x)

    if not rows:
        raise SystemExit("No landscape rows could be assembled.")

    out=pd.concat(rows,ignore_index=True)
    predictors=base_features+[
        "candidate_alpha",
        "candidate_pca_components",
        "candidate_log2_k",
        "candidate_is_identity",
    ]
    return out,predictors

def prep_x(df, predictors):
    x=df[predictors].copy()
    for c in predictors:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    return x

def training_weights(df):
    # equalize biological perturbation contribution despite 36 expert rows and repeats
    key=df[["dataset","perturbation"]].astype(str).agg("||".join,axis=1)
    counts=key.value_counts()
    w=key.map(lambda k:1.0/counts[k]).to_numpy(float)

    # equalize dataset contribution
    dcounts=df.groupby("dataset")["perturbation"].nunique()
    w*=df["dataset"].map(lambda d:1.0/dcounts[d]).to_numpy(float)
    return w*len(w)/w.sum()

def fit_model(model, x, y, w):
    model.fit(x,y,m__sample_weight=w)
    return model

def choose_config_per_perturbation(df, pred):
    q=df[[
        "dataset","seed","perturbation",
        "pca_components","alpha","pearson","rmse",
        "gain_over_identity","global_pearson",
        "global_pca_components","global_alpha",
        "oracle_pearson","oracle_regret_pearson",
    ]].copy()
    q["predicted_delta_vs_global"]=pred
    q=q.sort_values(
        ["dataset","seed","perturbation","predicted_delta_vs_global","alpha","pca_components"],
        ascending=[True,True,True,False,True,True],
    )
    chosen=q.groupby(["dataset","seed","perturbation"],as_index=False).first()
    chosen["realized_gain_over_global"]=chosen["pearson"]-chosen["global_pearson"]
    chosen["remaining_oracle_regret"]=chosen["oracle_pearson"]-chosen["pearson"]
    chosen["beats_global"]=chosen["realized_gain_over_global"]>0
    return chosen

def summarize(chosen):
    return {
        "n":int(len(chosen)),
        "median_gain_over_global":float(chosen["realized_gain_over_global"].median()),
        "mean_gain_over_global":float(chosen["realized_gain_over_global"].mean()),
        "q10_gain_over_global":float(chosen["realized_gain_over_global"].quantile(.10)),
        "q90_gain_over_global":float(chosen["realized_gain_over_global"].quantile(.90)),
        "fraction_beats_global":float(chosen["beats_global"].mean()),
        "median_remaining_oracle_regret":float(chosen["remaining_oracle_regret"].median()),
        "median_oracle_regret":float(chosen["oracle_regret_pearson"].median()),
    }

def inner_select(train_long,predictors,seed):
    model_rows=[]
    for name,make in model_specs(seed).items():
        fold_rows=[]
        for held in sorted(train_long.dataset.unique()):
            tr=train_long[train_long.dataset!=held].copy()
            va=train_long[train_long.dataset==held].copy()

            model=make()
            fit_model(
                model,
                prep_x(tr,predictors),
                tr["delta_vs_global"].to_numpy(float),
                training_weights(tr),
            )
            pred=model.predict(prep_x(va,predictors))
            chosen=choose_config_per_perturbation(va,pred)
            s=summarize(chosen)
            s["held_out_inner_dataset"]=held
            fold_rows.append(s)

        fm=pd.DataFrame(fold_rows)
        model_rows.append({
            "model":name,
            "median_inner_gain_over_global":float(fm["median_gain_over_global"].median()),
            "mean_inner_gain_over_global":float(fm["mean_gain_over_global"].mean()),
            "median_inner_fraction_beats_global":float(fm["fraction_beats_global"].median()),
            "worst_inner_median_gain":float(fm["median_gain_over_global"].min()),
            "median_remaining_oracle_regret":float(fm["median_remaining_oracle_regret"].median()),
        })

    tab=pd.DataFrame(model_rows).sort_values(
        [
            "median_inner_gain_over_global",
            "median_inner_fraction_beats_global",
            "worst_inner_median_gain",
        ],
        ascending=[False,False,False],
    ).reset_index(drop=True)
    return str(tab.iloc[0].model),tab

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--features",
        type=Path,
        default=Path("results/split_safe_gate_features/split_safe_gate_table.csv"),
    )
    ap.add_argument(
        "--landscape-root",
        type=Path,
        default=Path("results/correction_landscapes"),
    )
    ap.add_argument(
        "--regret-root",
        type=Path,
        default=Path("results/oracle_regret"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/correction_selector_v1"),
    )
    ap.add_argument("--seed",type=int,default=20261026)
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)

    long,predictors=load_long_table(
        args.features,args.landscape_root,args.regret_root
    )
    long.to_csv(args.output_dir/"selector_training_long.csv",index=False)

    outer_rows=[]
    pred_rows=[]
    selection_rows=[]

    for held in CORE:
        tr=long[long.dataset!=held].copy()
        te=long[long.dataset==held].copy()

        best,sel=inner_select(tr,predictors,args.seed)
        sel.insert(0,"outer_held_out_dataset",held)
        selection_rows.append(sel)

        model=model_specs(args.seed)[best]()
        fit_model(
            model,
            prep_x(tr,predictors),
            tr["delta_vs_global"].to_numpy(float),
            training_weights(tr),
        )
        p=model.predict(prep_x(te,predictors))
        chosen=choose_config_per_perturbation(te,p)
        chosen["outer_held_out_dataset"]=held
        chosen["selected_model"]=best
        pred_rows.append(chosen)

        s=summarize(chosen)
        s["held_out_dataset"]=held
        s["selected_model"]=best
        outer_rows.append(s)

    outer=pd.DataFrame(outer_rows)
    preds=pd.concat(pred_rows,ignore_index=True)
    sels=pd.concat(selection_rows,ignore_index=True)

    outer.to_csv(args.output_dir/"outer_metrics.csv",index=False)
    preds.to_csv(args.output_dir/"outer_selected_configs.csv",index=False)
    sels.to_csv(args.output_dir/"inner_model_selection.csv",index=False)

    summary={
        "task":"predict utility of each correction expert relative to validation-locked global shrinkage",
        "core_datasets":CORE,
        "n_predictors":len(predictors),
        "n_candidate_rows":int(len(long)),
        "n_unique_dataset_seed_perturbations":int(
            long[["dataset","seed","perturbation"]].drop_duplicates().shape[0]
        ),
        "median_outer_gain_over_global":float(outer["median_gain_over_global"].median()),
        "mean_outer_gain_over_global":float(outer["mean_gain_over_global"].mean()),
        "median_fraction_beats_global":float(outer["fraction_beats_global"].median()),
        "median_remaining_oracle_regret":float(outer["median_remaining_oracle_regret"].median()),
        "selection_protocol":"nested leave-one-dataset-out; regressor selected by realized adaptive gain on inner held-out datasets",
        "feature_provenance":"half-A only plus candidate expert descriptors",
        "uses_halfB_as_predictor":False,
        "uses_oracle_as_predictor":False,
        "note":"This evaluates the selector alone; Context Trust and Gate are integrated in the next end-to-end stage.",
    }
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2))

    print("\n=== CORRECTION SELECTOR V1 SUMMARY ===")
    print(json.dumps(summary,indent=2))
    print("\n=== OUTER METRICS ===")
    print(outer.to_string(index=False))

if __name__=="__main__":
    main()
