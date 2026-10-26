#!/usr/bin/env python3
"""Build split-specific, half-A-only features for adaptive gating.

Every predictor is computed from the observed half-A response and, where
needed, models fitted only on half-A TRAIN perturbations from the same seed.

No half-B values, oracle labels, full-data effect tables, or split-half
reliability estimates are used as predictors.

The output is a development table joined to oracle regret labels only AFTER
feature construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


DEFAULT_DATASETS = [
    "arch1",
    "wessels23",
    "replogle_k562",
    "replogle_rpe1",
    "nadig25hepg2",
    "nadig25jurkat",
    "kaden25rpe1",
]
KS = [8, 16, 32, 64, 128]


def row_cosine(a, b):
    num = np.sum(a*b, axis=1)
    den = np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    out=np.full(len(a),np.nan,float)
    m=den>0
    out[m]=num[m]/den[m]
    return out


def concentration_features(x):
    ax=np.abs(x)
    energy=x.astype(np.float64)**2
    total_abs=ax.sum(axis=1)+1e-12
    total_energy=energy.sum(axis=1)+1e-12

    out={}
    for k in [10,50,100,500]:
        kk=min(k,x.shape[1])
        top_abs=np.partition(ax, x.shape[1]-kk, axis=1)[:,-kk:].sum(axis=1)
        top_en=np.partition(energy, x.shape[1]-kk, axis=1)[:,-kk:].sum(axis=1)
        out[f"top{k}_abs_fraction"]=top_abs/total_abs
        out[f"top{k}_energy_fraction"]=top_en/total_energy
    return out


def build_seed_features(half_a: Path, seed: int, dataset: str):
    a=ad.read_h5ad(half_a)
    x=np.asarray(a.X,dtype=np.float32)
    split=a.obs["split"].astype(str).to_numpy()
    tr=split=="train"
    te=split=="test"

    names=np.asarray(a.obs_names,dtype=str)
    xtr=x[tr]
    xte=x[te]

    n_total=pd.to_numeric(a.obs["n_cells"],errors="coerce").to_numpy(float)
    # benchmark half A receives floor(total perturbation cells / 2)
    n_half=np.floor(n_total/2.0)

    feat=pd.DataFrame({
        "dataset":dataset,
        "seed":seed,
        "perturbation":names[te],
        "n_cells_total":n_total[te],
        "n_cells_halfA":n_half[te],
        "halfA_l2":np.linalg.norm(xte,axis=1),
        "halfA_mean_abs":np.mean(np.abs(xte),axis=1),
        "halfA_std":np.std(xte,axis=1),
        "halfA_max_abs":np.max(np.abs(xte),axis=1),
        "halfA_mean":np.mean(xte,axis=1),
    })

    for k,v in concentration_features(xte).items():
        feat[k]=v

    # One PCA fit at the maximum dimension, trained only on train perturbations.
    maxk=min(max(KS),xtr.shape[0]-1,xtr.shape[1])
    pca=PCA(n_components=maxk,svd_solver="randomized",random_state=0)
    ztr=pca.fit_transform(xtr)
    zte=pca.transform(xte)
    mean=pca.mean_.astype(np.float32)
    comps=pca.components_.astype(np.float32)

    centered=xte-mean
    centered_l2=np.linalg.norm(centered,axis=1)+1e-12

    for k in KS:
        kk=min(k,maxk)
        rec=zte[:,:kk]@comps[:kk] + mean
        resid=xte-rec
        feat[f"pca{int(k)}_reconstruction_l2"]=np.linalg.norm(rec,axis=1)
        feat[f"pca{int(k)}_residual_l2"]=np.linalg.norm(resid,axis=1)
        feat[f"pca{int(k)}_residual_fraction"]=(
            np.linalg.norm(resid,axis=1)/centered_l2
        )
        feat[f"pca{int(k)}_cosine"]=row_cosine(xte,rec)
        feat[f"pca{int(k)}_score_l2"]=np.linalg.norm(zte[:,:kk],axis=1)

    # Position relative to train perturbation geometry.
    train_norm=np.linalg.norm(xtr,axis=1)
    med=float(np.median(train_norm))
    q25=float(np.quantile(train_norm,.25))
    q75=float(np.quantile(train_norm,.75))
    iqr=max(q75-q25,1e-12)
    feat["train_l2_median"]=med
    feat["train_l2_iqr"]=iqr
    feat["halfA_l2_robust_z_vs_train"]=(feat["halfA_l2"]-med)/iqr

    return feat


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repeated-root",type=Path,default=Path("results/repeated_shrinkage"))
    ap.add_argument("--regret-root",type=Path,default=Path("results/oracle_regret"))
    ap.add_argument("--datasets",nargs="+",default=DEFAULT_DATASETS)
    ap.add_argument("--output-dir",type=Path,default=Path("results/split_safe_gate_features"))
    ap.add_argument("--overwrite",action="store_true")
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)

    all_rows=[]
    audit=[]

    for ds in args.datasets:
        dsout=args.output_dir/ds
        dsout.mkdir(parents=True,exist_ok=True)
        regret_path=args.regret_root/ds/"per_perturbation_regret.csv"
        if not regret_path.exists():
            audit.append({"dataset":ds,"status":"missing_regret"})
            continue
        regret=pd.read_csv(regret_path)
        regret["perturbation"]=regret["perturbation"].astype(str)

        completed=0
        for seed_dir in sorted((args.repeated_root/ds).glob("seed_*")):
            seed=int(seed_dir.name.split("_",1)[1])
            cache=dsout/f"seed_{seed}_features.csv"
            if cache.exists() and not args.overwrite:
                f=pd.read_csv(cache)
            else:
                half_a=seed_dir/"benchmark"/"half_A.h5ad"
                if not half_a.exists():
                    continue
                f=build_seed_features(half_a,seed,ds)
                f.to_csv(cache,index=False)

            lab=regret[regret["seed"]==seed][[
                "seed","perturbation","oracle_regret_pearson",
                "global_pearson","oracle_pearson"
            ]].copy()
            z=f.merge(lab,on=["seed","perturbation"],how="inner")
            z["target_regret_gt_0_01"]=(z["oracle_regret_pearson"]>.01).astype(int)
            z["target_regret_gt_0_02"]=(z["oracle_regret_pearson"]>.02).astype(int)
            all_rows.append(z)
            completed+=1

        audit.append({
            "dataset":ds,
            "status":"ok" if completed else "no_seeds",
            "n_seeds":completed,
            "n_rows":int(sum(len(x) for x in all_rows if len(x) and x["dataset"].iloc[0]==ds)),
        })

    if not all_rows:
        raise SystemExit("No split-safe rows built.")

    out=pd.concat(all_rows,ignore_index=True)

    # Equalize repeated appearances of the same biological perturbation.
    counts=out.groupby(["dataset","perturbation"])["seed"].transform("count")
    out["appearance_weight"]=1.0/counts

    out.to_csv(args.output_dir/"split_safe_gate_table.csv",index=False)
    pd.DataFrame(audit).to_csv(args.output_dir/"assembly_audit.csv",index=False)

    predictor_cols=[
        c for c in out.columns
        if c not in {
            "dataset","seed","perturbation",
            "oracle_regret_pearson","global_pearson","oracle_pearson",
            "target_regret_gt_0_01","target_regret_gt_0_02","appearance_weight"
        }
    ]
    summary={
        "n_rows":int(len(out)),
        "n_datasets":int(out["dataset"].nunique()),
        "n_unique_dataset_perturbations":int(out[["dataset","perturbation"]].drop_duplicates().shape[0]),
        "n_predictors":len(predictor_cols),
        "predictors":predictor_cols,
        "feature_provenance":"half-A only; PCA fit only on half-A train perturbations within each seed",
        "label_provenance":"oracle regret joined after feature construction; never a predictor",
        "warning":"Repeated seed rows are not independent biological replicates; appearance_weight equalizes perturbation contribution."
    }
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2))
    print("\n=== AUDIT ===")
    print(pd.DataFrame(audit).to_string(index=False))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
