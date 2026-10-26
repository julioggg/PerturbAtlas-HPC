#!/usr/bin/env python3
"""Unsupervised context-trust / OOD diagnostic from split-safe half-A features.

Reference regime:
    core CRISPRi essentiality datasets.

For every dataset×seed context we summarize the unlabeled perturbation feature
distribution (median and IQR). We then compute a robust standardized distance
to the reference core contexts.

Evaluation is diagnostic:
- core contexts are scored leave-one-dataset-out against the remaining core
  datasets;
- shifted contexts are scored against all core datasets;
- no oracle regret, half-B, target label, or perturbation outcome is used.

The threshold is calibrated only from leave-one-core-out reference scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CORE = {
    "replogle_k562",
    "replogle_rpe1",
    "nadig25hepg2",
    "nadig25jurkat",
}
REGIME = {
    "replogle_k562": "core_crispri_essentiality",
    "replogle_rpe1": "core_crispri_essentiality",
    "nadig25hepg2": "core_crispri_essentiality",
    "nadig25jurkat": "core_crispri_essentiality",
    "arch1": "context_shift_crispri_hesc",
    "wessels23": "modality_shift_cas13_combinatorial",
    "kaden25rpe1": "modality_shift_crispra_reprogramming",
}

NON_FEATURE = {
    "dataset","seed","perturbation","n_cells_total",
    "oracle_regret_pearson","global_pearson","oracle_pearson",
    "target_regret_gt_0_01","target_regret_gt_0_02","appearance_weight",
}

def summarize_contexts(df, features):
    rows=[]
    for (ds,seed),g in df.groupby(["dataset","seed"]):
        rec={"dataset":ds,"seed":seed}
        for c in features:
            x=pd.to_numeric(g[c],errors="coerce")
            rec[f"med__{c}"]=x.median()
            rec[f"iqr__{c}"]=x.quantile(.75)-x.quantile(.25)
        rows.append(rec)
    return pd.DataFrame(rows)

def robust_scale(reference, cols):
    med=reference[cols].median(axis=0)
    q25=reference[cols].quantile(.25,axis=0)
    q75=reference[cols].quantile(.75,axis=0)
    scale=(q75-q25).replace(0,np.nan)
    # fallback MAD-like / std-like scale where IQR is zero
    alt=reference[cols].std(axis=0).replace(0,np.nan)
    scale=scale.fillna(alt).fillna(1.0)
    return med, scale

def score_rows(test, reference, cols):
    med,scale=robust_scale(reference,cols)
    z=(test[cols]-med)/scale
    # RMS robust distance so dimensionality does not trivially inflate score.
    rms=np.sqrt(np.nanmean(np.asarray(z,float)**2,axis=1))
    # nearest-neighbour robust distance to a reference context seed.
    rz=(reference[cols]-med)/scale
    a=np.asarray(z,float); b=np.asarray(rz,float)
    nn=[]
    for row in a:
        d=np.sqrt(np.nanmean((b-row[None,:])**2,axis=1))
        nn.append(float(np.nanmin(d)))
    return np.asarray(rms,float),np.asarray(nn,float)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "table",type=Path,nargs="?",
        default=Path("results/split_safe_gate_features/split_safe_gate_table.csv")
    )
    ap.add_argument(
        "--output-dir",type=Path,
        default=Path("results/context_trust_v1")
    )
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(args.table)
    features=[
        c for c in df.columns
        if c not in NON_FEATURE and not any(k in c.lower() for k in ["oracle","regret","target"])
    ]

    ctx=summarize_contexts(df,features)
    ctx["transfer_group"]=ctx["dataset"].map(REGIME)
    ctx.to_csv(args.output_dir/"context_features.csv",index=False)

    cols=[c for c in ctx.columns if c.startswith("med__") or c.startswith("iqr__")]

    scored=[]
    # Leave-one-dataset-out scoring for core contexts.
    for ds in sorted(CORE):
        te=ctx[ctx.dataset==ds].copy()
        ref=ctx[(ctx.dataset.isin(CORE)) & (ctx.dataset!=ds)].copy()
        rms,nn=score_rows(te,ref,cols)
        te["reference"]="core_leave_one_dataset_out"
        te["ood_rms"]=rms
        te["ood_nearest"]=nn
        scored.append(te[["dataset","seed","transfer_group","reference","ood_rms","ood_nearest"]])

    # Shifted contexts are compared only against the full core reference.
    core_ref=ctx[ctx.dataset.isin(CORE)].copy()
    for ds in sorted(set(ctx.dataset)-CORE):
        te=ctx[ctx.dataset==ds].copy()
        rms,nn=score_rows(te,core_ref,cols)
        te["reference"]="all_core_crispri"
        te["ood_rms"]=rms
        te["ood_nearest"]=nn
        scored.append(te[["dataset","seed","transfer_group","reference","ood_rms","ood_nearest"]])

    score=pd.concat(scored,ignore_index=True)

    # Threshold from core LODO scores only, never using shifted contexts.
    core_scores=score[score.dataset.isin(CORE)]
    thr_rms=float(core_scores["ood_rms"].quantile(.95))
    thr_nn=float(core_scores["ood_nearest"].quantile(.95))
    score["ood_flag_rms"]=score["ood_rms"]>thr_rms
    score["ood_flag_nearest"]=score["ood_nearest"]>thr_nn
    score.to_csv(args.output_dir/"context_ood_scores.csv",index=False)

    ds=score.groupby(["dataset","transfer_group"],as_index=False).agg(
        median_ood_rms=("ood_rms","median"),
        q05_ood_rms=("ood_rms",lambda x:x.quantile(.05)),
        q95_ood_rms=("ood_rms",lambda x:x.quantile(.95)),
        median_ood_nearest=("ood_nearest","median"),
        q05_ood_nearest=("ood_nearest",lambda x:x.quantile(.05)),
        q95_ood_nearest=("ood_nearest",lambda x:x.quantile(.95)),
        fraction_flagged_rms=("ood_flag_rms","mean"),
        fraction_flagged_nearest=("ood_flag_nearest","mean"),
    )
    ds["rms_threshold_core95"]=thr_rms
    ds["nearest_threshold_core95"]=thr_nn
    ds.to_csv(args.output_dir/"dataset_summary.csv",index=False)

    summary={
        "n_contexts":int(len(ctx)),
        "n_features_per_perturbation":len(features),
        "n_context_summary_features":len(cols),
        "reference_regime":"core_crispri_essentiality",
        "threshold_calibration":"95th percentile of leave-one-core-dataset-out seed scores",
        "rms_threshold":thr_rms,
        "nearest_threshold":thr_nn,
        "feature_provenance":"unlabeled split-safe half-A features only",
        "uses_halfB":False,
        "uses_oracle_labels":False,
        "interpretation":"diagnostic OOD/context-trust score; not yet a final calibrated abstention guarantee",
    }
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2))

    print("\n=== CONTEXT TRUST SUMMARY ===")
    print(ds.to_string(index=False))
    print("\n=== CONFIG ===")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
