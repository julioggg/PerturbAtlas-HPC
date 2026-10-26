#!/usr/bin/env python3
"""Evaluate the combined Context Trust + Gate v3 stack.

This script joins:
1. split-safe Gate v3 perturbation-level predictions; and
2. Context Trust v2 dataset/seed support decisions.

The policy is:
- if a dataset/seed context passes Context Trust, keep Gate v3's decision;
- otherwise abstain from personalized correction.

Because Gate v3 outer predictions were already aggregated across seeds, this
script computes a conservative dataset-level support rate and evaluates:
- gate performance on fully/mostly supported contexts;
- coverage of contexts;
- regret capture among contexts where the gate is allowed to operate.

No model is retrained and no threshold is tuned here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def safe_auc(y,p):
    return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan

def safe_ap(y,p):
    return float(average_precision_score(y,p)) if np.sum(y)>0 else np.nan

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--gate-predictions",
        type=Path,
        default=Path("results/split_safe_gate_v3/outer_predictions_by_perturbation.csv"),
    )
    ap.add_argument(
        "--context-scores",
        type=Path,
        default=Path("results/context_trust_v2/context_trust_scores.csv"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/trust_stack_v1"),
    )
    ap.add_argument(
        "--min-context-pass-fraction",
        type=float,
        default=0.8,
        help="Allow gate if at least this fraction of dataset×seed contexts pass.",
    )
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)

    pred=pd.read_csv(args.gate_predictions)
    ctx=pd.read_csv(args.context_scores)

    # Dataset-level context support summary.
    support=(
        ctx.groupby("dataset",as_index=False)
        .agg(
            context_pass_fraction=("context_trust_pass","mean"),
            geometry_ood_fraction=("geometry_out_of_support","mean"),
            metadata_ood=("metadata_out_of_support","max"),
        )
    )
    support["allow_gate"]=support["context_pass_fraction"]>=args.min_context_pass_fraction

    pred=pred.merge(support,on="dataset",how="left")
    if pred["allow_gate"].isna().any():
        missing=sorted(pred.loc[pred["allow_gate"].isna(),"dataset"].unique())
        raise SystemExit(f"Missing context trust for datasets: {missing}")

    # Gate's own threshold decision.
    pred["gate_select"]=pred["predicted_probability"]>=pred["selected_threshold"]

    # Combined policy: personalization only if context is supported.
    pred["trust_stack_select"]=pred["allow_gate"] & pred["gate_select"]

    # If the context is unsupported, the system abstains from personalization.
    pred["abstain_context"]=~pred["allow_gate"]

    pred.to_csv(args.output_dir/"trust_stack_predictions.csv",index=False)
    support.to_csv(args.output_dir/"context_support_summary.csv",index=False)

    rows=[]
    for ds,g in pred.groupby("dataset"):
        y=g["target"].to_numpy(int)
        p=g["predicted_probability"].to_numpy(float)
        z_gate=g["gate_select"].to_numpy(bool)
        z_stack=g["trust_stack_select"].to_numpy(bool)
        r=np.clip(g["median_regret"].to_numpy(float),0,None)
        total=r.sum()

        rows.append({
            "dataset":ds,
            "n_perturbations":len(g),
            "allow_gate":bool(g["allow_gate"].iloc[0]),
            "context_pass_fraction":float(g["context_pass_fraction"].iloc[0]),
            "positive_fraction":float(y.mean()),
            "gate_auroc":safe_auc(y,p),
            "gate_auprc":safe_ap(y,p),
            "gate_selected_fraction":float(z_gate.mean()),
            "stack_selected_fraction":float(z_stack.mean()),
            "gate_regret_capture":float(r[z_gate].sum()/total) if total>0 else np.nan,
            "stack_regret_capture":float(r[z_stack].sum()/total) if total>0 else np.nan,
            "gate_balanced_accuracy":float(balanced_accuracy_score(y,z_gate)),
            "stack_balanced_accuracy":float(balanced_accuracy_score(y,z_stack)),
            "gate_f1":float(f1_score(y,z_gate,zero_division=0)),
            "stack_f1":float(f1_score(y,z_stack,zero_division=0)),
            "gate_precision":float(precision_score(y,z_gate,zero_division=0)),
            "stack_precision":float(precision_score(y,z_stack,zero_division=0)),
            "gate_recall":float(recall_score(y,z_gate,zero_division=0)),
            "stack_recall":float(recall_score(y,z_stack,zero_division=0)),
        })

    ds=pd.DataFrame(rows)
    ds.to_csv(args.output_dir/"dataset_metrics.csv",index=False)

    allowed=ds[ds["allow_gate"]]
    blocked=ds[~ds["allow_gate"]]

    summary={
        "min_context_pass_fraction":args.min_context_pass_fraction,
        "n_datasets_total":int(len(ds)),
        "n_datasets_gate_allowed":int(allowed.shape[0]),
        "n_datasets_gate_blocked":int(blocked.shape[0]),
        "gate_allowed_datasets":allowed["dataset"].tolist(),
        "gate_blocked_datasets":blocked["dataset"].tolist(),
        "median_auroc_allowed_contexts":float(allowed["gate_auroc"].median()) if len(allowed) else np.nan,
        "min_auroc_allowed_contexts":float(allowed["gate_auroc"].min()) if len(allowed) else np.nan,
        "median_regret_capture_allowed_contexts":float(allowed["gate_regret_capture"].median()) if len(allowed) else np.nan,
        "median_selected_fraction_allowed_contexts":float(allowed["gate_selected_fraction"].median()) if len(allowed) else np.nan,
        "interpretation":"Context Trust controls where Gate v3 is allowed to operate; unsupported contexts abstain from personalization.",
        "warning":"This combines development-benchmark components and is not a frozen external validation.",
    }
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2))

    print("\n=== CONTEXT SUPPORT ===")
    print(support.to_string(index=False))
    print("\n=== TRUST STACK DATASET METRICS ===")
    print(ds.to_string(index=False))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
