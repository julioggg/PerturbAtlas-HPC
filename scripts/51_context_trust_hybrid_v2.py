#!/usr/bin/env python3
"""Hybrid Context Trust v2: geometry + declared perturbation modality.

Rationale
---------
Unlabeled half-A geometry detects some context shifts (e.g. ARCH1) but can miss
semantically distinct perturbation modalities whose aggregate response
distributions overlap the reference regime (e.g. CRISPRa Kaden).

This module combines:
1. geometric OOD scores from Context Trust v1; and
2. pre-declared experimental metadata from the dataset registry.

No performance labels, regret, half-B values, or oracle quantities are used.

Reference regime:
    core CRISPRi essentiality datasets.

A context is considered outside the supported regime when:
- its perturbation modality is unseen in the reference regime; OR
- its perturbation scope is unseen in the reference regime; OR
- its geometric score exceeds the core-calibrated threshold.

The metadata component is a domain-support check, not a learned classifier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

ID_MAP = {
    "replogle_k562": "replogle22k562",
    "replogle_rpe1": "replogle22rpe1",
    "nadig25hepg2": "nadig25hepg2",
    "nadig25jurkat": "nadig25jurkat",
    "arch1": "arch1",
    "wessels23": "wessels23",
    "kaden25rpe1": "kaden25rpe1",
}

CORE = {
    "replogle_k562",
    "replogle_rpe1",
    "nadig25hepg2",
    "nadig25jurkat",
}


def load_registry(path: Path) -> dict[str, dict]:
    obj=yaml.safe_load(path.read_text())
    return {str(x["id"]):x for x in obj["datasets"]}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "ood_scores", type=Path, nargs="?",
        default=Path("results/context_trust_v1/context_ood_scores.csv")
    )
    ap.add_argument(
        "--registry", type=Path,
        default=Path("configs/datasets/scperteval_registry.yaml")
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=Path("results/context_trust_v2")
    )
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)

    score=pd.read_csv(args.ood_scores)
    reg=load_registry(args.registry)

    meta=[]
    for ds in sorted(score["dataset"].unique()):
        rid=ID_MAP.get(ds)
        if rid is None or rid not in reg:
            raise SystemExit(f"No registry mapping for {ds}")
        r=reg[rid]
        meta.append({
            "dataset":ds,
            "registry_id":rid,
            "perturbation_class":r.get("perturbation_class","unknown"),
            "perturbation_scope":r.get("perturbation_scope","unknown"),
            "screen_objective":r.get("screen_objective","unknown"),
            "cell_context":r.get("cell_context","unknown"),
            "transfer_group_registry":r.get("transfer_group","unknown"),
        })
    meta=pd.DataFrame(meta)

    core_meta=meta[meta.dataset.isin(CORE)]
    supported_classes=set(core_meta["perturbation_class"])
    supported_scopes=set(core_meta["perturbation_scope"])

    meta["novel_perturbation_class"]=~meta["perturbation_class"].isin(supported_classes)
    meta["novel_perturbation_scope"]=~meta["perturbation_scope"].isin(supported_scopes)
    meta["metadata_out_of_support"]=(
        meta["novel_perturbation_class"] | meta["novel_perturbation_scope"]
    )

    out=score.merge(meta,on="dataset",how="left")

    # Geometry OOD if either calibrated geometric detector rejects the context.
    out["geometry_out_of_support"]=(
        out["ood_flag_rms"].astype(bool) | out["ood_flag_nearest"].astype(bool)
    )

    # Hybrid abstention policy.
    out["context_out_of_support"]=(
        out["metadata_out_of_support"].astype(bool)
        | out["geometry_out_of_support"].astype(bool)
    )
    out["context_trust_pass"]=~out["context_out_of_support"]

    out.to_csv(args.output_dir/"context_trust_scores.csv",index=False)

    ds=(
        out.groupby([
            "dataset","perturbation_class","perturbation_scope",
            "screen_objective","metadata_out_of_support"
        ],as_index=False)
        .agg(
            median_ood_rms=("ood_rms","median"),
            median_ood_nearest=("ood_nearest","median"),
            fraction_geometry_ood=("geometry_out_of_support","mean"),
            fraction_context_out_of_support=("context_out_of_support","mean"),
            fraction_context_trust_pass=("context_trust_pass","mean"),
        )
    )
    ds.to_csv(args.output_dir/"dataset_summary.csv",index=False)

    summary={
        "reference_datasets":sorted(CORE),
        "supported_perturbation_classes":sorted(supported_classes),
        "supported_perturbation_scopes":sorted(supported_scopes),
        "metadata_rule":"flag unseen perturbation_class OR unseen perturbation_scope",
        "geometry_rule":"flag RMS OR nearest-neighbour OOD from Context Trust v1",
        "hybrid_rule":"abstain if metadata_out_of_support OR geometry_out_of_support",
        "uses_halfB":False,
        "uses_performance_labels":False,
        "uses_oracle_labels":False,
        "interpretation":(
            "Domain-support guardrail for deciding whether the in-regime adaptive "
            "gate may be applied. It is not evidence of performance on unseen modalities."
        ),
    }
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2))

    print("\n=== HYBRID CONTEXT TRUST V2 ===")
    print(ds.to_string(index=False))
    print("\n=== CONFIG ===")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
