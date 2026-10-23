#!/usr/bin/env python3
"""Checkpointed split-half reproducibility for large perturbation datasets.

This implementation defines each repeat as a global pseudo-replicate:
one control split is sampled per repeat and reused across all perturbations.
That removes redundant control-mean computation and makes each repeat internally
consistent across perturbations.

IMPORTANT: this protocol is not numerically identical to the legacy script 09,
which resampled control halves independently for every perturbation. For strict
cross-dataset comparisons, rerun all datasets with this script/protocol.

Outputs are checkpointed after every repeat so interrupted jobs can resume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def mean_rows(x, idx: np.ndarray) -> np.ndarray:
    sub = x[idx]
    if sparse.issparse(sub):
        return np.asarray(sub.mean(axis=0)).ravel().astype(np.float32)
    return np.asarray(sub, dtype=np.float32).mean(axis=0)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("perturbation", as_index=True)
        .agg(
            n_cells=("n_cells", "first"),
            pearson_median=("pearson", "median"),
            pearson_mean=("pearson", "mean"),
            cosine_median=("cosine", "median"),
            rmse_median=("rmse", "median"),
            nrmse_median=("nrmse", "median"),
        )
        .sort_index()
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("raw", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--perturbation-col", default="perturbation")
    p.add_argument("--control-label", default="control")
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "split_half_replicates.partial.csv"
    final_reps = out / "split_half_replicates.csv"

    print(f"Loading {args.raw} into memory...", flush=True)
    adata = sc.read_h5ad(args.raw)
    print(f"Loaded shape={adata.shape}", flush=True)

    labels = adata.obs[args.perturbation_col].astype(str).to_numpy()
    control_idx = np.flatnonzero(labels == args.control_label)
    perturbations = pd.Index(pd.unique(labels[labels != args.control_label]))
    perturbation_indices = {
        p: np.flatnonzero(labels == p) for p in perturbations
    }

    completed = set()
    if checkpoint.exists():
        old = pd.read_csv(checkpoint)
        if not old.empty and "repeat" in old:
            completed = set(old["repeat"].astype(int).unique())
        print(f"Found checkpoint with completed repeats: {sorted(completed)}", flush=True)

    all_frames = []
    if checkpoint.exists():
        all_frames.append(pd.read_csv(checkpoint))

    for rep in range(args.repeats):
        if rep in completed:
            print(f"repeat {rep+1}/{args.repeats}: already complete; skipping", flush=True)
            continue

        rng = np.random.default_rng(args.seed + rep)

        # One independent control partition defines this global pseudo-replicate.
        c_perm = rng.permutation(control_idx)
        c_cut = control_idx.size // 2
        c_a, c_b = c_perm[:c_cut], c_perm[c_cut:]
        control_a = mean_rows(adata.X, c_a)
        control_b = mean_rows(adata.X, c_b)

        rows = []
        for i, perturbation in enumerate(perturbations, start=1):
            p_idx = perturbation_indices[perturbation]
            n = p_idx.size
            if n < 4:
                continue

            p_perm = rng.permutation(p_idx)
            p_cut = n // 2
            p_a, p_b = p_perm[:p_cut], p_perm[p_cut:]

            delta_a = mean_rows(adata.X, p_a) - control_a
            delta_b = mean_rows(adata.X, p_b) - control_b

            diff = delta_a.astype(np.float64) - delta_b.astype(np.float64)
            rmse = float(np.sqrt(np.mean(diff**2)))
            signal_scale = float(
                0.5
                * (
                    np.sqrt(np.mean(delta_a.astype(np.float64) ** 2))
                    + np.sqrt(np.mean(delta_b.astype(np.float64) ** 2))
                )
            )
            nrmse = rmse / signal_scale if signal_scale > 0 else float("nan")

            rows.append(
                {
                    "perturbation": perturbation,
                    "repeat": rep,
                    "n_cells": int(n),
                    "pearson": pearson(delta_a, delta_b),
                    "cosine": cosine(delta_a, delta_b),
                    "rmse": rmse,
                    "nrmse": nrmse,
                }
            )

            if i % 250 == 0:
                print(
                    f"repeat {rep+1}/{args.repeats}: "
                    f"{i}/{len(perturbations)} perturbations",
                    flush=True,
                )

        rep_df = pd.DataFrame(rows)
        all_frames.append(rep_df)
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(checkpoint, index=False)
        print(
            f"repeat {rep+1}/{args.repeats}: checkpoint written "
            f"({len(rep_df)} perturbations)",
            flush=True,
        )

    reps = pd.concat(all_frames, ignore_index=True)
    reps = reps.sort_values(["repeat", "perturbation"]).reset_index(drop=True)
    reps.to_csv(final_reps, index=False)

    summary = summarize(reps)
    summary.to_csv(out / "split_half_summary.csv")

    global_summary = {
        "n_perturbations": int(summary.shape[0]),
        "repeats": int(args.repeats),
        "median_pearson_across_perturbations": float(summary["pearson_median"].median()),
        "median_cosine_across_perturbations": float(summary["cosine_median"].median()),
        "median_nrmse_across_perturbations": float(summary["nrmse_median"].median()),
        "spearman_ncells_vs_pearson": float(
            summary["n_cells"].corr(summary["pearson_median"], method="spearman")
        ),
        "protocol": "shared_control_split_per_repeat_checkpointed",
        "seed": int(args.seed),
    }
    (out / "summary.json").write_text(json.dumps(global_summary, indent=2), encoding="utf-8")

    metadata = {
        "raw": str(args.raw),
        "output_dir": str(out),
        "repeats": int(args.repeats),
        "seed": int(args.seed),
        "perturbation_col": args.perturbation_col,
        "control_label": args.control_label,
        "protocol": "shared_control_split_per_repeat_checkpointed",
        "note": (
            "One control split is shared across perturbations within each repeat. "
            "This differs from legacy script 09 and should be used consistently "
            "for cross-dataset comparisons."
        ),
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== SPLIT-HALF REPRODUCIBILITY ===", flush=True)
    for k, v in global_summary.items():
        print(f"{k}: {v}", flush=True)


if __name__ == "__main__":
    main()
