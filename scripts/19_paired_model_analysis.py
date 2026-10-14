#!/usr/bin/env python3
"""Paired statistical comparison of PCA and PyTorch predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from perturbatlas.evaluation import paired_model_statistics, stratified_model_performance
from perturbatlas.reproducibility import run_metadata, save_metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("table", type=Path)
    p.add_argument("--model-a", default="pca_128")
    p.add_argument("--model-b", default="pytorch")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/model_comparison/replogle_k562"),
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.table, index_col=0)
    paired, summary = paired_model_statistics(
        df,
        args.model_a,
        args.model_b,
        seed=args.seed,
        n_bootstrap=args.bootstrap,
    )
    stratified = stratified_model_performance(
        df,
        ["identity_half_A", args.model_a, args.model_b],
    )

    paired.to_csv(args.output_dir / "paired_model_differences.csv")
    stratified.to_csv(args.output_dir / "performance_by_reliability_quintile.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_metadata(run_metadata(vars(args)), args.output_dir / "run_metadata.json")

    print("\n=== PAIRED MODEL COMPARISON ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\n=== PERFORMANCE BY INTRINSIC-RELIABILITY QUINTILE ===")
    print(stratified.to_string(index=False))


if __name__ == "__main__":
    main()
