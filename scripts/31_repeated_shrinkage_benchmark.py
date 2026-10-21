#!/usr/bin/env python3
"""Repeated, validation-locked shrinkage benchmarking across perturbation splits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("raw", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261016)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    py = sys.executable

    for i in range(args.repeats):
        seed = args.seed + i
        root = args.output_dir / f"seed_{seed}"
        bench = root / "benchmark"
        model = root / "shrinkage"

        run([
            py, "scripts/15_build_denoising_benchmark.py",
            str(args.raw),
            "--output-dir", str(bench),
            "--seed", str(seed),
        ])
        run([
            py, "scripts/30_validation_locked_shrinkage.py",
            str(bench / "half_A.h5ad"),
            str(bench / "half_B.h5ad"),
            "--output-dir", str(model),
        ])

        summary = json.loads((model / "summary.json").read_text())
        rows.append({
            "seed": seed,
            "selected_pca_components": summary["selected_pca_components"],
            "selected_alpha": summary["selected_alpha"],
            "validation_median_pearson": summary["validation_median_pearson"],
            "test_median_pearson": summary["test_median_pearson"],
            "identity_test_median_pearson": summary["identity_test_median_pearson"],
            "median_gain_over_identity": summary["median_gain_over_identity"],
            "test_median_rmse": summary["test_median_rmse"],
            "n_test": summary["n_test"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "repeated_split_summary.csv", index=False)

    gains = df["median_gain_over_identity"].to_numpy(float)
    agg = {
        "repeats": int(args.repeats),
        "seed_start": int(args.seed),
        "median_test_pearson": float(df["test_median_pearson"].median()),
        "median_identity_test_pearson": float(df["identity_test_median_pearson"].median()),
        "median_gain_over_identity_across_splits": float(np.median(gains)),
        "mean_gain_over_identity_across_splits": float(np.mean(gains)),
        "fraction_splits_gain_positive": float(np.mean(gains > 0)),
        "fraction_splits_alpha_zero": float(np.mean(df["selected_alpha"].to_numpy(float) == 0)),
        "selected_alpha_counts": {
            str(k): int(v) for k, v in df["selected_alpha"].value_counts().sort_index().items()
        },
        "selected_pca_counts": {
            str(int(k)): int(v) for k, v in df["selected_pca_components"].value_counts().sort_index().items()
        },
    }
    (args.output_dir / "aggregate_summary.json").write_text(
        json.dumps(agg, indent=2), encoding="utf-8"
    )

    print("\n=== REPEATED SHRINKAGE SUMMARY ===")
    print(df.to_string(index=False))
    print("\n=== AGGREGATE ===")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
