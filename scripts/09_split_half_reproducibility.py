#!/usr/bin/env python3
"""Estimate intrinsic perturbation-response reproducibility by split halves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from perturbatlas.reproducibility import run_metadata, save_metadata
from perturbatlas.reproducibility_biology import (
    split_half_reproducibility,
    summarize_split_half,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/reproducibility/replogle_k562"),
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reps = split_half_reproducibility(
        args.raw,
        repeats=args.repeats,
        seed=args.seed,
    )
    summary = summarize_split_half(reps)

    reps.to_csv(args.output_dir / "split_half_replicates.csv", index=False)
    summary.to_csv(args.output_dir / "split_half_summary.csv")

    global_summary = {
        "n_perturbations": int(summary.shape[0]),
        "repeats": int(args.repeats),
        "median_pearson_across_perturbations": float(summary["pearson_median"].median()),
        "median_cosine_across_perturbations": float(summary["cosine_median"].median()),
        "median_nrmse_across_perturbations": float(summary["nrmse_median"].median()),
        "spearman_ncells_vs_pearson": float(
            summary["n_cells"].corr(summary["pearson_median"], method="spearman")
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(global_summary, indent=2), encoding="utf-8"
    )
    save_metadata(
        run_metadata({"raw": str(args.raw), "repeats": args.repeats, "seed": args.seed}),
        args.output_dir / "run_metadata.json",
    )

    print("\n=== SPLIT-HALF REPRODUCIBILITY ===")
    for k, v in global_summary.items():
        print(f"{k}: {v}")

    print("\nMost reproducible perturbations:")
    print(summary.sort_values("pearson_median", ascending=False).head(20))

    print("\nLeast reproducible perturbations:")
    print(summary.sort_values("pearson_median", ascending=True).head(20))


if __name__ == "__main__":
    main()
