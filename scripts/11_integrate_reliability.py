#!/usr/bin/env python3
"""Integrate effect magnitude, empirical null, and split-half reproducibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from perturbatlas.reliability import build_reliability_table, quartile_labels
from perturbatlas.reproducibility import run_metadata, save_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("effects", type=Path)
    parser.add_argument("reproducibility", type=Path)
    parser.add_argument("--empirical-null", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/reliability/replogle_k562"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    effects = pd.read_csv(args.effects, index_col=0)
    repro = pd.read_csv(args.reproducibility, index_col=0)
    empirical = pd.read_csv(args.empirical_null, index_col=0) if args.empirical_null else None

    table, summary = build_reliability_table(effects, repro, empirical)
    table["median_quadrant"] = quartile_labels(table)

    table.to_csv(args.output_dir / "reliability_table.csv")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_metadata(
        run_metadata(
            {
                "effects": str(args.effects),
                "reproducibility": str(args.reproducibility),
                "empirical_null": str(args.empirical_null) if args.empirical_null else None,
            }
        ),
        args.output_dir / "run_metadata.json",
    )

    print("\n=== RELIABILITY INTEGRATION ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nMedian-based visualization quadrants:")
    print(table["median_quadrant"].value_counts())

    print("\nTop 20 by reproducibility:")
    print(
        table.sort_values("pearson_median", ascending=False)
        [["n_cells", "noise_adjusted_l2", "l2_to_null_ratio", "pearson_median", "nrmse_median"]]
        .head(20)
    )


if __name__ == "__main__":
    main()
