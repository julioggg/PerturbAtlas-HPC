#!/usr/bin/env python3
"""Test whether effect-reproducibility coupling persists after sample-size control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from perturbatlas.association import reliability_associations
from perturbatlas.reproducibility import run_metadata, save_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/association/replogle_k562"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.table, index_col=0)
    assoc, within, summary = reliability_associations(df)

    assoc.to_csv(args.output_dir / "association_table.csv")
    within.to_csv(args.output_dir / "within_sample_size_bins.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_metadata(
        run_metadata({"table": str(args.table)}),
        args.output_dir / "run_metadata.json",
    )

    print("\n=== DECONFOUNDED RELIABILITY ASSOCIATIONS ===")
    print(assoc.to_string())
    print("\n=== WITHIN SAMPLE-SIZE QUINTILES ===")
    print(within.to_string(index=False))
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
