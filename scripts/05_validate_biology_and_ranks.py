#!/usr/bin/env python3
"""Validate biological directionality and quantify ranking changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from perturbatlas.reproducibility import run_metadata, save_metadata
from perturbatlas.validation import (
    ranking_diagnostics,
    target_response_summary,
    target_response_table,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("atlas", type=Path)
    parser.add_argument("effects", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/validation/replogle_k562"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_table = target_response_table(args.atlas)
    target_summary = target_response_summary(target_table)
    target_table.to_csv(args.output_dir / "target_response_qc.csv")

    effects = pd.read_csv(args.effects, index_col=0)
    rank_table, rank_summary = ranking_diagnostics(effects)
    rank_table.to_csv(args.output_dir / "rank_diagnostics.csv")

    payload = {
        "target_response": target_summary,
        "ranking": rank_summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    save_metadata(
        run_metadata({"atlas": str(args.atlas), "effects": str(args.effects)}),
        args.output_dir / "run_metadata.json",
    )

    print("\n=== ON-TARGET BIOLOGICAL QC ===")
    for k, v in target_summary.items():
        print(f"{k}: {v}")

    print("\n=== RANKING DIAGNOSTICS ===")
    for k, v in rank_summary.items():
        print(f"{k}: {v}")

    print("\nLargest downward rank shifts after calibration:")
    print(
        rank_table.sort_values("rank_shift_adjusted_minus_naive", ascending=False)
        [["n_cells", "l2_effect", "noise_adjusted_l2", "rank_naive_l2", "rank_adjusted_l2", "rank_shift_adjusted_minus_naive"]]
        .head(15)
    )

    print("\nLargest upward rank shifts after calibration:")
    print(
        rank_table.sort_values("rank_shift_adjusted_minus_naive", ascending=True)
        [["n_cells", "l2_effect", "noise_adjusted_l2", "rank_naive_l2", "rank_adjusted_l2", "rank_shift_adjusted_minus_naive"]]
        .head(15)
    )


if __name__ == "__main__":
    main()
