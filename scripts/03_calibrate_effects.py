#!/usr/bin/env python3
"""Calibrate perturbation effect magnitudes for unequal cell counts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from perturbatlas.effects import calibrate_effects
from perturbatlas.reproducibility import run_metadata, save_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("atlas", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/effects/replogle_k562"),
    )
    parser.add_argument("--perturbation-col", default="perturbation")
    parser.add_argument("--control-label", default="control")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    table, summary = calibrate_effects(
        args.raw,
        args.atlas,
        perturbation_col=args.perturbation_col,
        control_label=args.control_label,
    )

    table.to_csv(args.output_dir / "effect_calibration.csv")
    (args.output_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )
    save_metadata(
        run_metadata(
            {
                "raw": str(args.raw),
                "atlas": str(args.atlas),
                "perturbation_col": args.perturbation_col,
                "control_label": args.control_label,
            }
        ),
        args.output_dir / "run_metadata.json",
    )

    print("\n=== SAMPLE-SIZE CONFOUNDING ===")
    print(f"Spearman(n_cells, L2): {summary.spearman_ncells_l2:.6f}")
    print(
        "Spearman(n_cells, mean_abs_effect): "
        f"{summary.spearman_ncells_mean_abs:.6f}"
    )

    print("\nTop 20 by naive L2:")
    print(
        table.sort_values("l2_effect", ascending=False)
        [["n_cells", "l2_effect", "expected_null_l2", "noise_adjusted_l2", "l2_to_null_ratio"]]
        .head(20)
    )

    print("\nTop 20 by noise-adjusted L2:")
    print(
        table.sort_values("noise_adjusted_l2", ascending=False)
        [["n_cells", "l2_effect", "expected_null_l2", "noise_adjusted_l2", "l2_to_null_ratio"]]
        .head(20)
    )

    print("\nTop 20 by L2/null ratio:")
    print(
        table.sort_values("l2_to_null_ratio", ascending=False)
        [["n_cells", "l2_effect", "expected_null_l2", "noise_adjusted_l2", "l2_to_null_ratio"]]
        .head(20)
    )


if __name__ == "__main__":
    main()
