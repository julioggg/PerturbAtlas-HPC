#!/usr/bin/env python3
"""Validate analytic sampling-noise correction with an empirical control null."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from perturbatlas.nulls import empirical_control_null, interpolate_empirical_thresholds
from perturbatlas.reproducibility import run_metadata, save_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("effects", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nulls/replogle_k562"),
    )
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--max-sizes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    effects = pd.read_csv(args.effects, index_col=0)
    null = empirical_control_null(
        args.raw,
        effects["n_cells"].to_numpy(),
        n_replicates=args.replicates,
        max_sizes=args.max_sizes,
        seed=args.seed,
    )
    calibrated = interpolate_empirical_thresholds(effects, null)

    null.to_csv(args.output_dir / "empirical_control_null.csv", index=False)
    calibrated.to_csv(args.output_dir / "effects_with_empirical_null.csv")

    summary = {
        "n_null_sizes": int(null.shape[0]),
        "n_replicates_per_size": int(args.replicates),
        "fraction_above_q95": float(calibrated["above_empirical_q95"].mean()),
        "fraction_above_q99": float(calibrated["above_empirical_q99"].mean()),
        "median_empirical_l2_over_q95": float(calibrated["empirical_l2_over_q95"].median()),
        "spearman_analytic_null_vs_empirical_q95": float(
            calibrated["expected_null_l2"].corr(calibrated["null_q95_l2"], method="spearman")
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_metadata(
        run_metadata(
            {
                "raw": str(args.raw),
                "effects": str(args.effects),
                "replicates": args.replicates,
                "max_sizes": args.max_sizes,
                "seed": args.seed,
            }
        ),
        args.output_dir / "run_metadata.json",
    )

    print("\n=== EMPIRICAL CONTROL NULL ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\nNull grid:")
    print(null.to_string(index=False))


if __name__ == "__main__":
    main()
