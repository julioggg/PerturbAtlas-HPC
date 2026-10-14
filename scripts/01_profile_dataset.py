#!/usr/bin/env python3
"""Profile a perturbational AnnData dataset and save compact derived metadata."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from perturbatlas.data import profile_h5ad, sample_expression_summary
from perturbatlas.reproducibility import run_metadata, save_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/profile"))
    parser.add_argument("--perturbation-col", default="perturbation")
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--sample-size", type=int, default=5000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    profile, counts = profile_h5ad(
        args.input,
        perturbation_col=args.perturbation_col,
        control_label=args.control_label,
    )
    expr = sample_expression_summary(args.input, sample_size=args.sample_size)

    counts.to_csv(args.output_dir / "perturbation_counts.csv")
    payload = {"dataset_profile": asdict(profile), "sample_expression": expr}
    (args.output_dir / "dataset_profile.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    metadata = run_metadata(
        {
            "input": str(args.input),
            "perturbation_col": args.perturbation_col,
            "control_label": args.control_label,
            "sample_size": args.sample_size,
        }
    )
    save_metadata(metadata, args.output_dir / "run_metadata.json")

    print("\n=== DATASET PROFILE ===")
    for key, value in asdict(profile).items():
        print(f"{key}: {value}")

    print("\n=== EXPRESSION SAMPLE ===")
    for key, value in expr.items():
        print(f"{key}: {value}")

    print("\nSaved:")
    print(args.output_dir / "dataset_profile.json")
    print(args.output_dir / "perturbation_counts.csv")
    print(args.output_dir / "run_metadata.json")


if __name__ == "__main__":
    main()
