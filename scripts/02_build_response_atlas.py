#!/usr/bin/env python3
"""Build a perturbation-level response atlas relative to control cells."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from perturbatlas.atlas import build_response_atlas
from perturbatlas.reproducibility import run_metadata, save_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/atlas/replogle_k562_response_atlas.h5ad"),
    )
    parser.add_argument("--perturbation-col", default="perturbation")
    parser.add_argument("--control-label", default="control")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    atlas, summary = build_response_atlas(
        args.input,
        perturbation_col=args.perturbation_col,
        control_label=args.control_label,
    )
    atlas.write_h5ad(args.output, compression="gzip")

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    metadata = run_metadata(
        {
            "input": str(args.input),
            "output": str(args.output),
            "perturbation_col": args.perturbation_col,
            "control_label": args.control_label,
        }
    )
    save_metadata(metadata, args.output.with_suffix(".run_metadata.json"))

    print("\n=== RESPONSE ATLAS ===")
    for key, value in asdict(summary).items():
        print(f"{key}: {value}")

    print("\nTop perturbations by L2 effect:")
    print(atlas.obs.sort_values("l2_effect", ascending=False).head(20))

    print("\nSaved:")
    print(args.output)
    print(summary_path)
    print(args.output.with_suffix(".run_metadata.json"))


if __name__ == "__main__":
    main()
