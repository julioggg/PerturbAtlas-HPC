#!/usr/bin/env python3
"""Compare intrinsic reliability and calibrated effects across two contexts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from perturbatlas.cross_context import cross_context_summary, merge_context_tables


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--effects-a", type=Path, required=True)
    p.add_argument("--repro-a", type=Path, required=True)
    p.add_argument("--effects-b", type=Path, required=True)
    p.add_argument("--repro-b", type=Path, required=True)
    p.add_argument("--name-a", default="k562")
    p.add_argument("--name-b", default="rpe1")
    p.add_argument("--output-dir", type=Path, default=Path("results/cross_context/replogle_k562_vs_rpe1"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ea = pd.read_csv(args.effects_a, index_col=0)
    ra = pd.read_csv(args.repro_a, index_col=0)
    eb = pd.read_csv(args.effects_b, index_col=0)
    rb = pd.read_csv(args.repro_b, index_col=0)

    merged = merge_context_tables(
        ea, ra, eb, rb,
        suffix_a=args.name_a,
        suffix_b=args.name_b,
    )
    summary = cross_context_summary(
        merged,
        suffix_a=args.name_a,
        suffix_b=args.name_b,
    )

    merged.to_csv(args.output_dir / "shared_perturbations.csv")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=== CROSS-CONTEXT SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nTop 20 shared perturbations by mean intrinsic reliability:")
    ca = f"pearson_median_{args.name_a}"
    cb = f"pearson_median_{args.name_b}"
    ranked = merged.assign(mean_reliability=merged[[ca, cb]].mean(axis=1))
    cols = [ca, cb, "mean_reliability"]
    print(ranked.sort_values("mean_reliability", ascending=False)[cols].head(20))


if __name__ == "__main__":
    main()
