#!/usr/bin/env python3
"""Generate initial diagnostic figures for perturbation effect magnitudes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/effects/replogle_k562"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.table, index_col=0)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(df["n_cells"], df["l2_effect"], s=12, alpha=0.45)
    ax.set_xscale("log")
    ax.set_xlabel("Cells per perturbation")
    ax.set_ylabel("Naive L2 effect magnitude")
    ax.set_title("Sample size versus naive perturbation effect")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig01_ncells_vs_l2.png", dpi=300)
    fig.savefig(args.output_dir / "fig01_ncells_vs_l2.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(df["expected_null_l2"], df["l2_effect"], s=12, alpha=0.45)
    lo = min(df["expected_null_l2"].min(), df["l2_effect"].min())
    hi = max(df["expected_null_l2"].max(), df["l2_effect"].max())
    ax.plot([lo, hi], [lo, hi], linewidth=1)
    ax.set_xlabel("Expected null L2 from sampling noise")
    ax.set_ylabel("Observed naive L2")
    ax.set_title("Observed effect versus sample-size-dependent null")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig02_observed_vs_null_l2.png", dpi=300)
    fig.savefig(args.output_dir / "fig02_observed_vs_null_l2.pdf")
    plt.close(fig)

    top = df.sort_values("noise_adjusted_l2", ascending=False).head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    ax.barh(top.index, top["noise_adjusted_l2"])
    ax.set_xlabel("Noise-adjusted L2 effect")
    ax.set_ylabel("Perturbation")
    ax.set_title("Top perturbations after sample-size calibration")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig03_top_adjusted_effects.png", dpi=300)
    fig.savefig(args.output_dir / "fig03_top_adjusted_effects.pdf")
    plt.close(fig)

    print("Saved figures to:", args.output_dir)


if __name__ == "__main__":
    main()
