#!/usr/bin/env python3
"""Visualize reliability coupling within sample-size strata."""

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
        default=Path("figures/association/replogle_k562"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.table, index_col=0)
    df["sample_size_quintile"] = pd.qcut(df["n_cells"], q=5, duplicates="drop")

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    for label, part in df.groupby("sample_size_quintile", observed=True):
        ax.scatter(
            part["l2_to_null_ratio"],
            part["pearson_median"],
            s=11,
            alpha=0.35,
            label=str(label),
        )
    ax.set_xlabel("Effect-to-null ratio")
    ax.set_ylabel("Median split-half Pearson correlation")
    ax.set_title("Reliability coupling within sample-size strata")
    ax.legend(title="Cells per perturbation", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig10_reliability_deconfounded.png", dpi=300)
    fig.savefig(args.output_dir / "fig10_reliability_deconfounded.pdf")
    plt.close(fig)

    print("Saved figure to:", args.output_dir)


if __name__ == "__main__":
    main()
