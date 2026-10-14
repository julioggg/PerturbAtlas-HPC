#!/usr/bin/env python3
"""Plot the perturbation effect-reproducibility landscape."""

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
        default=Path("figures/reliability/replogle_k562"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.table, index_col=0)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.scatter(df["noise_adjusted_l2"], df["pearson_median"], s=12, alpha=0.45)
    ax.axvline(df["noise_adjusted_l2"].median(), linewidth=1)
    ax.axhline(df["pearson_median"].median(), linewidth=1)
    ax.set_xlabel("Noise-adjusted perturbation effect (L2)")
    ax.set_ylabel("Median split-half Pearson correlation")
    ax.set_title("Perturbation effect-reproducibility landscape")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig09_effect_vs_reproducibility.png", dpi=300)
    fig.savefig(args.output_dir / "fig09_effect_vs_reproducibility.pdf")
    plt.close(fig)

    print("Saved figure to:", args.output_dir)


if __name__ == "__main__":
    main()
