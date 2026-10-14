#!/usr/bin/env python3
"""Generate intrinsic reproducibility diagnostic figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/reproducibility/replogle_k562"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary, index_col=0)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(df["n_cells"], df["pearson_median"], s=12, alpha=0.45)
    ax.set_xscale("log")
    ax.set_xlabel("Cells per perturbation")
    ax.set_ylabel("Median split-half Pearson correlation")
    ax.set_title("Intrinsic response reproducibility versus sample size")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig07_ncells_vs_reproducibility.png", dpi=300)
    fig.savefig(args.output_dir / "fig07_ncells_vs_reproducibility.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.hist(df["pearson_median"].dropna(), bins=60)
    ax.set_xlabel("Median split-half Pearson correlation")
    ax.set_ylabel("Perturbations")
    ax.set_title("Distribution of intrinsic perturbation reproducibility")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig08_reproducibility_distribution.png", dpi=300)
    fig.savefig(args.output_dir / "fig08_reproducibility_distribution.pdf")
    plt.close(fig)

    print("Saved figures to:", args.output_dir)


if __name__ == "__main__":
    main()
