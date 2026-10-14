#!/usr/bin/env python3
"""Generate model performance versus intrinsic reliability figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("table", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("figures/models/replogle_k562"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.table, index_col=0)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(df["pearson_median"], df["pca_128"], s=14, alpha=0.45)
    ax.set_xlabel("Intrinsic split-half reproducibility")
    ax.set_ylabel("PCA-128 held-out Pearson")
    ax.set_title("PCA prediction tracks intrinsic perturbation reliability")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig11_pca_vs_intrinsic_reliability.png", dpi=300)
    fig.savefig(args.output_dir / "fig11_pca_vs_intrinsic_reliability.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(df["pearson_median"], df["pytorch"], s=14, alpha=0.45)
    ax.set_xlabel("Intrinsic split-half reproducibility")
    ax.set_ylabel("PyTorch held-out Pearson")
    ax.set_title("Neural prediction tracks intrinsic perturbation reliability")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig12_pytorch_vs_intrinsic_reliability.png", dpi=300)
    fig.savefig(args.output_dir / "fig12_pytorch_vs_intrinsic_reliability.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(df["pca_128"], df["pytorch"], s=14, alpha=0.45)
    lo = float(min(df["pca_128"].min(), df["pytorch"].min()))
    hi = float(max(df["pca_128"].max(), df["pytorch"].max()))
    ax.plot([lo, hi], [lo, hi], linewidth=1)
    ax.set_xlabel("PCA-128 held-out Pearson")
    ax.set_ylabel("PyTorch held-out Pearson")
    ax.set_title("Paired perturbation-level model performance")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig13_pca_vs_pytorch.png", dpi=300)
    fig.savefig(args.output_dir / "fig13_pca_vs_pytorch.pdf")
    plt.close(fig)

    print("Saved figures to:", args.output_dir)


if __name__ == "__main__":
    main()
