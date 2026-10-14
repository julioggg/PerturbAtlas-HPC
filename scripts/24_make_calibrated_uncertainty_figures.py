#!/usr/bin/env python3
"""Plot scale-invariant uncertainty diagnostics and selective prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("table", type=Path)
    p.add_argument("selective", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("figures/uncertainty_calibrated/replogle_k562"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.table, index_col=0)
    sel = pd.read_csv(args.selective)

    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    ax.scatter(df["relative_ensemble_variance"], 1.0 - df["pearson"], s=14, alpha=0.45)
    ax.set_xlabel("Relative ensemble variance")
    ax.set_ylabel("Prediction error (1 - Pearson)")
    ax.set_title("Scale-normalized uncertainty versus prediction error")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig16_relative_uncertainty_vs_error.png", dpi=300)
    fig.savefig(args.output_dir / "fig16_relative_uncertainty_vs_error.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    ax.scatter(df["pairwise_correlation_disagreement"], 1.0 - df["pearson"], s=14, alpha=0.45)
    ax.set_xlabel("Ensemble directional disagreement (1 - pairwise Pearson)")
    ax.set_ylabel("Prediction error (1 - Pearson)")
    ax.set_title("Directional ensemble disagreement versus prediction error")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig17_directional_disagreement_vs_error.png", dpi=300)
    fig.savefig(args.output_dir / "fig17_directional_disagreement_vs_error.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    for metric, part in sel.groupby("uncertainty_metric"):
        ax.plot(part["coverage"], part["median_pearson"], marker="o", label=metric)
    ax.set_xlabel("Retained coverage")
    ax.set_ylabel("Median Pearson among retained perturbations")
    ax.set_title("Selective prediction by uncertainty ranking")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig18_selective_prediction.png", dpi=300)
    fig.savefig(args.output_dir / "fig18_selective_prediction.pdf")
    plt.close(fig)

    print("Saved figures to:", args.output_dir)


if __name__ == "__main__":
    main()
