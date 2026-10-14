#!/usr/bin/env python3
"""Generate uncertainty-versus-error diagnostic figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("table", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("figures/uncertainty/replogle_k562"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.table, index_col=0)

    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    ax.scatter(df["ensemble_variance"], 1.0 - df["pearson"], s=14, alpha=0.45)
    ax.set_xlabel("Deep-ensemble predictive variance")
    ax.set_ylabel("Prediction error (1 - Pearson)")
    ax.set_title("Does predictive uncertainty identify model error?")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig14_uncertainty_vs_error.png", dpi=300)
    fig.savefig(args.output_dir / "fig14_uncertainty_vs_error.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    ax.scatter(df["intrinsic_reproducibility"], df["ensemble_variance"], s=14, alpha=0.45)
    ax.set_xlabel("Intrinsic split-half reproducibility")
    ax.set_ylabel("Deep-ensemble predictive variance")
    ax.set_title("Predictive uncertainty versus intrinsic reliability")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig15_uncertainty_vs_intrinsic_reliability.png", dpi=300)
    fig.savefig(args.output_dir / "fig15_uncertainty_vs_intrinsic_reliability.pdf")
    plt.close(fig)

    print("Saved figures to:", args.output_dir)


if __name__ == "__main__":
    main()
