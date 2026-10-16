#!/usr/bin/env python3
"""Generate cross-context reliability figures for Replogle K562 vs RPE1."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


def panel(x, y, xlabel, ylabel, title, outbase: Path) -> None:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    rho, p = spearmanr(valid["x"], valid["y"])
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(valid["x"], valid["y"], s=14, alpha=0.45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(
        0.03, 0.97,
        f"n = {len(valid):,}\nSpearman ρ = {rho:.3f}\np = {p:.2e}",
        transform=ax.transAxes,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=300)
    fig.savefig(outbase.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "shared_table",
        type=Path,
        nargs="?",
        default=Path("results/cross_context/replogle_k562_vs_rpe1/shared_perturbations.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/cross_context/replogle_k562_vs_rpe1"),
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.shared_table, index_col=0)

    panel(
        df["pearson_median_k562"],
        df["pearson_median_rpe1"],
        "Intrinsic reliability in K562 (split-half Pearson)",
        "Intrinsic reliability in RPE1 (split-half Pearson)",
        "Cross-context transfer of intrinsic perturbation reliability",
        args.output_dir / "fig14_reliability_transfer",
    )
    panel(
        df["noise_adjusted_l2_k562"],
        df["noise_adjusted_l2_rpe1"],
        "Noise-adjusted effect magnitude in K562",
        "Noise-adjusted effect magnitude in RPE1",
        "Cross-context transfer of calibrated perturbation effect",
        args.output_dir / "fig15_effect_transfer",
    )
    panel(
        df["l2_to_null_ratio_k562"],
        df["l2_to_null_ratio_rpe1"],
        "L2 / expected-null ratio in K562",
        "L2 / expected-null ratio in RPE1",
        "Cross-context transfer of perturbation signal-to-noise",
        args.output_dir / "fig16_snr_transfer",
    )

    print(f"Saved cross-context figures to {args.output_dir}")


if __name__ == "__main__":
    main()
