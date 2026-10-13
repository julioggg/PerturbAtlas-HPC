#!/usr/bin/env python3
"""Generate figures for on-target QC and ranking stability."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_table", type=Path)
    parser.add_argument("rank_table", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/validation/replogle_k562"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target = pd.read_csv(args.target_table, index_col=0)
    ranks = pd.read_csv(args.rank_table, index_col=0)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.hist(target["target_delta"], bins=60)
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("On-target expression change (perturbation - control)")
    ax.set_ylabel("Perturbations")
    ax.set_title("On-target transcript response")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig04_target_delta_distribution.png", dpi=300)
    fig.savefig(args.output_dir / "fig04_target_delta_distribution.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(ranks["rank_naive_l2"], ranks["rank_adjusted_l2"], s=12, alpha=0.45)
    lim = float(max(ranks["rank_naive_l2"].max(), ranks["rank_adjusted_l2"].max()))
    ax.plot([1, lim], [1, lim], linewidth=1)
    ax.set_xlabel("Naive L2 rank")
    ax.set_ylabel("Noise-adjusted L2 rank")
    ax.set_title("Rank stability after sample-size calibration")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig05_rank_stability.png", dpi=300)
    fig.savefig(args.output_dir / "fig05_rank_stability.pdf")
    plt.close(fig)

    print("Saved figures to:", args.output_dir)


if __name__ == "__main__":
    main()
