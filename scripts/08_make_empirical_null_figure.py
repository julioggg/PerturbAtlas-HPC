#!/usr/bin/env python3
"""Plot observed perturbation effects against empirical matched-size nulls."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("effects", type=Path)
    parser.add_argument("null", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/nulls/replogle_k562"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    effects = pd.read_csv(args.effects, index_col=0)
    null = pd.read_csv(args.null)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(effects["n_cells"], effects["l2_effect"], s=10, alpha=0.35, label="Observed perturbations")
    ax.plot(null["n_cells"], null["null_median_l2"], linewidth=1.5, label="Empirical null median")
    ax.plot(null["n_cells"], null["null_q95_l2"], linewidth=1.5, label="Empirical null 95th percentile")
    ax.plot(null["n_cells"], null["null_q99_l2"], linewidth=1.5, label="Empirical null 99th percentile")
    ax.set_xscale("log")
    ax.set_xlabel("Cells per perturbation")
    ax.set_ylabel("L2 effect magnitude")
    ax.set_title("Observed perturbation effects versus matched-size control null")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig06_empirical_null_validation.png", dpi=300)
    fig.savefig(args.output_dir / "fig06_empirical_null_validation.pdf")
    plt.close(fig)

    print("Saved figure to:", args.output_dir)


if __name__ == "__main__":
    main()
