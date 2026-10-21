#!/usr/bin/env python3
"""Aggregate repeated validation-locked shrinkage results across datasets.

Split-level tests quantify robustness across repeated perturbation partitions.
They are not biological-replicate inference because splits from the same dataset
reuse observations and are statistically dependent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def summarize_dataset(path: Path) -> dict:
    df = pd.read_csv(path)
    g = df["median_gain_over_identity"].to_numpy(float)
    n = len(g)
    pos = int((g > 0).sum())

    try:
        w = wilcoxon(g, alternative="two-sided", zero_method="wilcox")
        wp = float(w.pvalue)
        ws = float(w.statistic)
    except ValueError:
        wp = float("nan")
        ws = float("nan")

    sign_p = float(binomtest(pos, n=n, p=0.5, alternative="two-sided").pvalue)

    return {
        "n_splits": int(n),
        "median_gain": float(np.median(g)),
        "mean_gain": float(np.mean(g)),
        "q25_gain": float(np.quantile(g, 0.25)),
        "q75_gain": float(np.quantile(g, 0.75)),
        "min_gain": float(np.min(g)),
        "max_gain": float(np.max(g)),
        "positive_splits": pos,
        "fraction_positive": float(pos / n),
        "exact_sign_test_p": sign_p,
        "wilcoxon_statistic": ws,
        "wilcoxon_p": wp,
        "modal_alpha": float(df["selected_alpha"].mode().iloc[0]),
        "alpha_values": ",".join(map(str, sorted(df["selected_alpha"].unique()))),
        "median_test_pearson": float(df["test_median_pearson"].median()),
        "median_identity_pearson": float(df["identity_test_median_pearson"].median()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path("results/repeated_shrinkage"),
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "arch1",
            "replogle_k562",
            "replogle_rpe1",
            "nadig25hepg2",
            "nadig25jurkat",
        ],
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/meta_shrinkage"),
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_split_rows = []

    for ds in args.datasets:
        path = args.root / ds / "repeated_split_summary.csv"
        if not path.exists():
            print(f"SKIP {ds}: missing {path}")
            continue

        s = summarize_dataset(path)
        rows.append({"dataset": ds, **s})

        x = pd.read_csv(path)
        x.insert(0, "dataset", ds)
        all_split_rows.append(x)

    if not rows:
        raise SystemExit("No repeated_split_summary.csv files found.")

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "dataset_level_summary.csv", index=False)

    splits = pd.concat(all_split_rows, ignore_index=True)
    splits.to_csv(args.output_dir / "all_split_results.csv", index=False)

    core = summary[summary["dataset"].isin([
        "replogle_k562", "replogle_rpe1", "nadig25hepg2", "nadig25jurkat"
    ])]
    aggregate = {
        "n_datasets": int(summary.shape[0]),
        "n_core_large_datasets": int(core.shape[0]),
        "core_all_splits_positive": bool((core["fraction_positive"] == 1.0).all()),
        "core_median_of_dataset_median_gains": float(core["median_gain"].median()) if len(core) else float("nan"),
        "core_min_dataset_median_gain": float(core["median_gain"].min()) if len(core) else float("nan"),
        "core_max_dataset_median_gain": float(core["median_gain"].max()) if len(core) else float("nan"),
        "note": (
            "Repeated splits are robustness analyses, not independent biological replicates; "
            "do not pool split-level p-values as if observations were independent."
        ),
    }
    (args.output_dir / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    print("\n=== DATASET-LEVEL REPEATED SHRINKAGE ===")
    print(summary.to_string(index=False))
    print("\n=== AGGREGATE ===")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
