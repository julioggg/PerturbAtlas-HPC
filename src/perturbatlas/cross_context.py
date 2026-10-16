"""Cross-context comparison utilities for perturbation reliability."""

from __future__ import annotations

import pandas as pd


def merge_context_tables(
    effects_a: pd.DataFrame,
    repro_a: pd.DataFrame,
    effects_b: pd.DataFrame,
    repro_b: pd.DataFrame,
    *,
    suffix_a: str,
    suffix_b: str,
) -> pd.DataFrame:
    """Merge effect and reproducibility summaries for shared perturbations."""
    a = effects_a.join(
        repro_a[["pearson_median", "cosine_median", "nrmse_median"]],
        how="inner",
    ).copy()
    b = effects_b.join(
        repro_b[["pearson_median", "cosine_median", "nrmse_median"]],
        how="inner",
    ).copy()

    keep = [
        "n_cells",
        "l2_effect",
        "expected_null_l2",
        "noise_adjusted_l2",
        "l2_to_null_ratio",
        "pearson_median",
        "cosine_median",
        "nrmse_median",
    ]
    a = a[[c for c in keep if c in a.columns]]
    b = b[[c for c in keep if c in b.columns]]

    merged = a.add_suffix(f"_{suffix_a}").join(
        b.add_suffix(f"_{suffix_b}"),
        how="inner",
    )
    merged.index.name = "perturbation"
    return merged.sort_index()


def cross_context_summary(
    merged: pd.DataFrame,
    *,
    suffix_a: str,
    suffix_b: str,
) -> dict[str, float | int]:
    """Summarize transfer of effect magnitude and intrinsic reproducibility."""
    out: dict[str, float | int] = {"n_shared_perturbations": int(len(merged))}
    pairs = {
        "pearson_reliability": (
            f"pearson_median_{suffix_a}",
            f"pearson_median_{suffix_b}",
        ),
        "noise_adjusted_l2": (
            f"noise_adjusted_l2_{suffix_a}",
            f"noise_adjusted_l2_{suffix_b}",
        ),
        "l2_to_null_ratio": (
            f"l2_to_null_ratio_{suffix_a}",
            f"l2_to_null_ratio_{suffix_b}",
        ),
        "n_cells": (
            f"n_cells_{suffix_a}",
            f"n_cells_{suffix_b}",
        ),
    }
    for name, (ca, cb) in pairs.items():
        if ca in merged.columns and cb in merged.columns:
            valid = merged[[ca, cb]].dropna()
            out[f"spearman_{name}"] = float(valid[ca].corr(valid[cb], method="spearman"))
    return out
