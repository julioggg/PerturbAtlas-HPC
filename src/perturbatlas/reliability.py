"""Integrate perturbation effect strength with intrinsic reproducibility."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_reliability_table(
    effects: pd.DataFrame,
    reproducibility: pd.DataFrame,
    empirical_null: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Merge effect and split-half metrics and compute continuous diagnostics."""
    df = effects.join(reproducibility, how="inner", rsuffix="_repro")
    if empirical_null is not None:
        keep = [
            c for c in [
                "null_q95_l2",
                "null_q99_l2",
                "above_empirical_q95",
                "above_empirical_q99",
                "empirical_l2_over_q95",
            ]
            if c in empirical_null.columns
        ]
        df = df.join(empirical_null[keep], how="left")

    df["reproducible_r_gt_0_5"] = df["pearson_median"] > 0.5
    df["reproducible_r_gt_0_8"] = df["pearson_median"] > 0.8

    if "above_empirical_q95" in df:
        df["strong_and_reproducible"] = (
            df["above_empirical_q95"].astype(bool)
            & df["reproducible_r_gt_0_5"]
        )

    summary = {
        "n_perturbations": int(df.shape[0]),
        "spearman_l2_vs_reproducibility": float(
            df["l2_effect"].corr(df["pearson_median"], method="spearman")
        ),
        "spearman_adjusted_l2_vs_reproducibility": float(
            df["noise_adjusted_l2"].corr(df["pearson_median"], method="spearman")
        ),
        "spearman_null_ratio_vs_reproducibility": float(
            df["l2_to_null_ratio"].corr(df["pearson_median"], method="spearman")
        ),
        "fraction_reproducible_r_gt_0_5": float(df["reproducible_r_gt_0_5"].mean()),
        "fraction_reproducible_r_gt_0_8": float(df["reproducible_r_gt_0_8"].mean()),
    }
    if "strong_and_reproducible" in df:
        summary["fraction_strong_and_reproducible"] = float(
            df["strong_and_reproducible"].mean()
        )

    return df, summary


def quartile_labels(df: pd.DataFrame) -> pd.Series:
    """Label perturbations by median effect and reproducibility for visualization."""
    effect_cut = float(df["noise_adjusted_l2"].median())
    repro_cut = float(df["pearson_median"].median())
    labels = np.where(
        df["noise_adjusted_l2"] >= effect_cut,
        np.where(df["pearson_median"] >= repro_cut, "high-effect / high-repro", "high-effect / low-repro"),
        np.where(df["pearson_median"] >= repro_cut, "low-effect / high-repro", "low-effect / low-repro"),
    )
    return pd.Series(labels, index=df.index, name="median_quadrant")
