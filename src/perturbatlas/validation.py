"""Biological validation utilities for perturbation response atlases."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def target_response_table(atlas_path: str | Path) -> pd.DataFrame:
    """Extract on-target response for perturbations whose target gene is measured.

    Assumes atlas observations are perturbation gene symbols and atlas variables
    are measured gene symbols. ``atlas.X`` stores perturbation minus control
    mean expression.
    """
    atlas = ad.read_h5ad(atlas_path)
    var_index = pd.Index(atlas.var_names.astype(str))
    gene_to_pos = {g: i for i, g in enumerate(var_index)}

    rows = []
    for i, perturbation in enumerate(atlas.obs_names.astype(str)):
        j = gene_to_pos.get(perturbation)
        if j is None:
            continue

        target_delta = float(atlas.X[i, j])
        control_mean = float(atlas.var.iloc[j]["control_mean"])
        perturb_mean = float(atlas.layers["mean_expression"][i, j])
        rows.append(
            {
                "perturbation": perturbation,
                "n_cells": int(atlas.obs.iloc[i]["n_cells"]),
                "target_delta": target_delta,
                "control_mean": control_mean,
                "perturbation_mean": perturb_mean,
                "target_decreased": bool(target_delta < 0),
                "target_abs_delta": abs(target_delta),
            }
        )

    out = pd.DataFrame(rows).set_index("perturbation")
    return out


def target_response_summary(table: pd.DataFrame) -> dict[str, float | int]:
    """Summarize on-target directionality across measured perturbations."""
    if table.empty:
        return {
            "n_targets_measured": 0,
            "fraction_target_decreased": float("nan"),
            "median_target_delta": float("nan"),
            "mean_target_delta": float("nan"),
        }

    return {
        "n_targets_measured": int(table.shape[0]),
        "fraction_target_decreased": float(table["target_decreased"].mean()),
        "median_target_delta": float(table["target_delta"].median()),
        "mean_target_delta": float(table["target_delta"].mean()),
    }


def ranking_diagnostics(effect_table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compare naive and sample-size-calibrated perturbation rankings."""
    df = effect_table.copy()
    df["rank_naive_l2"] = df["l2_effect"].rank(ascending=False, method="min")
    df["rank_adjusted_l2"] = df["noise_adjusted_l2"].rank(ascending=False, method="min")
    df["rank_null_ratio"] = df["l2_to_null_ratio"].rank(ascending=False, method="min")
    df["rank_shift_adjusted_minus_naive"] = df["rank_adjusted_l2"] - df["rank_naive_l2"]
    df["abs_rank_shift"] = df["rank_shift_adjusted_minus_naive"].abs()

    top100_naive = set(df.nsmallest(100, "rank_naive_l2").index)
    top100_adjusted = set(df.nsmallest(100, "rank_adjusted_l2").index)

    summary = {
        "n_perturbations": int(df.shape[0]),
        "spearman_naive_vs_adjusted": float(
            df["rank_naive_l2"].corr(df["rank_adjusted_l2"], method="spearman")
        ),
        "median_absolute_rank_shift": float(df["abs_rank_shift"].median()),
        "mean_absolute_rank_shift": float(df["abs_rank_shift"].mean()),
        "top100_overlap": int(len(top100_naive & top100_adjusted)),
        "top100_jaccard": float(
            len(top100_naive & top100_adjusted) / len(top100_naive | top100_adjusted)
        ),
    }
    return df, summary
