"""Statistical evaluation of predictive models against intrinsic reliability."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired_model_statistics(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    reliability_col: str = "pearson_median",
    seed: int = 20261013,
    n_bootstrap: int = 10000,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Compare two models on matched perturbations with bootstrap uncertainty.

    Positive differences mean model_a outperforms model_b. Bootstrap intervals
    are computed for the median paired Pearson difference. Reliability-normalized
    summaries are restricted to perturbations with intrinsic split-half Pearson
    > 0.5 to avoid unstable ratios around zero.
    """
    cols = [model_a, model_b, reliability_col]
    clean = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    clean["paired_difference"] = clean[model_a] - clean[model_b]

    d = clean["paired_difference"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    n = len(d)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.median(d[idx])

    try:
        w = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        wilcoxon_stat = float(w.statistic)
        wilcoxon_p = float(w.pvalue)
    except ValueError:
        wilcoxon_stat = float("nan")
        wilcoxon_p = float("nan")

    reliable = clean[reliability_col] > 0.5
    high = clean[reliability_col] > 0.8
    rel = clean.loc[reliable].copy()
    if not rel.empty:
        rel[f"{model_a}_fraction_of_intrinsic"] = rel[model_a] / rel[reliability_col]
        rel[f"{model_b}_fraction_of_intrinsic"] = rel[model_b] / rel[reliability_col]

    summary = {
        "n_matched": int(n),
        "model_a": model_a,
        "model_b": model_b,
        "median_paired_difference": float(np.median(d)),
        "mean_paired_difference": float(np.mean(d)),
        "bootstrap_median_difference_ci95_low": float(np.quantile(boot, 0.025)),
        "bootstrap_median_difference_ci95_high": float(np.quantile(boot, 0.975)),
        "fraction_model_a_wins": float(np.mean(d > 0)),
        "fraction_model_b_wins": float(np.mean(d < 0)),
        "fraction_ties": float(np.mean(d == 0)),
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_pvalue": wilcoxon_p,
        "n_intrinsic_r_gt_0_5": int(reliable.sum()),
        "median_model_a_fraction_of_intrinsic_r_gt_0_5": float(
            rel[f"{model_a}_fraction_of_intrinsic"].median()
        ) if not rel.empty else float("nan"),
        "median_model_b_fraction_of_intrinsic_r_gt_0_5": float(
            rel[f"{model_b}_fraction_of_intrinsic"].median()
        ) if not rel.empty else float("nan"),
        "n_intrinsic_r_gt_0_8": int(high.sum()),
    }
    return clean, summary


def stratified_model_performance(
    df: pd.DataFrame,
    model_cols: list[str],
    reliability_col: str = "pearson_median",
    q: int = 5,
) -> pd.DataFrame:
    """Summarize model Pearson performance across intrinsic-reliability bins."""
    work = df[[reliability_col] + model_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work["reliability_bin"] = pd.qcut(work[reliability_col], q=q, duplicates="drop")
    rows = []
    for label, part in work.groupby("reliability_bin", observed=True):
        row = {
            "reliability_bin": str(label),
            "n": int(part.shape[0]),
            "intrinsic_reliability_median": float(part[reliability_col].median()),
        }
        for col in model_cols:
            row[f"{col}_median"] = float(part[col].median())
            row[f"{col}_mean"] = float(part[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)
