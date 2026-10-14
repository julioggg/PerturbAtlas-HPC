"""Association diagnostics for perturbation reliability metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def _partial_spearman(x, y, controls) -> float:
    """Partial Spearman correlation via residualized ranks."""
    xr = rankdata(np.asarray(x, dtype=float))
    yr = rankdata(np.asarray(y, dtype=float))
    c = np.asarray(controls, dtype=float)
    if c.ndim == 1:
        c = c[:, None]
    cr = np.column_stack([rankdata(c[:, j]) for j in range(c.shape[1])])
    design = np.column_stack([np.ones(len(xr)), cr])
    bx, *_ = np.linalg.lstsq(design, xr, rcond=None)
    by, *_ = np.linalg.lstsq(design, yr, rcond=None)
    rx = xr - design @ bx
    ry = yr - design @ by
    return float(np.corrcoef(rx, ry)[0, 1])


def reliability_associations(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Quantify raw and sample-size-adjusted association with reproducibility."""
    required = [
        "n_cells",
        "l2_effect",
        "noise_adjusted_l2",
        "l2_to_null_ratio",
        "pearson_median",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    clean = df[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    log_n = np.log10(clean["n_cells"].to_numpy(dtype=float))
    y = clean["pearson_median"].to_numpy(dtype=float)

    rows = []
    for predictor in ["l2_effect", "noise_adjusted_l2", "l2_to_null_ratio"]:
        x = clean[predictor].to_numpy(dtype=float)
        raw = float(spearmanr(x, y).statistic)
        partial_n = _partial_spearman(x, y, log_n)
        rows.append(
            {
                "predictor": predictor,
                "spearman_raw": raw,
                "partial_spearman_controlling_log_n": partial_n,
                "spearman_predictor_vs_log_n": float(spearmanr(x, log_n).statistic),
            }
        )

    assoc = pd.DataFrame(rows).set_index("predictor")

    q = pd.qcut(clean["n_cells"], q=5, duplicates="drop")
    within = []
    for bin_label, part in clean.assign(n_bin=q).groupby("n_bin", observed=True):
        within.append(
            {
                "n_bin": str(bin_label),
                "n": int(part.shape[0]),
                "n_cells_median": float(part["n_cells"].median()),
                "rho_adjusted_l2_vs_repro": float(
                    part["noise_adjusted_l2"].corr(part["pearson_median"], method="spearman")
                ),
                "rho_null_ratio_vs_repro": float(
                    part["l2_to_null_ratio"].corr(part["pearson_median"], method="spearman")
                ),
            }
        )
    within_df = pd.DataFrame(within)

    summary = {
        "n_perturbations": int(clean.shape[0]),
        "partial_rho_adjusted_l2_controlling_log_n": float(
            assoc.loc["noise_adjusted_l2", "partial_spearman_controlling_log_n"]
        ),
        "partial_rho_null_ratio_controlling_log_n": float(
            assoc.loc["l2_to_null_ratio", "partial_spearman_controlling_log_n"]
        ),
        "median_within_bin_rho_adjusted_l2_vs_repro": float(
            within_df["rho_adjusted_l2_vs_repro"].median()
        ),
        "median_within_bin_rho_null_ratio_vs_repro": float(
            within_df["rho_null_ratio_vs_repro"].median()
        ),
    }
    return assoc, within_df, summary
