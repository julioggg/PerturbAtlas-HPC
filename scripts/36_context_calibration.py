#!/usr/bin/env python3
"""Evaluate zero-shot ranking and few-shot context calibration of gain predictions.

Takes LODO predictions from script 35. Tests whether cross-dataset failure is
primarily rank failure or context-level calibration shift.

For each held-out dataset and model:
1) zero-shot ranking/sign diagnostics;
2) repeated few-shot calibration using a small labeled subset from the held-out
   dataset;
3) evaluate intercept-only and affine calibration on the remaining perturbations.

Few-shot calibration is explicitly NOT zero-shot generalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score


def safe_spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return float(spearmanr(x[m], y[m]).statistic)


def metrics(y, pred):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    out = {
        "n": int(len(y)),
        "spearman": safe_spearman(pred, y),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "sign_accuracy": float(np.mean((pred > 0) == (y > 0))),
        "fraction_true_positive_gain": float(np.mean(y > 0)),
        "fraction_pred_positive_gain": float(np.mean(pred > 0)),
    }
    labels = (y > 0).astype(int)
    if len(np.unique(labels)) == 2:
        out["auroc_beneficial"] = float(roc_auc_score(labels, pred))
    else:
        out["auroc_beneficial"] = np.nan
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "predictions",
        type=Path,
        nargs="?",
        default=Path("results/gain_drivers/lodo_predictions.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/context_calibration"),
    )
    p.add_argument("--fractions", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--seed", type=int, default=20261023)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions)
    need = {"dataset", "perturbation", "model", "predicted_gain", "gain_over_identity_median"}
    missing = sorted(need.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    zero_rows = []
    few_rows = []

    for (model, ds), g in df.groupby(["model", "dataset"], sort=True):
        g = g.dropna(subset=["predicted_gain", "gain_over_identity_median"]).copy()
        y = g["gain_over_identity_median"].to_numpy(float)
        pred = g["predicted_gain"].to_numpy(float)

        z = metrics(y, pred)
        zero_rows.append({"model": model, "dataset": ds, **z})

        n = len(g)
        for frac in args.fractions:
            # Ensure both calibration and evaluation sets are meaningful,
            # especially for Wessels/ARCH1.
            ncal = max(10, int(round(frac * n)))
            ncal = min(ncal, n - 10)
            if ncal < 3:
                continue

            for rep in range(args.repeats):
                rng = np.random.default_rng(
                    args.seed + rep + 1009 * sum(map(ord, ds)) + 9173 * sum(map(ord, model))
                )
                idx = rng.permutation(n)
                ci, ei = idx[:ncal], idx[ncal:]

                yc, pc = y[ci], pred[ci]
                ye, pe = y[ei], pred[ei]

                # Intercept-only robust calibration: correct the median residual.
                shift = float(np.median(yc - pc))
                p_shift = pe + shift
                m = metrics(ye, p_shift)
                few_rows.append({
                    "model": model,
                    "dataset": ds,
                    "calibration": "median_intercept",
                    "fraction_requested": frac,
                    "n_calibration": ncal,
                    "repeat": rep,
                    **m,
                })

                # Affine calibration, regularized toward a stable linear map.
                cal = Ridge(alpha=1.0)
                cal.fit(pc.reshape(-1, 1), yc)
                p_affine = cal.predict(pe.reshape(-1, 1))
                m = metrics(ye, p_affine)
                few_rows.append({
                    "model": model,
                    "dataset": ds,
                    "calibration": "ridge_affine",
                    "fraction_requested": frac,
                    "n_calibration": ncal,
                    "repeat": rep,
                    **m,
                })

    zero = pd.DataFrame(zero_rows)
    few = pd.DataFrame(few_rows)
    zero.to_csv(args.output_dir / "zero_shot_metrics.csv", index=False)
    few.to_csv(args.output_dir / "few_shot_replicates.csv", index=False)

    agg = (
        few.groupby(
            ["model", "dataset", "calibration", "fraction_requested", "n_calibration"],
            as_index=False,
        )
        .agg(
            median_spearman=("spearman", "median"),
            median_rmse=("rmse", "median"),
            median_mae=("mae", "median"),
            median_sign_accuracy=("sign_accuracy", "median"),
            median_auroc_beneficial=("auroc_beneficial", "median"),
        )
    )

    # Join zero-shot values so improvement is transparent.
    zsmall = zero[
        ["model", "dataset", "rmse", "mae", "spearman", "sign_accuracy", "auroc_beneficial"]
    ].rename(columns={
        "rmse": "zero_rmse",
        "mae": "zero_mae",
        "spearman": "zero_spearman",
        "sign_accuracy": "zero_sign_accuracy",
        "auroc_beneficial": "zero_auroc_beneficial",
    })
    agg = agg.merge(zsmall, on=["model", "dataset"], how="left")
    agg["rmse_improvement_vs_zero"] = agg["zero_rmse"] - agg["median_rmse"]
    agg["mae_improvement_vs_zero"] = agg["zero_mae"] - agg["median_mae"]
    agg.to_csv(args.output_dir / "few_shot_summary.csv", index=False)

    summary = {}
    for model, gm in agg.groupby("model"):
        summary[model] = {}
        for (cal, frac), h in gm.groupby(["calibration", "fraction_requested"]):
            key = f"{cal}_frac_{frac:g}"
            summary[model][key] = {
                "median_rmse_improvement_vs_zero": float(
                    np.nanmedian(h["rmse_improvement_vs_zero"])
                ),
                "datasets_rmse_improved": int(
                    (h["rmse_improvement_vs_zero"] > 0).sum()
                ),
                "median_mae_improvement_vs_zero": float(
                    np.nanmedian(h["mae_improvement_vs_zero"])
                ),
                "median_auroc_beneficial": float(
                    np.nanmedian(h["median_auroc_beneficial"])
                ),
            }

    summary["interpretation"] = (
        "Zero-shot Spearman measures cross-context ranking. Few-shot calibration "
        "uses labeled perturbations from the held-out context and therefore tests "
        "context adaptation, not zero-shot generalization."
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== ZERO-SHOT ===")
    print(zero.to_string(index=False))
    print("\n=== FEW-SHOT SUMMARY ===")
    print(agg.to_string(index=False))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
