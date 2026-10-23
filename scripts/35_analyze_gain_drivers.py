#!/usr/bin/env python3
"""Analyze perturbation-level drivers of shrinkage benefit across datasets.

Key design principles:
- outcome is repeated-split median gain over identity per perturbation
- selected alpha is NOT treated as a perturbation-specific target
- report within-dataset associations separately from pooled associations
- test cross-dataset generalization with leave-one-dataset-out (LODO)
- distinguish an oracle model using intrinsic reliability from a deployable
  feature set that does not require split-half reliability

This is an analysis script, not a causal model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_FEATURES = [
    "reliability_n_cells",
    "effect_l2",
    "effect_mean_abs",
    "expected_null_l2",
    "noise_adjusted_l2",
    "l2_to_null_ratio",
]

ORACLE_FEATURES = ["intrinsic_reliability", *BASE_FEATURES]
DEPLOYABLE_FEATURES = BASE_FEATURES


def safe_spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan, int(m.sum())
    r, p = spearmanr(np.asarray(x)[m], np.asarray(y)[m])
    return float(r), float(p), int(m.sum())


def transform_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    x = df[features].copy()
    # Stabilize heavy-tailed non-negative quantities. Keep noise_adjusted_l2
    # untransformed because it can in principle be signed.
    for c in [
        "reliability_n_cells",
        "effect_l2",
        "effect_mean_abs",
        "expected_null_l2",
        "l2_to_null_ratio",
    ]:
        if c in x:
            v = pd.to_numeric(x[c], errors="coerce")
            # log1p only where domain is valid; otherwise leave missing.
            x[c] = np.where(v >= 0, np.log1p(v), np.nan)
    return x


def make_model(features: list[str]) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    prep = ColumnTransformer(
        [("num", numeric, features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("prep", prep),
        ("ridge", Ridge(alpha=10.0)),
    ])


def lodo(df: pd.DataFrame, features: list[str], label: str):
    rows = []
    pred_rows = []

    for held in sorted(df["dataset"].unique()):
        tr = df[df["dataset"] != held].copy()
        te = df[df["dataset"] == held].copy()

        tr = tr.dropna(subset=["gain_over_identity_median"])
        te = te.dropna(subset=["gain_over_identity_median"])
        if len(tr) < 20 or len(te) < 3:
            continue

        xtr = transform_features(tr, features)
        xte = transform_features(te, features)
        ytr = tr["gain_over_identity_median"].to_numpy(float)
        yte = te["gain_over_identity_median"].to_numpy(float)

        model = make_model(features)
        model.fit(xtr, ytr)
        pred = model.predict(xte)

        rho, _, n = safe_spearman(pred, yte)
        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        mae = float(mean_absolute_error(yte, pred))

        # Baseline: predict the training-set median gain for all held-out rows.
        base = np.full_like(yte, np.nanmedian(ytr), dtype=float)
        base_rmse = float(np.sqrt(mean_squared_error(yte, base)))
        base_mae = float(mean_absolute_error(yte, base))

        rows.append({
            "model": label,
            "held_out_dataset": held,
            "n_test": int(n),
            "spearman_pred_vs_gain": rho,
            "rmse": rmse,
            "mae": mae,
            "baseline_rmse": base_rmse,
            "baseline_mae": base_mae,
            "rmse_improvement_vs_baseline": base_rmse - rmse,
            "mae_improvement_vs_baseline": base_mae - mae,
        })

        pr = te[["dataset", "perturbation", "gain_over_identity_median"]].copy()
        pr["model"] = label
        pr["predicted_gain"] = pred
        pred_rows.append(pr)

    return pd.DataFrame(rows), pd.concat(pred_rows, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "master",
        type=Path,
        nargs="?",
        default=Path("results/multidataset_master/perturbation_master_table.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gain_drivers"),
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.master)
    required = {
        "dataset", "perturbation", "gain_over_identity_median",
        "intrinsic_reliability", *BASE_FEATURES,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    d = df.dropna(subset=["gain_over_identity_median"]).copy()

    # 1) Within-dataset univariate associations.
    assoc = []
    candidate_features = [
        "intrinsic_reliability",
        "reliability_n_cells",
        "effect_l2",
        "effect_mean_abs",
        "expected_null_l2",
        "noise_adjusted_l2",
        "l2_to_null_ratio",
    ]
    for ds, g in d.groupby("dataset"):
        for feature in candidate_features:
            rho, pv, n = safe_spearman(
                pd.to_numeric(g[feature], errors="coerce").to_numpy(float),
                g["gain_over_identity_median"].to_numpy(float),
            )
            assoc.append({
                "dataset": ds,
                "feature": feature,
                "n": n,
                "spearman_rho": rho,
                "spearman_p": pv,
            })
    assoc_df = pd.DataFrame(assoc)
    assoc_df.to_csv(args.output_dir / "within_dataset_associations.csv", index=False)

    # 2) Within-dataset standardized pooled descriptive associations.
    zframes = []
    for ds, g in d.groupby("dataset"):
        h = g.copy()
        for c in candidate_features + ["gain_over_identity_median"]:
            v = pd.to_numeric(h[c], errors="coerce")
            mu, sd = v.mean(), v.std(ddof=0)
            h[c + "_z"] = (v - mu) / sd if np.isfinite(sd) and sd > 0 else np.nan
        zframes.append(h)
    z = pd.concat(zframes, ignore_index=True)

    pooled = []
    for feature in candidate_features:
        rho, pv, n = safe_spearman(
            z[feature + "_z"].to_numpy(float),
            z["gain_over_identity_median_z"].to_numpy(float),
        )
        pooled.append({
            "feature": feature,
            "n": n,
            "within_dataset_standardized_spearman": rho,
            "p_value_descriptive_only": pv,
        })
    pooled_df = pd.DataFrame(pooled)
    pooled_df.to_csv(
        args.output_dir / "within_dataset_standardized_associations.csv",
        index=False,
    )

    # 3) Leave-one-dataset-out prediction.
    oracle_metrics, oracle_pred = lodo(d, ORACLE_FEATURES, "oracle_with_reliability")
    deploy_metrics, deploy_pred = lodo(d, DEPLOYABLE_FEATURES, "deployable_no_reliability")
    lodo_metrics = pd.concat([oracle_metrics, deploy_metrics], ignore_index=True)
    lodo_metrics.to_csv(args.output_dir / "lodo_metrics.csv", index=False)
    pd.concat([oracle_pred, deploy_pred], ignore_index=True).to_csv(
        args.output_dir / "lodo_predictions.csv", index=False
    )

    # 4) Summary emphasizing generalization, not in-sample fit.
    summary = {}
    for label, g in lodo_metrics.groupby("model"):
        summary[label] = {
            "n_held_out_datasets": int(len(g)),
            "median_lodo_spearman": float(np.nanmedian(g["spearman_pred_vs_gain"])),
            "median_rmse_improvement_vs_baseline": float(
                np.nanmedian(g["rmse_improvement_vs_baseline"])
            ),
            "datasets_rmse_better_than_baseline": int(
                (g["rmse_improvement_vs_baseline"] > 0).sum()
            ),
            "median_mae_improvement_vs_baseline": float(
                np.nanmedian(g["mae_improvement_vs_baseline"])
            ),
        }

    summary["notes"] = [
        "selected_alpha is global per validation split and is not modeled as a perturbation-specific target",
        "split-level p-values are not biological-replicate inference",
        "within-dataset association p-values are descriptive and not corrected for multiple testing",
        "LODO is the primary test of cross-context generalization",
        "oracle_with_reliability uses split-half intrinsic reliability and is not assumed deployable for unseen perturbations",
    ]

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== WITHIN-DATASET RELIABILITY VS GAIN ===")
    print(
        assoc_df[assoc_df["feature"] == "intrinsic_reliability"]
        .sort_values("dataset")
        .to_string(index=False)
    )
    print("\n=== WITHIN-DATASET STANDARDIZED ASSOCIATIONS ===")
    print(pooled_df.to_string(index=False))
    print("\n=== LODO METRICS ===")
    print(lodo_metrics.to_string(index=False))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
