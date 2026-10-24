#!/usr/bin/env python3
"""Nested LODO benchmark for the adaptive personalization gate.

Task
----
Predict whether a perturbation has median oracle regret > threshold, using only
features available without outer-test labels.

Protocol
--------
- Outer leave-one-dataset-out evaluation.
- Inner leave-one-training-dataset-out model selection.
- Probability threshold chosen ONLY from pooled inner validation predictions.
- Dataset-balanced sample weights during training.
- Reports ordinary classification metrics and regret-weighted utility metrics.

The regret-weighted metrics are diagnostic upper bounds for the gate only:
they assume a downstream selector could recover the oracle regret of a flagged
perturbation. They are NOT end-to-end adaptive-method performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEPLOYABLE = [
    "reliability_n_cells",
    "effect_l2",
    "effect_mean_abs",
    "expected_null_l2",
    "noise_adjusted_l2",
    "l2_to_null_ratio",
]
ORACLE = ["intrinsic_reliability", *DEPLOYABLE]

LOG_FEATURES = {
    "reliability_n_cells",
    "effect_l2",
    "effect_mean_abs",
    "expected_null_l2",
    "l2_to_null_ratio",
}

THRESHOLDS = np.linspace(0.10, 0.90, 33)


def add_context_features(df: pd.DataFrame, base: list[str]):
    out = df.copy()
    new = []
    for c in base:
        if c not in out:
            continue
        g = out.groupby("dataset")[c]
        med = g.transform("median")
        q25 = g.transform(lambda x: x.quantile(.25))
        q75 = g.transform(lambda x: x.quantile(.75))
        for name, val in [
            (f"ctx_{c}_median", med),
            (f"ctx_{c}_iqr", q75 - q25),
        ]:
            out[name] = val
            new.append(name)
    return out, base + new


def prep_x(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    x = df[feats].copy()
    for c in feats:
        v = pd.to_numeric(x[c], errors="coerce")
        if any(c == lf or c.startswith(f"ctx_{lf}_") for lf in LOG_FEATURES):
            v = np.where(v >= 0, np.log1p(v), np.nan)
        x[c] = v
    return x


def dataset_balanced_weights(df: pd.DataFrame) -> np.ndarray:
    counts = df["dataset"].value_counts()
    w = df["dataset"].map(lambda d: 1.0 / counts[d]).to_numpy(float)
    w *= len(w) / w.sum()
    return w


def model_specs(seed: int):
    return {
        "logistic": lambda: Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                random_state=seed,
            )),
        ]),
        "rf": lambda: Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            )),
        ]),
        "extra": lambda: Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", ExtraTreesClassifier(
                n_estimators=400,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            )),
        ]),
        "hgb": lambda: Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=seed,
            )),
        ]),
    }


def fit_model(model, x, y, sample_weight):
    # All pipelines name the final estimator "m".
    model.fit(x, y, m__sample_weight=sample_weight)
    return model


def safe_auc(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def safe_ap(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    if y.sum() == 0:
        return np.nan
    return float(average_precision_score(y, p))


def choose_threshold(y, p):
    rows = []
    for t in THRESHOLDS:
        z = p >= t
        rows.append({
            "threshold": float(t),
            "balanced_accuracy": float(balanced_accuracy_score(y, z)),
            "f1": float(f1_score(y, z, zero_division=0)),
            "recall": float(recall_score(y, z, zero_division=0)),
            "precision": float(precision_score(y, z, zero_division=0)),
        })
    tab = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "f1"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return float(tab.iloc[0]["threshold"]), tab


def utility_metrics(regret, selected):
    r = np.asarray(regret, float)
    s = np.asarray(selected, bool)
    total = np.nansum(np.clip(r, 0, None))
    captured = np.nansum(np.clip(r[s], 0, None))
    return {
        "selected_fraction": float(np.mean(s)),
        "regret_capture_fraction": float(captured / total) if total > 0 else np.nan,
        "median_regret_selected": float(np.nanmedian(r[s])) if s.any() else np.nan,
        "median_regret_not_selected": float(np.nanmedian(r[~s])) if (~s).any() else np.nan,
    }


def classification_metrics(y, p, threshold, regret):
    z = p >= threshold
    out = {
        "auroc": safe_auc(y, p),
        "auprc": safe_ap(y, p),
        "brier": float(brier_score_loss(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, z)),
        "f1": float(f1_score(y, z, zero_division=0)),
        "recall": float(recall_score(y, z, zero_division=0)),
        "precision": float(precision_score(y, z, zero_division=0)),
    }
    out.update(utility_metrics(regret, z))
    return out


def inner_select(train, feats, target_col, seed):
    specs = model_specs(seed)
    model_rows = []
    threshold_rows = []

    for name, make in specs.items():
        parts = []
        for held in sorted(train["dataset"].unique()):
            tr = train[train["dataset"] != held].copy()
            va = train[train["dataset"] == held].copy()
            if len(va) < 5:
                continue

            ytr = tr[target_col].astype(int).to_numpy()
            yva = va[target_col].astype(int).to_numpy()
            w = dataset_balanced_weights(tr)

            model = make()
            fit_model(model, prep_x(tr, feats), ytr, w)
            p = model.predict_proba(prep_x(va, feats))[:, 1]

            part = pd.DataFrame({
                "dataset": held,
                "y": yva,
                "p": p,
                "regret": va["oracle_regret_median"].to_numpy(float),
            })
            parts.append(part)

        oof = pd.concat(parts, ignore_index=True)
        t, ttab = choose_threshold(oof["y"].to_numpy(), oof["p"].to_numpy())
        m = classification_metrics(
            oof["y"].to_numpy(),
            oof["p"].to_numpy(),
            t,
            oof["regret"].to_numpy(),
        )
        model_rows.append({
            "model": name,
            "inner_threshold": t,
            **m,
        })
        ttab.insert(0, "model", name)
        threshold_rows.append(ttab)

    tab = pd.DataFrame(model_rows).sort_values(
        ["auprc", "auroc", "balanced_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best = str(tab.iloc[0]["model"])
    best_threshold = float(tab.iloc[0]["inner_threshold"])
    return best, best_threshold, tab, pd.concat(threshold_rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "atlas",
        type=Path,
        nargs="?",
        default=Path("results/adaptive_regime_atlas/adaptive_regime_atlas.csv"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/adaptive_gate_v1"),
    )
    ap.add_argument("--regret-threshold", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_col = (
        "target_personalize_regret_gt_0_01"
        if np.isclose(args.regret_threshold, 0.01)
        else "target_personalize_regret_gt_0_02"
    )

    df = pd.read_csv(args.atlas)
    df = df.dropna(subset=[target_col, "oracle_regret_median"]).copy()
    df[target_col] = df[target_col].astype(int)

    configs = []
    for fs_name, base in [("deployable", DEPLOYABLE), ("oracle", ORACLE)]:
        configs.append((fs_name, base, False))
        configs.append((fs_name + "_context", base, True))

    outer_rows = []
    selection_tables = []
    threshold_tables = []
    pred_rows = []

    for fs_name, base, use_ctx in configs:
        work = df.copy()
        feats = list(base)
        if use_ctx:
            work, feats = add_context_features(work, feats)

        for held in sorted(work["dataset"].unique()):
            tr = work[work["dataset"] != held].copy()
            te = work[work["dataset"] == held].copy()

            best, threshold, stab, ttab = inner_select(
                tr, feats, target_col, args.seed
            )
            stab.insert(0, "feature_set", fs_name)
            stab.insert(1, "outer_held_out_dataset", held)
            selection_tables.append(stab)

            ttab.insert(0, "feature_set", fs_name)
            ttab.insert(1, "outer_held_out_dataset", held)
            threshold_tables.append(ttab)

            ytr = tr[target_col].astype(int).to_numpy()
            yte = te[target_col].astype(int).to_numpy()
            w = dataset_balanced_weights(tr)

            model = model_specs(args.seed)[best]()
            fit_model(model, prep_x(tr, feats), ytr, w)
            p = model.predict_proba(prep_x(te, feats))[:, 1]
            met = classification_metrics(
                yte,
                p,
                threshold,
                te["oracle_regret_median"].to_numpy(float),
            )

            outer_rows.append({
                "feature_set": fs_name,
                "held_out_dataset": held,
                "selected_model": best,
                "selected_threshold": threshold,
                "n_test": len(te),
                "positive_fraction": float(np.mean(yte)),
                **met,
            })

            q = te[["dataset", "perturbation", "oracle_regret_median"]].copy()
            q["feature_set"] = fs_name
            q["selected_model"] = best
            q["selected_threshold"] = threshold
            q["target"] = yte
            q["predicted_probability"] = p
            q["selected_for_personalization"] = p >= threshold
            pred_rows.append(q)

    outer = pd.DataFrame(outer_rows)
    selections = pd.concat(selection_tables, ignore_index=True)
    thresholds = pd.concat(threshold_tables, ignore_index=True)
    preds = pd.concat(pred_rows, ignore_index=True)

    outer.to_csv(args.output_dir / "outer_metrics.csv", index=False)
    selections.to_csv(args.output_dir / "inner_model_selection.csv", index=False)
    thresholds.to_csv(args.output_dir / "inner_threshold_search.csv", index=False)
    preds.to_csv(args.output_dir / "outer_predictions.csv", index=False)

    summary = []
    for fs, g in outer.groupby("feature_set"):
        summary.append({
            "feature_set": fs,
            "median_outer_auroc": float(np.nanmedian(g["auroc"])),
            "min_outer_auroc": float(np.nanmin(g["auroc"])),
            "median_outer_auprc": float(np.nanmedian(g["auprc"])),
            "median_outer_balanced_accuracy": float(np.nanmedian(g["balanced_accuracy"])),
            "median_regret_capture_fraction": float(np.nanmedian(g["regret_capture_fraction"])),
            "median_selected_fraction": float(np.nanmedian(g["selected_fraction"])),
            "datasets_auroc_gt_0_7": int((g["auroc"] > 0.7).sum()),
        })
    summary_df = pd.DataFrame(summary).sort_values(
        ["median_outer_auprc", "median_outer_auroc"],
        ascending=[False, False],
    )
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)

    meta = {
        "task": f"predict median oracle regret > {args.regret_threshold}",
        "target_column": target_col,
        "selection_protocol": (
            "nested leave-one-training-dataset-out; outer dataset excluded from "
            "model and probability-threshold selection"
        ),
        "training_weighting": "inverse dataset size; classifiers also class-balanced where supported",
        "selection_metric": "inner pooled AUPRC, tie-break AUROC then balanced accuracy",
        "threshold_selection": "inner pooled validation balanced accuracy, tie-break F1",
        "regret_capture_warning": (
            "gate-only oracle diagnostic; assumes perfect downstream recovery and "
            "is not end-to-end adaptive performance"
        ),
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(meta, indent=2))

    print("\n=== ADAPTIVE GATE V1 SUMMARY ===")
    print(summary_df.to_string(index=False))
    print("\n=== OUTER METRICS ===")
    print(outer.to_string(index=False))


if __name__ == "__main__":
    main()
