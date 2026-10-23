#!/usr/bin/env python3
"""Build a multidataset perturbation-level master table.

Combines:
- v2 split-half intrinsic reproducibility
- effect calibration
- repeated validation-locked shrinkage outcomes aggregated per perturbation
- calibrated uncertainty when available

The output is intended for exploratory/statistical modeling of which
perturbations benefit from correction. Repeated train/val/test partitions are
not treated as independent biological replicates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASETS = [
    "arch1",
    "replogle_k562",
    "replogle_rpe1",
    "nadig25hepg2",
    "nadig25jurkat",
    "kaden25rpe1",
    "wessels23",
]


def read_indexed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.index.name = "perturbation"
    return df


def reliability_path(root: Path, ds: str) -> Path:
    if ds == "kaden25rpe1":
        return root / "reproducibility" / "kaden25rpe1_v2" / "split_half_summary.csv"
    return root / "reproducibility_v2" / ds / "split_half_summary.csv"


def aggregate_repeated_shrinkage(root: Path, ds: str) -> pd.DataFrame:
    droot = root / "repeated_shrinkage" / ds
    rows = []

    for seed_dir in sorted(droot.glob("seed_*")):
        p = seed_dir / "shrinkage" / "per_perturbation.csv"
        summary = seed_dir / "shrinkage" / "summary.json"
        if not p.exists():
            continue

        seed = int(seed_dir.name.split("_", 1)[1])
        df = pd.read_csv(p)
        if "perturbation" not in df:
            raise ValueError(f"{p}: missing perturbation column")
        df["perturbation"] = df["perturbation"].astype(str)
        df["seed"] = seed

        if summary.exists():
            s = json.loads(summary.read_text())
            df["selected_alpha"] = float(s.get("selected_alpha", np.nan))
            df["selected_pca_components"] = float(
                s.get("selected_pca_components", np.nan)
            )
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    x = pd.concat(rows, ignore_index=True)

    def q25(s):
        return float(np.nanquantile(s, 0.25))

    def q75(s):
        return float(np.nanquantile(s, 0.75))

    agg = (
        x.groupby("perturbation", as_index=True)
        .agg(
            shrinkage_test_appearances=("seed", "nunique"),
            shrinkage_pearson_median=("pearson", "median"),
            identity_pearson_median=("identity_pearson", "median"),
            gain_over_identity_median=("gain_over_identity", "median"),
            gain_over_identity_mean=("gain_over_identity", "mean"),
            gain_over_identity_q25=("gain_over_identity", q25),
            gain_over_identity_q75=("gain_over_identity", q75),
            gain_positive_fraction=(
                "gain_over_identity",
                lambda s: float(np.mean(np.asarray(s) > 0)),
            ),
            selected_alpha_median=("selected_alpha", "median"),
            selected_pca_components_median=("selected_pca_components", "median"),
        )
        .sort_index()
    )
    return agg


def find_uncertainty(root: Path, ds: str) -> Path | None:
    candidates = [
        root / "uncertainty_calibrated" / f"{ds}_gpu" / "per_perturbation_calibrated.csv",
        root / "uncertainty_calibrated" / ds / "per_perturbation_calibrated.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_uncertainty(path: Path) -> pd.DataFrame:
    df = read_indexed(path)

    # Remove reliability columns inherited from older protocols. Reliability v2
    # is joined separately and is the only reliability used in the master table.
    drop = [
        c for c in df.columns
        if c in {"intrinsic_reproducibility", "pearson_median", "n_cells"}
        or c.startswith("intrinsic_")
    ]
    df = df.drop(columns=drop, errors="ignore")

    keep = [
        c for c in [
            "pearson",
            "rmse",
            "ensemble_variance",
            "raw_ensemble_variance",
            "relative_ensemble_variance",
            "pairwise_correlation_disagreement",
        ]
        if c in df.columns
    ]
    out = df[keep].copy()
    out = out.rename(columns={
        "pearson": "ensemble_pearson",
        "rmse": "ensemble_rmse",
        "ensemble_variance": "ensemble_variance_legacy",
    })
    return out


def load_effects(path: Path) -> pd.DataFrame:
    df = read_indexed(path)
    rename = {
        "n_cells": "effect_n_cells",
        "l2_effect": "effect_l2",
        "mean_abs_effect": "effect_mean_abs",
        "expected_null_l2": "expected_null_l2",
        "noise_adjusted_l2": "noise_adjusted_l2",
        "l2_to_null_ratio": "l2_to_null_ratio",
    }
    cols = [c for c in rename if c in df.columns]
    return df[cols].rename(columns=rename)


def load_reliability(path: Path) -> pd.DataFrame:
    df = read_indexed(path)
    rename = {
        "n_cells": "reliability_n_cells",
        "pearson_median": "intrinsic_reliability",
        "pearson_mean": "intrinsic_reliability_mean",
        "cosine_median": "intrinsic_cosine",
        "rmse_median": "intrinsic_rmse",
        "nrmse_median": "intrinsic_nrmse",
    }
    cols = [c for c in rename if c in df.columns]
    return df[cols].rename(columns=rename)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/multidataset_master"),
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    audit = []

    for ds in args.datasets:
        rel_p = reliability_path(args.results_root, ds)
        eff_p = args.results_root / "effects" / ds / "effect_calibration.csv"
        unc_p = find_uncertainty(args.results_root, ds)

        missing_required = [
            str(x) for x in [rel_p, eff_p] if not x.exists()
        ]
        shrink = aggregate_repeated_shrinkage(args.results_root, ds)
        if shrink.empty:
            missing_required.append(
                str(args.results_root / "repeated_shrinkage" / ds)
            )

        if missing_required:
            audit.append({
                "dataset": ds,
                "status": "skipped_missing_required",
                "missing": " | ".join(missing_required),
                "uncertainty_available": bool(unc_p),
            })
            print(f"SKIP {ds}: missing required inputs: {missing_required}")
            continue

        rel = load_reliability(rel_p)
        eff = load_effects(eff_p)

        master = rel.join(eff, how="outer").join(shrink, how="outer")

        if unc_p is not None:
            unc = load_uncertainty(unc_p)
            master = master.join(unc, how="left")

        master.insert(0, "dataset", ds)
        master.insert(1, "perturbation", master.index.astype(str))
        master = master.reset_index(drop=True)

        all_rows.append(master)
        audit.append({
            "dataset": ds,
            "status": "ok",
            "n_rows": int(len(master)),
            "n_reliability": int(rel.shape[0]),
            "n_effects": int(eff.shape[0]),
            "n_shrinkage_perturbations": int(shrink.shape[0]),
            "uncertainty_available": bool(unc_p),
            "uncertainty_path": str(unc_p) if unc_p else "",
        })

    if not all_rows:
        raise SystemExit("No datasets could be assembled.")

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(args.output_dir / "perturbation_master_table.csv", index=False)

    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(args.output_dir / "assembly_audit.csv", index=False)

    # Dataset-level descriptors derived from perturbation-level information.
    grouped = []
    for ds, d in out.groupby("dataset", sort=False):
        r = d["intrinsic_reliability"].dropna()
        g = d["gain_over_identity_median"].dropna()
        common = d.dropna(
            subset=["intrinsic_reliability", "gain_over_identity_median"]
        )
        grouped.append({
            "dataset": ds,
            "n_rows": int(len(d)),
            "median_intrinsic_reliability": float(r.median()) if len(r) else np.nan,
            "median_gain_over_identity": float(g.median()) if len(g) else np.nan,
            "spearman_reliability_vs_gain": float(
                common["intrinsic_reliability"].corr(
                    common["gain_over_identity_median"], method="spearman"
                )
            ) if len(common) >= 3 else np.nan,
            "n_with_uncertainty": int(
                d.get(
                    "relative_ensemble_variance",
                    pd.Series(index=d.index, dtype=float),
                ).notna().sum()
            ),
        })

    ds_summary = pd.DataFrame(grouped)
    ds_summary.to_csv(args.output_dir / "dataset_level_summary.csv", index=False)

    global_common = out.dropna(
        subset=["intrinsic_reliability", "gain_over_identity_median"]
    )
    global_summary = {
        "n_datasets": int(out["dataset"].nunique()),
        "n_rows": int(len(out)),
        "n_rows_reliability_and_gain": int(len(global_common)),
        "pooled_spearman_reliability_vs_gain": float(
            global_common["intrinsic_reliability"].corr(
                global_common["gain_over_identity_median"], method="spearman"
            )
        ) if len(global_common) >= 3 else np.nan,
        "warning": (
            "The pooled perturbation-level correlation ignores dataset clustering "
            "and is descriptive only. Use dataset-aware/mixed-effects inference "
            "for manuscript claims."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(global_summary, indent=2), encoding="utf-8"
    )

    print("\n=== ASSEMBLY AUDIT ===")
    print(audit_df.to_string(index=False))
    print("\n=== DATASET SUMMARY ===")
    print(ds_summary.to_string(index=False))
    print("\n=== GLOBAL SUMMARY ===")
    print(json.dumps(global_summary, indent=2))


if __name__ == "__main__":
    main()
