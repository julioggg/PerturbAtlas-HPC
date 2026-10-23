#!/usr/bin/env python3
"""Build perturbation-specific correction landscapes from existing benchmark splits.

For every test perturbation in each repeated split, evaluate:
- identity: alpha=0 (stored once with pca_components=0)
- PCA shrinkage for each k and alpha>0

The resulting oracle labels are analysis targets only. They MUST NOT be used
from an outer held-out dataset when training a cross-dataset adaptive policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from perturbatlas.benchmark import pca_denoise_baseline, row_pearson
from perturbatlas.reproducibility import run_metadata, save_metadata


DEFAULT_K = [8, 16, 32, 64, 128]
DEFAULT_ALPHA = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def evaluate_rows(pred: np.ndarray, target: np.ndarray):
    r = row_pearson(pred, target)
    rmse = np.sqrt(np.mean((pred - target) ** 2, axis=1))
    return r, rmse


def process_seed(
    seed_dir: Path,
    output_dir: Path,
    ks: list[int],
    alphas: list[float],
    overwrite: bool,
) -> dict:
    bench = seed_dir / "benchmark"
    half_a = bench / "half_A.h5ad"
    half_b = bench / "half_B.h5ad"
    if not half_a.exists() or not half_b.exists():
        return {
            "seed_dir": str(seed_dir),
            "status": "missing_benchmark",
        }

    seed = int(seed_dir.name.split("_", 1)[1])
    out = output_dir / seed_dir.name
    out.mkdir(parents=True, exist_ok=True)
    landscape_path = out / "correction_landscape.csv"
    oracle_path = out / "oracle_targets.csv"

    if landscape_path.exists() and oracle_path.exists() and not overwrite:
        return {
            "seed": seed,
            "status": "skipped_existing",
            "landscape": str(landscape_path),
            "oracle": str(oracle_path),
        }

    a = ad.read_h5ad(half_a)
    b = ad.read_h5ad(half_b)
    xa = np.asarray(a.X, dtype=np.float32)
    xb = np.asarray(b.X, dtype=np.float32)
    split = a.obs["split"].astype(str).to_numpy()
    tr = split == "train"
    te = split == "test"

    names = np.asarray(a.obs_names[te], dtype=str)
    xtr = xa[tr]
    xte = xa[te]
    target = xb[te]

    rows = []

    # Identity appears exactly once in the landscape.
    r0, e0 = evaluate_rows(xte, target)
    rows.append(pd.DataFrame({
        "seed": seed,
        "perturbation": names,
        "pca_components": 0,
        "alpha": 0.0,
        "pearson": r0,
        "rmse": e0,
        "gain_over_identity": np.zeros(len(names), dtype=float),
    }))

    # Cache PCA predictions once per k, then sweep alpha cheaply.
    for k in ks:
        pca_pred = pca_denoise_baseline(xtr, xte, k)
        for alpha in alphas:
            if alpha <= 0:
                continue
            pred = (1.0 - alpha) * xte + alpha * pca_pred
            r, e = evaluate_rows(pred, target)
            rows.append(pd.DataFrame({
                "seed": seed,
                "perturbation": names,
                "pca_components": int(k),
                "alpha": float(alpha),
                "pearson": r,
                "rmse": e,
                "gain_over_identity": r - r0,
            }))

    landscape = pd.concat(rows, ignore_index=True)
    landscape.to_csv(landscape_path, index=False)

    # Oracle = highest Pearson, then lower RMSE, then smaller alpha, then smaller k.
    ranked = landscape.sort_values(
        ["perturbation", "pearson", "rmse", "alpha", "pca_components"],
        ascending=[True, False, True, True, True],
        na_position="last",
    )
    oracle = ranked.groupby("perturbation", as_index=False).first()
    oracle = oracle.rename(columns={
        "pca_components": "oracle_pca_components",
        "alpha": "oracle_alpha",
        "pearson": "oracle_pearson",
        "rmse": "oracle_rmse",
        "gain_over_identity": "oracle_gain_over_identity",
    })

    identity = pd.DataFrame({
        "perturbation": names,
        "identity_pearson": r0,
        "identity_rmse": e0,
    })
    oracle = oracle.merge(identity, on="perturbation", how="left")
    oracle["oracle_improves_identity"] = oracle["oracle_gain_over_identity"] > 0
    oracle["oracle_uses_identity"] = oracle["oracle_alpha"] == 0
    oracle.to_csv(oracle_path, index=False)

    summary = {
        "seed": seed,
        "n_test": int(te.sum()),
        "n_configs_per_perturbation": int(1 + len(ks) * sum(a > 0 for a in alphas)),
        "median_identity_pearson": float(np.nanmedian(r0)),
        "median_oracle_pearson": float(np.nanmedian(oracle["oracle_pearson"])),
        "median_oracle_gain_over_identity": float(
            np.nanmedian(oracle["oracle_gain_over_identity"])
        ),
        "fraction_oracle_gain_positive": float(
            np.mean(oracle["oracle_gain_over_identity"] > 0)
        ),
        "fraction_oracle_identity": float(np.mean(oracle["oracle_alpha"] == 0)),
        "oracle_alpha_counts": {
            str(k): int(v)
            for k, v in oracle["oracle_alpha"].value_counts().sort_index().items()
        },
        "oracle_pca_counts": {
            str(k): int(v)
            for k, v in oracle["oracle_pca_components"].value_counts().sort_index().items()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    save_metadata(
        run_metadata({
            "half_a": str(half_a),
            "half_b": str(half_b),
            "pca_components": ks,
            "alphas": alphas,
            "seed": seed,
        }),
        out / "run_metadata.json",
    )
    print(f"seed={seed} done: median oracle gain={summary['median_oracle_gain_over_identity']:.6f}")
    return {"seed": seed, "status": "done", **summary}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "repeated_root",
        type=Path,
        help="e.g. results/repeated_shrinkage/replogle_k562",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pca-components", nargs="+", type=int, default=DEFAULT_K)
    p.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_dirs = sorted(args.repeated_root.glob("seed_*"))
    if not seed_dirs:
        raise SystemExit(f"No seed_* directories found in {args.repeated_root}")

    audit = []
    for seed_dir in seed_dirs:
        rec = process_seed(
            seed_dir,
            args.output_dir,
            args.pca_components,
            args.alphas,
            args.overwrite,
        )
        audit.append(rec)
        pd.DataFrame(audit).to_csv(args.output_dir / "progress.csv", index=False)

    done = [x for x in audit if x.get("status") in {"done", "skipped_existing"}]
    summary = {
        "repeated_root": str(args.repeated_root),
        "n_seed_dirs": len(seed_dirs),
        "n_completed_or_existing": len(done),
        "pca_components": args.pca_components,
        "alphas": args.alphas,
        "identity_encoding": {"pca_components": 0, "alpha": 0.0},
        "oracle_definition": "max Pearson; tie-break lower RMSE, lower alpha, lower k",
        "warning": (
            "Oracle targets use test half_B and are labels for method development. "
            "Outer-held-out dataset oracle labels must never enter training/model selection."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
