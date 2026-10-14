#!/usr/bin/env python3
"""Recompute scale-invariant uncertainty metrics from saved ensemble predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from perturbatlas.uncertainty import (
    pairwise_correlation_disagreement,
    perturbation_uncertainty,
    relative_ensemble_variance,
    selective_prediction_curve,
    uncertainty_diagnostics,
)
from perturbatlas.reproducibility import run_metadata, save_metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("predictions", type=Path)
    p.add_argument("table", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/uncertainty_calibrated/replogle_k562"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.predictions)
    preds = z["member_predictions"]
    table = pd.read_csv(args.table, index_col=0)

    table["raw_ensemble_variance"] = perturbation_uncertainty(preds)
    table["relative_ensemble_variance"] = relative_ensemble_variance(preds)
    table["pairwise_correlation_disagreement"] = pairwise_correlation_disagreement(preds)

    metrics = {}
    curves = []
    for col in [
        "raw_ensemble_variance",
        "relative_ensemble_variance",
        "pairwise_correlation_disagreement",
    ]:
        metrics[col] = uncertainty_diagnostics(table, col)
        curve = selective_prediction_curve(table, col)
        curve.insert(0, "uncertainty_metric", col)
        curves.append(curve)

    table.to_csv(args.output_dir / "per_perturbation_calibrated.csv")
    pd.concat(curves, ignore_index=True).to_csv(
        args.output_dir / "selective_prediction.csv", index=False
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    save_metadata(run_metadata(vars(args)), args.output_dir / "run_metadata.json")

    print("\n=== UNCERTAINTY RECALIBRATION ===")
    for name, vals in metrics.items():
        print(f"\n[{name}]")
        for k, v in vals.items():
            print(f"{k}: {v}")
    print("\n=== SELECTIVE PREDICTION ===")
    print(pd.concat(curves, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
