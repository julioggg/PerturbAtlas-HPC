#!/usr/bin/env python3
"""Build disjoint half-signature pairs for predictive benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from perturbatlas.benchmark import make_half_signature_pair, split_perturbations
from perturbatlas.reproducibility import run_metadata, save_metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("raw", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/benchmark/replogle_k562"))
    p.add_argument("--seed", type=int, default=20261013)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    a, b = make_half_signature_pair(args.raw, seed=args.seed)
    train, val, test = split_perturbations(a.obs_names, seed=args.seed)
    split = ["train"] * a.n_obs
    for i in val:
        split[i] = "val"
    for i in test:
        split[i] = "test"
    a.obs["split"] = split
    b.obs["split"] = split

    a.write_h5ad(args.output_dir / "half_A.h5ad", compression="gzip")
    b.write_h5ad(args.output_dir / "half_B.h5ad", compression="gzip")
    summary = {
        "n_perturbations": int(a.n_obs),
        "n_genes": int(a.n_vars),
        "n_train": int((a.obs["split"] == "train").sum()),
        "n_val": int((a.obs["split"] == "val").sum()),
        "n_test": int((a.obs["split"] == "test").sum()),
        "seed": int(args.seed),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_metadata(run_metadata({"raw": str(args.raw), "seed": args.seed}), args.output_dir / "run_metadata.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
