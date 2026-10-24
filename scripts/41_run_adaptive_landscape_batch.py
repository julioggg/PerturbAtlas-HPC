#!/usr/bin/env python3
"""Run correction-landscape, stability and regret analyses for multiple datasets.

Checkpoint-friendly orchestrator: existing outputs are reused by script 38 unless
--overwrite is explicitly passed there. Each dataset is processed serially.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["replogle_rpe1", "nadig25hepg2", "nadig25jurkat"],
    )
    args = p.parse_args()

    py = sys.executable
    for ds in args.datasets:
        repeated = Path("results/repeated_shrinkage") / ds
        landscape = Path("results/correction_landscapes") / ds
        stability = Path("results/oracle_stability") / ds
        regret = Path("results/oracle_regret") / ds

        run([
            py, "scripts/38_build_correction_landscapes.py",
            str(repeated),
            "--output-dir", str(landscape),
        ])
        run([
            py, "scripts/39_aggregate_oracle_stability.py",
            str(landscape),
            "--output-dir", str(stability),
        ])
        run([
            py, "scripts/40_oracle_vs_global_regret.py",
            str(landscape),
            str(repeated),
            "--output-dir", str(regret),
        ])

        print(f"\n=== COMPLETED {ds} ===", flush=True)

    print("\nAll requested datasets completed.", flush=True)


if __name__ == "__main__":
    main()
