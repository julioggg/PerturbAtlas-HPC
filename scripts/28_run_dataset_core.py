#!/usr/bin/env python3
"""Run the core reliability pipeline for one standardized dataset.

This orchestrator intentionally reuses the validated command-line scripts so
the same analysis steps are executed for every dataset with explicit paths.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset_id")
    p.add_argument("raw", type=Path)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--skip-repro", action="store_true")
    args = p.parse_args()

    py = sys.executable
    ds = args.dataset_id

    atlas = Path(f"results/atlas/{ds}_response_atlas.h5ad")
    effects_dir = Path(f"results/effects/{ds}")
    validation_dir = Path(f"results/validation/{ds}")
    repro_dir = Path(f"results/reproducibility/{ds}")

    run([py, "scripts/01_profile_dataset.py", str(args.raw)])
    run([py, "scripts/02_build_response_atlas.py", str(args.raw), "--output", str(atlas)])
    run([
        py, "scripts/03_calibrate_effects.py",
        str(args.raw), str(atlas),
        "--output-dir", str(effects_dir),
    ])
    run([
        py, "scripts/05_validate_biology_and_ranks.py",
        str(atlas), str(effects_dir / "effect_calibration.csv"),
        "--output-dir", str(validation_dir),
    ])
    if not args.skip_repro:
        run([
            py, "scripts/09_split_half_reproducibility.py",
            str(args.raw),
            "--output-dir", str(repro_dir),
            "--repeats", str(args.repeats),
            "--seed", str(args.seed),
        ])


if __name__ == "__main__":
    main()
