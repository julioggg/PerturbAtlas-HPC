#!/usr/bin/env python3
"""Inspect an AnnData object without modifying it."""

from __future__ import annotations

import argparse
from pathlib import Path

import scanpy as sc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Path to input .h5ad file")
    args = parser.parse_args()

    adata = sc.read_h5ad(args.input, backed="r")

    print(f"shape: {adata.shape}")
    print(f"obs columns: {list(adata.obs.columns)}")
    print(f"var columns: {list(adata.var.columns)}")
    print(f"layers: {list(adata.layers.keys())}")
    print(f"obsm: {list(adata.obsm.keys())}")
    print(f"uns keys: {list(adata.uns.keys())}")


if __name__ == "__main__":
    main()
