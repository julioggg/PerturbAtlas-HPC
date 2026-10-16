#!/usr/bin/env python3
"""Inspect/download standardized scPertEval datasets listed in the registry."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

import yaml


def load_registry(path: Path) -> dict[str, dict]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {d["id"]: d for d in payload["datasets"]}


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            value = r.headers.get("Content-Length")
            return int(value) if value is not None else None
    except Exception:
        return None


def human_bytes(n: int | None) -> str:
    if n is None:
        return "unknown"
    x = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024 or unit == "TB":
            return f"{x:.1f}{unit}"
        x /= 1024
    return f"{x:.1f}TB"


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as src, tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    tmp.replace(path)
    print(f"Saved: {path} ({human_bytes(path.stat().st_size)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/datasets/scperteval_registry.yaml"),
    )
    ap.add_argument("--list", action="store_true", help="List registry and local/remote status.")
    ap.add_argument("--download", nargs="*", default=None, metavar="DATASET_ID")
    args = ap.parse_args()

    registry = load_registry(args.registry)

    if args.list or args.download is None:
        print("dataset_id	context	class	role	local	remote_size")
        for dataset_id, meta in registry.items():
            path = Path(meta["local_path"])
            local = human_bytes(path.stat().st_size) if path.exists() else "missing"
            size = remote_size(meta["url"])
            print(
                f"{dataset_id}	{meta['cell_context']}	{meta['perturbation_class']}	"
                f"{meta['role']}	{local}	{human_bytes(size)}"
            )

    if args.download is not None:
        wanted = args.download or list(registry)
        unknown = [x for x in wanted if x not in registry]
        if unknown:
            print(f"Unknown dataset IDs: {unknown}", file=sys.stderr)
            sys.exit(2)

        for dataset_id in wanted:
            meta = registry[dataset_id]
            path = Path(meta["local_path"])
            if path.exists() and path.stat().st_size > 0:
                print(f"SKIP {dataset_id}: already exists at {path} ({human_bytes(path.stat().st_size)})")
                continue
            download(meta["url"], path)


if __name__ == "__main__":
    main()
