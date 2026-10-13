"""Utilities for reproducible experiment metadata."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_commit() -> str:
    """Return the current Git commit SHA, if available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def run_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Build compact metadata for a computational experiment."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": config,
    }


def save_metadata(metadata: dict[str, Any], path: str | Path) -> None:
    """Write metadata as human-readable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
