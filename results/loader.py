"""Unified results loading utility for AEGIS.

Finds and loads JSON result files across categorized subdirectories:
results/ncmapss, results/mimii, results/conformal, results/ablations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

RESULTS_DIR = Path(__file__).resolve().parent

SEARCH_DIRS = [
    RESULTS_DIR,
    RESULTS_DIR / "ncmapss",
    RESULTS_DIR / "mimii",
    RESULTS_DIR / "conformal",
    RESULTS_DIR / "ablations",
]


def find_result_path(filename: str) -> Optional[Path]:
    """Find absolute Path for a given result filename across results directories."""
    p = Path(filename)
    if p.is_absolute() and p.exists():
        return p

    for d in SEARCH_DIRS:
        cand = d / p.name
        if cand.exists():
            return cand.resolve()
    return None


def load_result(filename: str) -> Dict[str, Any]:
    """Load JSON result file by name from results/ registry."""
    path = find_result_path(filename)
    if path is None:
        raise FileNotFoundError(
            f"Result file '{filename}' not found in results/ search paths: "
            f"{[str(d) for d in SEARCH_DIRS]}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
