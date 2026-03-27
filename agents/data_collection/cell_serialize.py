"""Deterministic string form for tabular cells (hashing, duplicate-safe paths)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from agents.data_collection.text_unified_schema import scalar_cell_is_na


def stable_cell_token(value: Any) -> str:
    """Serialize one cell for hashing / fingerprinting."""

    if value is None:
        return ""
    if scalar_cell_is_na(value):
        return ""
    if hasattr(value, "item"):
        try:
            if getattr(value, "size", 1) == 1:
                value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return str(value)
