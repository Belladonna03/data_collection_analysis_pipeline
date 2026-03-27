"""Safe handling of heterogeneous object cells (list / ndarray / dict) in quality checks.

Boolean checks like ``if value`` or ``if pd.isna(value)`` are undefined for array-like
scalars and raise ``ValueError`` from NumPy/pandas. Normalization runs before text /
missing logic so downstream code can keep using string ops on object columns.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def needs_cell_normalization(value: Any) -> bool:
    """True when *value* is not a safe scalar for ``pd.isna`` / truthiness checks."""

    if value is None:
        return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return True
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, pd.Series):
        return True
    try:
        if pd.api.types.is_scalar(value):
            return False
    except Exception:
        pass
    return True


def normalize_scalar_like(value: Any) -> Any:
    """Normalize a single cell for quality pipelines.

    - ``None`` and scalar NA-like values -> ``None``
    - ``numpy.ndarray`` (incl. 0-d) -> JSON string of a list or JSON primitive
    - ``list`` / ``tuple`` / ``set`` -> JSON string (sets sorted for stability)
    - ``dict`` -> JSON string (``sort_keys=True``)
    - ordinary scalars (numbers, ``str``, ``bool``, ``numpy`` scalars, timestamps) unchanged
    """

    if value is None:
        return None

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.ndim == 0:
            return normalize_scalar_like(value.item())
        return json.dumps(_to_jsonable(value.tolist()), ensure_ascii=False)

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return json.dumps(_to_jsonable(list(value)), ensure_ascii=False)

    if isinstance(value, set):
        if len(value) == 0:
            return None
        return json.dumps(_to_jsonable(sorted(value, key=lambda x: str(x))), ensure_ascii=False)

    if isinstance(value, dict):
        if len(value) == 0:
            return None
        return json.dumps(_to_jsonable(value), ensure_ascii=False, sort_keys=True)

    if isinstance(value, pd.Series):
        if value.empty:
            return None
        return normalize_scalar_like(value.to_list())

    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return str(value)

    try:
        if pd.api.types.is_scalar(value):
            try:
                if pd.isna(value):
                    return None
            except (ValueError, TypeError):
                return str(value)
            return value
    except Exception:
        pass

    return str(value)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in sorted(obj.items(), key=lambda kv: str(kv[0])):
            out[str(k)] = _to_jsonable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, set):
        return [_to_jsonable(x) for x in sorted(obj, key=lambda x: str(x))]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if obj is None:
        return None
    try:
        if pd.api.types.is_scalar(obj):
            try:
                if pd.isna(obj):
                    return None
            except (ValueError, TypeError):
                return str(obj)
            if hasattr(obj, "isoformat"):
                try:
                    return obj.isoformat()
                except Exception:
                    return str(obj)
            if isinstance(obj, (str, int, float, bool)):
                return obj
            return obj.item() if hasattr(obj, "item") else str(obj)
    except Exception:
        pass
    return str(obj)


def scalar_pd_notna(value: Any) -> bool:
    """Single-cell replacement for ``pd.notna`` that never returns an array truth value."""

    if value is None:
        return False
    if isinstance(value, (list, tuple, set)):
        return len(value) > 0
    if isinstance(value, dict):
        return len(value) > 0
    if isinstance(value, np.ndarray):
        return value.size > 0
    if isinstance(value, pd.Series):
        return not value.empty
    try:
        if pd.api.types.is_scalar(value):
            try:
                return bool(pd.notna(value))
            except (ValueError, TypeError):
                return True
    except Exception:
        pass
    return True


def scalar_pd_isna(value: Any) -> bool:
    """Single-cell ``pd.isna`` semantics without ambiguous array bool."""

    return not scalar_pd_notna(value)


def coerce_quality_row_id(value: Any) -> int | None:
    """Coerce ``__quality_row_id`` cell to ``int`` or ``None``."""

    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return int(value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.size == 1:
            return coerce_quality_row_id(value.item())
        return None
    try:
        if pd.api.types.is_scalar(value) and pd.notna(value):
            return int(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return None


def iter_config_column_names(raw: Any) -> list[str]:
    """Flatten config-provided column name(s) to ``str`` without ambiguous membership checks."""

    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        return iter_config_column_names(raw.tolist())
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            name = config_column_name(item)
            if name is not None:
                out.append(name)
        return out
    name = config_column_name(raw)
    return [name] if name is not None else []


def config_column_name(value: Any) -> str | None:
    """Coerce YAML/config column hints to ``str``; avoids ``if col`` on ``ndarray``."""

    if value is None or value is False:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return None
        first = value[0]
        return config_column_name(first)
    try:
        iff_scalar = pd.api.types.is_scalar(value)
    except Exception:
        iff_scalar = False
    if iff_scalar:
        try:
            if pd.isna(value):
                return None
        except (ValueError, TypeError):
            return str(value).strip() or None
        return str(value).strip() or None
    return None


def normalize_dataframe_object_cells(
    df: pd.DataFrame,
    *,
    internal_cols: frozenset[str],
    columns_subset: list[str] | None = None,
) -> pd.DataFrame:
    """Replace non-scalar object cells with JSON/text scalars; cheap no-op when possible.

    When *columns_subset* is set, only those columns (if present) are scanned. This avoids
    rewriting unrelated object columns that are not part of text/missing/label semantics.
    """

    out = df
    copied = False
    names = columns_subset if columns_subset is not None else [c for c in df.columns if c not in internal_cols]
    for col in names:
        if col in internal_cols or col not in df.columns:
            continue
        s = df[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        try:
            mask = s.map(needs_cell_normalization)
        except (ValueError, TypeError):
            mask = s.map(lambda v: needs_cell_normalization(v))
        if not bool(mask.any()):
            continue
        if not copied:
            out = df.copy(deep=True)
            copied = True
        out.loc[mask, col] = out.loc[mask, col].map(normalize_scalar_like)
    return out
