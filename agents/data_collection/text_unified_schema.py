"""Unified output schema for text collection → quality → annotate pipeline.

Official **text pipeline contract** (every normalized per-source frame and the merged
dataframe MUST expose these columns; values may be NA / empty JSON where a source
does not provide data). This extends the historical
``text, label, source, collected_at`` subset with ``target_text`` (optional assistant / reference
string) without breaking older code paths.

Additional columns (``record_hash``, ``source_id``, ``audio``, ``image``, connector-specific
fields, etc.) are allowed and preserved through merge. Chat / instruction flattening is
implemented in :mod:`agents.data_collection.canonical_sample`.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

# Canonical column order for documentation, reordering, and merge alignment.
TEXT_PIPELINE_CONTRACT_COLUMNS: tuple[str, ...] = (
    "id",
    "text",
    "target_text",
    "title",
    "body",
    "label",
    "source",
    "source_type",
    "source_url",
    "collected_at",
    "metadata",
)

# Explicit alias for documentation / validation (same as TEXT_PIPELINE_CONTRACT_COLUMNS).
TEXT_PIPELINE_CONTRACT_REQUIRED: tuple[str, ...] = TEXT_PIPELINE_CONTRACT_COLUMNS

# Show tqdm during slow row-wise steps (HF-sized frames); download tqdm comes from ``datasets`` only.
_TEXT_FILL_PROGRESS_MIN_ROWS = 5000


def scalar_cell_is_na(value: Any) -> bool:
    """True only for missing *scalar* cells. Avoids ``if pd.isna(ndarray)`` truth-value errors."""

    if value is None or value is pd.NA:
        return True
    try:
        if value is pd.NaT:
            return True
    except Exception:
        pass
    if not pd.api.types.is_scalar(value):
        return False
    try:
        res = pd.isna(value)
    except TypeError:
        return False
    if res is True:
        return True
    if res is False:
        return False
    try:
        return bool(res)
    except (TypeError, ValueError):
        return False


def series_all_na(s: pd.Series) -> bool:
    if s.empty:
        return True
    return bool(s.isna().all())


def reorder_dataframe_contract_first(df: pd.DataFrame) -> pd.DataFrame:
    """Place contract columns first; preserve order of any extra columns."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected pandas DataFrame")
    seen: set[str] = set()
    ordered: list[str] = []
    for col in TEXT_PIPELINE_CONTRACT_COLUMNS:
        if col in df.columns and col not in seen:
            ordered.append(col)
            seen.add(col)
    for col in df.columns:
        if col not in seen:
            ordered.append(col)
            seen.add(col)
    return df.loc[:, ordered]


def ensure_text_pipeline_contract_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every contract column exists; unknown values use NA or ``{}`` for ``metadata``."""

    out = df.copy()
    for col in TEXT_PIPELINE_CONTRACT_COLUMNS:
        if col not in out.columns:
            if col == "metadata":
                out[col] = "{}"
            else:
                out[col] = pd.NA
    return out


def _cell_str(value: Any) -> str:
    if value is None:
        return ""
    if scalar_cell_is_na(value):
        return ""
    return str(value).strip()


def fill_missing_text_from_title_body(df: pd.DataFrame) -> pd.DataFrame:
    """Where ``text`` is missing or blank, compose from non-empty ``title`` / ``body``."""

    out = df.copy()
    if out.empty:
        return out
    if "text" not in out.columns:
        out["text"] = pd.NA
    for col in ("title", "body"):
        if col not in out.columns:
            out[col] = pd.NA

    new_text = []
    _rows = out.iterrows()
    if len(out) >= _TEXT_FILL_PROGRESS_MIN_ROWS:
        try:
            from tqdm.auto import tqdm

            _rows = tqdm(_rows, total=len(out), desc="fill text from title/body", leave=False, unit="row")
        except Exception:
            pass
    for _, row in _rows:
        raw_t = row["text"]
        if _cell_str(raw_t):
            new_text.append(raw_t)
            continue
        t = _cell_str(row.get("title"))
        b = _cell_str(row.get("body"))
        parts = [p for p in (t, b) if p]
        new_text.append("\n\n".join(parts) if parts else pd.NA)
    out["text"] = new_text
    return out


def validate_merged_text_pipeline_contract(df: pd.DataFrame) -> list[str]:
    """Return human-readable issues (warnings) for the merged dataset.

    Missing contract columns are reported as errors in the message text; callers typically
    attach the list to :class:`ValidationReport.warnings`.
    """

    issues: list[str] = []
    missing = [c for c in TEXT_PIPELINE_CONTRACT_REQUIRED if c not in df.columns]
    if missing:
        issues.append(
            "Merged dataframe missing unified schema columns: " + ", ".join(missing)
        )
    if df.empty:
        issues.append("Merged dataframe is empty.")
        return issues

    if df["source"].isna().all():
        issues.append(
            "Unified schema: column `source` is entirely null (provenance may be incomplete)."
        )
    if df["source_type"].isna().all():
        issues.append(
            "Unified schema: column `source_type` is entirely null (provenance may be incomplete)."
        )

    text_series = df["text"] if "text" in df.columns else pd.Series(dtype=object)
    nonempty = text_series.notna() & (text_series.astype(str).str.strip().ne(""))
    if not bool(nonempty.any()):
        issues.append(
            "Unified schema: no non-empty `text` values after merge (check title/body/prompt mapping)."
        )

    return issues
