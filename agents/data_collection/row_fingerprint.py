"""Row fingerprints and duplicate detection safe for unhashable cells (e.g. ndarray)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pandas as pd

from agents.data_collection.cell_serialize import stable_cell_token

# Match collection normalizer: show tqdm for large frames on row-wise apply.
ROW_FINGERPRINT_PROGRESS_MIN_ROWS = 5000


def series_row_fingerprints(
    df: pd.DataFrame, columns: list[str], *, tqdm_desc: str
) -> pd.Series:
    """SHA256 hex per row over *columns* (same serialization as ``record_hash``)."""

    cols = list(columns)

    def fingerprint(row: pd.Series) -> str:
        payload = "\x1f".join(f"{col}={stable_cell_token(row[col])}" for col in cols)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    if len(df) >= ROW_FINGERPRINT_PROGRESS_MIN_ROWS:
        try:
            from tqdm.auto import tqdm

            tqdm.pandas(desc=tqdm_desc, leave=False, unit="row")
            return df.progress_apply(fingerprint, axis=1)
        except Exception:
            pass
    return df.apply(fingerprint, axis=1)


def dataframe_duplicate_count(df: pd.DataFrame) -> int:
    """Full-row duplicate count; falls back to fingerprints when ``duplicated()`` cannot hash rows."""

    if df.empty:
        return 0
    try:
        return int(df.duplicated().sum())
    except TypeError:
        sigs = series_row_fingerprints(df, list(df.columns), tqdm_desc="dup_count")
        return int(sigs.duplicated().sum())


def safe_duplicated(
    df: pd.DataFrame,
    *,
    subset: Sequence[str] | None = None,
    keep: str | bool = "first",
) -> pd.Series:
    """Like ``DataFrame.duplicated`` but supports unhashable object cells (ndarray, etc.)."""

    if df.empty:
        return pd.Series(dtype=bool, index=df.index)
    if subset is None:
        work = df
    else:
        cols = [c for c in subset if c in df.columns]
        if not cols:
            return pd.Series(False, index=df.index, dtype=bool)
        work = df.loc[:, cols]
    try:
        return work.duplicated(keep=keep)
    except TypeError:
        sigs = series_row_fingerprints(work, list(work.columns), tqdm_desc="quality_dup")
        return sigs.duplicated(keep=keep)
