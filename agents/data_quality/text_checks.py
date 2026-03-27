"""Text-oriented quality signals for forum / NLP datasets (PII scan, near-duplicates, optional lang)."""

from __future__ import annotations

import difflib
import re
from collections import defaultdict
from typing import Any

import pandas as pd

# Email ( pragmatic, not RFC-complete )
RE_EMAIL = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
    re.IGNORECASE,
)
# Phone-like digit runs (international-friendly; may have false positives)
RE_PHONE = re.compile(
    r"(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{2,4}\)?[\s.\-]?)?\d{3,4}[\s.\-]?\d{3,5}\b"
)
# Social / forum handle @user (exclude email @ already matched separately)
RE_HANDLE = re.compile(r"(?<![A-Za-z0-9._%+\-])@[A-Za-z0-9_]{2,32}\b")


def redact_pii(
    text: str,
    *,
    emails: bool = True,
    phones: bool = True,
    handles: bool = True,
) -> str:
    """Replace PII-like substrings with fixed tokens."""

    out = text
    if emails:
        out = RE_EMAIL.sub("[REDACTED_EMAIL]", out)
    if phones:
        out = RE_PHONE.sub("[REDACTED_PHONE]", out)
    if handles:
        out = RE_HANDLE.sub("[REDACTED_HANDLE]", out)
    return out


def pii_hit_mask(series: pd.Series) -> pd.Series:
    """True where string value matches at least one PII pattern."""

    if not len(series):
        return pd.Series(dtype=bool)
    s = series.fillna("").astype(str)

    def hit(v: str) -> bool:
        return bool(RE_EMAIL.search(v) or RE_PHONE.search(v) or RE_HANDLE.search(v))

    return s.map(hit)


def pii_breakdown_counts(series: pd.Series) -> dict[str, int]:
    """Counts of rows with at least one hit per pattern type."""

    if not len(series):
        return {"email_rows": 0, "phone_rows": 0, "handle_rows": 0, "any_rows": 0}
    s = series.fillna("").astype(str)
    email = s.map(lambda v: bool(RE_EMAIL.search(v)))
    phone = s.map(lambda v: bool(RE_PHONE.search(v)))
    handle = s.map(lambda v: bool(RE_HANDLE.search(v)))
    any_m = email | phone | handle
    return {
        "email_rows": int(email.sum()),
        "phone_rows": int(phone.sum()),
        "handle_rows": int(handle.sum()),
        "any_rows": int(any_m.sum()),
    }


def near_duplicate_drop_indices(
    texts: list[str],
    indices: list[Any],
    *,
    similarity_threshold: float,
    max_length_delta_ratio: float = 0.25,
) -> set[Any]:
    """Return index labels to drop (keep lexicographically smallest index per near-dup cluster).

    *texts* and *indices* must align; only non-empty normalized strings are clustered.
    """

    if len(texts) != len(indices) or len(texts) < 2:
        return set()

    norm = [re.sub(r"\s+", " ", t.lower().strip()) for t in texts]
    buckets: dict[tuple[int, str], list[int]] = defaultdict(list)
    for i, nt in enumerate(norm):
        if not nt:
            continue
        key = (len(nt) // 40, nt[:48])
        buckets[key].append(i)

    drop_labels: set[Any] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        # Sort by text so similar strings are adjacent; also walk a small sliding window.
        members_sorted = sorted(members, key=lambda j: norm[j])
        n = len(members_sorted)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        def similar(ai: int, bi: int) -> bool:
            ta, tb = norm[ai], norm[bi]
            if not ta or not tb:
                return False
            la, lb = len(ta), len(tb)
            if max(la, lb) == 0:
                return False
            if abs(la - lb) / max(la, lb) > max_length_delta_ratio:
                return False
            quick = difflib.SequenceMatcher(a=ta, b=tb).quick_ratio()
            if quick < similarity_threshold - 0.08:
                return False
            return difflib.SequenceMatcher(a=ta, b=tb).ratio() >= similarity_threshold

        window = 4
        for i in range(n):
            for j in range(i + 1, min(i + window, n)):
                ai, bi = members_sorted[i], members_sorted[j]
                if similar(ai, bi):
                    union(i, j)

        clusters: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(i)

        for grp in clusters.values():
            if len(grp) < 2:
                continue
            orig_idxs = [members_sorted[i] for i in grp]
            keep = min(orig_idxs, key=lambda k: (str(indices[k]), indices[k]))
            for oi in orig_idxs:
                if oi != keep:
                    drop_labels.add(indices[oi])
    return drop_labels


def language_mismatch_indices(
    series: pd.Series,
    expected: str,
    *,
    max_rows: int,
) -> tuple[set[Any] | None, str | None]:
    """Return original index labels where detected language != *expected*.

    Uses ``langdetect`` when available; otherwise returns (None, reason).
    """

    try:
        from langdetect import DetectorFactory, LangDetectException, detect  # type: ignore[import-untyped]
    except ImportError:
        return None, "langdetect_not_installed"

    DetectorFactory.seed = 0
    working = series.dropna()
    working = working[working.astype(str).str.strip().ne("")]
    if working.empty:
        return set(), None
    if len(working) > max_rows:
        working = working.sample(n=max_rows, random_state=0)

    mismatched: set[Any] = set()
    for idx, val in working.items():
        text = str(val).strip()
        if len(text) < 20:
            continue
        try:
            got = detect(text)
        except LangDetectException:
            continue
        if got != expected:
            mismatched.add(idx)
    return mismatched, None
