"""Canonical text sample normalization: structure-based routing (no source / dataset names).

Pipeline per row (**structural schema**, not provenance):

1. **Direct text** — если ``text`` непустой → canonical ``text``; ``target_text`` из колонки ``target_text``,
   если непустая.

2. **Chat-style** — иначе, если ячейка ``messages`` **семантически непустая** → парсинг чата →
   ``text`` / ``target_text``.

3. **Instruction-style** — иначе ищем первый непустой *source* среди кандидатов и первый непустой *target*
   среди кандидатов (см. ``SOURCE_TEXT_CANDIDATES`` / ``TARGET_TEXT_CANDIDATES``).

4. **Title/body** не обрабатываются здесь: их подставляет ``fill_missing_text_from_title_body`` в нормализаторе;
   итоговый пустой ``text`` → дроп с причиной ``missing_text``.

**Причины дропа (chat, до title/body):** ``malformed_messages``, ``missing_messages``,
``unsupported_message_schema``, ``empty_user_content``, ``empty_instruction``.

**После полного пайплайна:** ``missing_text``; ``unsupported_record_schema`` — если не осталось ни одного
признака известной схемы (см. :func:`structural_row_has_recognized_payload`).

Сырые поля чата/инструкций переносятся в ``metadata`` (ключи ``raw_*``) и снимаются с верхнего уровня.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from agents.data_collection.text_unified_schema import scalar_cell_is_na

STASH_AND_DROP_COLUMNS: frozenset[str] = frozenset(
    {
        "messages",
        "instructions",
        "instruction",
        "prompt",
        "input",
        "question",
        "outputs",
        "output",
        "response",
        "answer",
        "completion",
    }
)

SOURCE_TEXT_CANDIDATES: tuple[str, ...] = (
    "text",
    "instructions",
    "instruction",
    "prompt",
    "input",
    "question",
)

TARGET_TEXT_CANDIDATES: tuple[str, ...] = (
    "target_text",
    "outputs",
    "output",
    "response",
    "answer",
    "completion",
)

USER_ROLES: frozenset[str] = frozenset({"user", "human", "question"})
ASSISTANT_ROLES: frozenset[str] = frozenset(
    {"assistant", "gpt", "model", "agent", "answer"},
)


@dataclass
class CanonicalNormalizationSummary:
    """Per-dataframe stats for collect logging and tests."""

    source_key: str
    rows_in: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    rows_after: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_key": self.source_key,
            "rows_in": self.rows_in,
            "rows_after": self.rows_after,
            "dropped": dict(self.dropped),
        }


def _cell_strip(value: Any) -> str:
    if value is None or scalar_cell_is_na(value):
        return ""
    return str(value).strip()


def _instruction_schema_columns_present(row: pd.Series) -> bool:
    return any(c in row.index for c in SOURCE_TEXT_CANDIDATES if c != "text")


def structural_row_has_recognized_payload(row: pd.Series) -> bool:
    """True if the row still carries recognizable raw fields (for ``unsupported_record_schema``)."""

    if "messages" in row.index and _messages_cell_nonempty(row.get("messages")):
        return True
    if _cell_strip(row["text"]) if "text" in row.index else "":
        return True
    if "target_text" in row.index and _cell_strip(row.get("target_text")):
        return True
    for col in STASH_AND_DROP_COLUMNS:
        if col not in row.index or col == "messages":
            continue
        val = row[col]
        if scalar_cell_is_na(val):
            continue
        if isinstance(val, (list, tuple)):
            if len(val) > 0:
                return True
            continue
        if isinstance(val, dict) and len(val) > 0:
            return True
        if _cell_strip(val):
            return True
    return False


def _messages_cell_nonempty(value: Any) -> bool:
    """True iff ``messages`` should trigger the chat branch (non-empty payload)."""

    if value is None or scalar_cell_is_na(value):
        return False
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    if isinstance(value, dict):
        return len(value) > 0
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in ("null", "none", "[]", "{}"):
            return False
        try:
            parsed = json.loads(s)
            if isinstance(parsed, (list, tuple)):
                return len(parsed) > 0
            if isinstance(parsed, dict):
                return len(parsed) > 0
            return True
        except json.JSONDecodeError:
            return bool(s)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return bool(value.size)
    except ImportError:
        pass
    if not isinstance(value, (str, bytes)) and hasattr(value, "__len__"):
        try:
            return len(value) > 0
        except Exception:
            pass
    return True


def _unwrap_ndarray(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.dtype == object and value.size == 1:
                return value.item()
            return value.tolist()
    except ImportError:
        pass
    return value


def _coerce_message_dicts(value: Any) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Return ``(list of dict messages, error_reason)``.

    *error_reason*: ``malformed_messages`` | ``missing_messages`` | ``None``.
    """

    value = _unwrap_ndarray(value)
    if value is None or scalar_cell_is_na(value):
        return None, "missing_messages"

    if isinstance(value, dict):
        return [value], None

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None, "missing_messages"
        out: list[dict[str, Any]] = []
        for el in value:
            el = _unwrap_ndarray(el)
            if isinstance(el, dict):
                out.append(el)
        return out, None

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None, "missing_messages"
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return None, "malformed_messages"
        if isinstance(parsed, dict):
            return [parsed], None
        if isinstance(parsed, (list, tuple)):
            if len(parsed) == 0:
                return None, "missing_messages"
            out: list[dict[str, Any]] = []
            for x in parsed:
                u = _unwrap_ndarray(x)
                if isinstance(u, dict):
                    out.append(u)
            if not out and len(parsed) > 0:
                return None, "unsupported_message_schema"
            if not out:
                return None, "missing_messages"
            return out, None
        return None, "malformed_messages"

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return _coerce_message_dicts(value.tolist())
    except ImportError:
        pass

    return None, "malformed_messages"


def _message_role_raw(item: dict[str, Any]) -> str:
    r = item.get("role")
    if r is None or scalar_cell_is_na(r):
        r = item.get("from")
    if r is None or scalar_cell_is_na(r):
        r = item.get("speaker")
    return str(r).strip().lower() if r is not None and not scalar_cell_is_na(r) else ""


def _message_body_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if content is None or scalar_cell_is_na(content):
        content = item.get("value")
    if content is None or scalar_cell_is_na(content):
        content = item.get("text")
    if content is None or scalar_cell_is_na(content):
        return ""

    content = _unwrap_ndarray(content)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for chunk in content:
            chunk = _unwrap_ndarray(chunk)
            if isinstance(chunk, dict):
                t = str(chunk.get("type", "")).lower()
                if chunk.get("text") is not None:
                    chunks.append(str(chunk.get("text", "")).strip())
                elif t == "text" and chunk.get("content") is not None:
                    chunks.append(str(chunk.get("content", "")).strip())
            elif chunk is not None and not scalar_cell_is_na(chunk):
                chunks.append(str(chunk).strip())
        return "\n".join(c for c in chunks if c)
    return str(content).strip()


def parse_messages_to_texts(messages_value: Any) -> tuple[str | None, str | None, str | None]:
    """Extract ``(user_text, assistant_text, drop_reason)`` from a chat payload.

    ``assistant_text`` may be ``None`` / empty — row is still valid if ``user_text`` is non-empty.
    """

    items, err = _coerce_message_dicts(messages_value)
    if err:
        return None, None, err
    assert items is not None
    if len(items) == 0:
        return None, None, "missing_messages"

    user_parts: list[str] = []
    asst_parts: list[str] = []
    any_recognized_line = False

    for item in items:
        role = _message_role_raw(item)
        body = _message_body_text(item)
        if not body:
            continue
        any_recognized_line = True
        if role in USER_ROLES:
            user_parts.append(body)
        elif role in ASSISTANT_ROLES:
            asst_parts.append(body)

    if not user_parts:
        if not any_recognized_line:
            return None, None, "unsupported_message_schema"
        return None, None, "empty_user_content"

    user_joined = "\n\n".join(user_parts)
    asst_joined = "\n\n".join(asst_parts) if asst_parts else ""
    return user_joined, (asst_joined if asst_joined else None), None


def _stash_raw_row_fields(row: pd.Series) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    for col in STASH_AND_DROP_COLUMNS:
        if col in row.index and not scalar_cell_is_na(row[col]):
            if col == "messages" and not _messages_cell_nonempty(row[col]):
                continue
            patch[f"raw_{col}"] = row[col]
    return patch


def first_instruction_source_target(
    row: pd.Series,
) -> tuple[str | None, str | None, dict[str, Any]]:
    """First non-empty source text and first non-empty target (instruction-style)."""

    meta = _stash_raw_row_fields(row)
    src_val: str | None = None
    for col in SOURCE_TEXT_CANDIDATES:
        if col not in row.index:
            continue
        s = _cell_strip(row.get(col))
        if s:
            src_val = s
            break
    tgt_val: str | None = None
    for col in TARGET_TEXT_CANDIDATES:
        if col not in row.index:
            continue
        s = _cell_strip(row.get(col))
        if s:
            tgt_val = s
            break
    return src_val, tgt_val, meta


def route_and_extract_row(row: pd.Series) -> tuple[str | None, str | None, dict[str, Any], str | None]:
    """Structural routing: direct text → chat → instruction → none (title/body later)."""

    meta_patch: dict[str, Any] = {}
    text_now = _cell_strip(row["text"]) if "text" in row.index else ""

    if text_now:
        tt0 = _cell_strip(row["target_text"]) if "target_text" in row.index else ""
        patch = _stash_raw_row_fields(row)
        return text_now, (tt0 or None), patch, None

    if "messages" in row.index:
        mv = row.get("messages")
        if _messages_cell_nonempty(mv):
            ut, at, err = parse_messages_to_texts(mv)
            mp = _stash_raw_row_fields(row)
            mp.setdefault("raw_messages", mv)
            if err:
                return None, None, mp, err
            return ut, at, mp, None

    src, tgt, mp_inst = first_instruction_source_target(row)
    if src:
        return src, tgt, mp_inst, None

    if _instruction_schema_columns_present(row):
        return None, None, mp_inst, "empty_instruction"

    if structural_row_has_recognized_payload(row):
        mp = _stash_raw_row_fields(row)
        return None, None, mp, "unsupported_record_schema"

    return None, None, {}, None


def validate_canonical_dataframe(df: pd.DataFrame, *, require_non_empty_text: bool = True) -> list[str]:
    issues: list[str] = []
    if df.empty:
        issues.append("canonical: empty dataframe")
        return issues
    if "text" not in df.columns:
        issues.append("canonical: missing text column")
        return issues
    ok = df["text"].notna() & (df["text"].astype(str).str.strip().ne(""))
    if require_non_empty_text and not bool(ok.any()):
        issues.append("canonical: no non-empty text values")
    if "source" in df.columns and df["source"].isna().all():
        issues.append("canonical: source all null")
    if "collected_at" in df.columns and df["collected_at"].isna().all():
        issues.append("canonical: collected_at all null")
    return issues


def apply_chat_instruction_extraction(
    df: pd.DataFrame,
    *,
    source_key: str = "unknown",
) -> tuple[pd.DataFrame, CanonicalNormalizationSummary]:
    summary = CanonicalNormalizationSummary(source_key=source_key, rows_in=len(df))
    if df.empty:
        summary.rows_after = 0
        return df.copy(), summary

    work = df.copy()
    if "target_text" not in work.columns:
        work["target_text"] = pd.NA

    new_text: list[Any] = []
    new_target: list[Any] = []
    meta_patches: list[dict[str, Any]] = []
    hard_drops: list[str | None] = []

    index_list = list(work.index)
    n = len(index_list)
    for pos in range(n):
        row = work.iloc[pos]
        txt, tgt, patch, hd = route_and_extract_row(row)
        meta_patches.append(patch)
        hard_drops.append(hd)
        if hd:
            new_text.append(pd.NA)
            new_target.append(pd.NA)
        else:
            new_text.append(txt if txt else pd.NA)
            new_target.append(tgt if tgt is not None else pd.NA)

    work["text"] = new_text
    work["target_text"] = new_target

    if "metadata" not in work.columns:
        work["metadata"] = "{}"
    new_meta: list[str] = []
    for pos in range(n):
        original = work.iloc[pos]["metadata"]
        new_meta.append(_merge_json_metadata(str(original), meta_patches[pos]))
    work["metadata"] = new_meta

    to_drop: list[Any] = []
    for pos in range(n):
        reason = hard_drops[pos]
        if reason:
            summary.dropped[reason] = summary.dropped.get(reason, 0) + 1
            to_drop.append(index_list[pos])

    if to_drop:
        work = work.drop(index=to_drop).reset_index(drop=True)

    summary.rows_after = len(work)

    drop_cols = [c for c in work.columns if c in STASH_AND_DROP_COLUMNS]
    if drop_cols:
        work = work.drop(columns=drop_cols, errors="ignore")

    return work, summary


def _merge_json_metadata(existing: str, patch: dict[str, Any]) -> str:
    base: dict[str, Any] = {}
    if existing and str(existing).strip():
        try:
            loaded = json.loads(str(existing))
            if isinstance(loaded, dict):
                base = loaded
        except json.JSONDecodeError:
            base = {"_unparsed_metadata": str(existing)}
    for key, val in patch.items():
        if val is None:
            continue
        if key not in base or base.get(key) in (None, {}, []):
            base[key] = val
    return json.dumps(base, sort_keys=True, ensure_ascii=False, default=str)


def finalize_drop_empty_text(
    df: pd.DataFrame,
    summary: CanonicalNormalizationSummary | None,
    *,
    reason_key: str = "missing_text",
) -> tuple[pd.DataFrame, int]:
    if df.empty or "text" not in df.columns:
        return df, 0
    nonempty = df["text"].notna() & (df["text"].astype(str).str.strip().ne(""))
    dropped_n = int((~nonempty).sum())
    out = df.loc[nonempty].reset_index(drop=True)
    if summary is not None and dropped_n:
        summary.dropped[reason_key] = summary.dropped.get(reason_key, 0) + dropped_n
        summary.rows_after = len(out)
    elif summary is not None:
        summary.rows_after = len(out)
    return out, dropped_n


def merge_canonical_summaries(summaries: list[CanonicalNormalizationSummary]) -> dict[str, Any]:
    out: dict[str, Any] = {"per_source": [s.to_dict() for s in summaries], "totals": {}}
    tot_drop: dict[str, int] = {}
    for s in summaries:
        for k, v in s.dropped.items():
            tot_drop[k] = tot_drop.get(k, 0) + v
    out["totals"]["dropped"] = tot_drop
    return out


def normalize_record(row: pd.Series) -> dict[str, Any]:
    t, tt, patch, drop = route_and_extract_row(row)
    return {
        "text": t,
        "target_text": tt,
        "metadata_patch": patch,
        "drop_reason": drop,
    }


def to_canonical_text(row: pd.Series) -> tuple[str | None, str | None]:
    t, tt, _, drop = route_and_extract_row(row)
    if drop:
        return None, None
    return t, (tt if tt is not None else None)


def normalize_dataframe(
    df: pd.DataFrame,
    *,
    source_key: str = "unknown",
    fill_title_body: bool = True,
) -> tuple[pd.DataFrame, CanonicalNormalizationSummary]:
    out, summary = apply_chat_instruction_extraction(df, source_key=source_key)
    if fill_title_body:
        from agents.data_collection.text_unified_schema import fill_missing_text_from_title_body

        out = fill_missing_text_from_title_body(out)
    out, _ = finalize_drop_empty_text(out, summary)
    return out, summary
