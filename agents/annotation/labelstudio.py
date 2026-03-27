from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from agents.annotation.schemas import AnnotationTaskConfig


def build_labelstudio_config(task_config: AnnotationTaskConfig) -> str:
    """Build a minimal Label Studio config for text classification."""

    choice_lines = "\n".join(
        f'    <Choice value="{html.escape(str(label), quote=True)}" />'
        for label in task_config.labels
    )
    header = html.escape(
        task_config.name.replace("_", " ").strip() or "Text classification review",
        quote=True,
    )
    return (
        "<View>\n"
        f'  <Header value="{header}" />\n'
        '  <Text name="text" value="$text" />\n'
        '  <Choices name="label" toName="text" choice="single-radio">\n'
        f"{choice_lines}\n"
        "  </Choices>\n"
        "</View>\n"
    )


def build_labelstudio_tasks(
    df: pd.DataFrame,
    task_config: AnnotationTaskConfig,
    text_column: str,
    id_column: str,
    include_predictions: bool = True,
) -> list[dict[str, Any]]:
    """Convert a dataframe into Label Studio import tasks (JSON list).

    Exports a ``data`` object compatible with LS “Import → JSON” for text classification.
    Join keys for round-trip: ``data.annotation_id`` and ``data.source_id`` (same value).
    Top-level task ``id`` is omitted so Label Studio assigns stable internal task ids.
    """

    tasks: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        aid = _json_value(row.get(id_column))
        task: dict[str, Any] = {
            "data": {
                "text": _json_value(row.get(text_column)),
                "annotation_id": aid,
                "source_id": aid,
                "task": task_config.name,
                "needs_review": _json_value(row.get("needs_review")),
                "review_reason": _json_value(row.get("review_reason")),
            },
        }
        if include_predictions and isinstance(row.get("auto_label"), str):
            task["predictions"] = [
                {
                    "model_version": _json_value(row.get("annotator_version") or task_config.model_version),
                    "score": _json_value(row.get("confidence")),
                    "result": [
                        {
                            "id": f"pred-{_json_value(row.get(id_column))}",
                            "from_name": "label",
                            "to_name": "text",
                            "type": "choices",
                            "value": {"choices": [row.get("auto_label")]},
                        }
                    ],
                    "meta": {
                        "class_scores": _json_value(row.get("class_scores")),
                        "margin": _json_value(row.get("margin")),
                        "entropy": _json_value(row.get("entropy")),
                        "review_reason": _json_value(row.get("review_reason")),
                    },
                }
            ]
        tasks.append(task)
    return tasks


def _records_from_export(export: list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(export, dict):
        if "tasks" in export and isinstance(export["tasks"], list):
            return export["tasks"]
        raise ValueError(
            "Label Studio dict export must contain a 'tasks' list, or pass a JSON array of tasks."
        )
    return export


def _resolve_annotation_id(item: dict[str, Any]) -> str | None:
    data = item.get("data") or {}
    aid = data.get("annotation_id")
    if aid is None or (isinstance(aid, float) and math.isnan(aid)):
        aid = data.get("source_id")
    if aid is None and item.get("id") is not None:
        aid = item.get("id")
    if aid is None:
        return None
    s = str(aid).strip()
    return s or None


def _pick_best_annotation(annotations: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not annotations:
        return None
    usable = [a for a in annotations if not a.get("was_canceled")]
    if not usable:
        return None

    def sort_key(a: dict[str, Any]) -> tuple[int, str]:
        ann_id = a.get("id")
        try:
            aid = int(ann_id) if ann_id is not None else -1
        except (TypeError, ValueError):
            aid = -1
        return (aid, str(a.get("created_at") or ""))

    usable.sort(key=sort_key, reverse=True)
    return usable[0]


def _extract_choice_label(result: list[dict[str, Any]] | None) -> str | None:
    for r in result or []:
        if r.get("type") != "choices":
            continue
        if r.get("from_name") not in {"label", "choice"}:
            continue
        choices = (r.get("value") or {}).get("choices") or []
        if choices:
            return str(choices[0]).strip()
    return None


def labelstudio_export_to_human_labels(
    export: list[dict[str, Any]] | dict[str, Any],
    *,
    allowed_labels: set[str] | frozenset[str] | None = None,
    on_duplicate_id: str = "error",
) -> pd.DataFrame:
    """Parse Label Studio JSON export into rows joinable on ``annotation_id``.

    Columns:
    - ``annotation_id``, ``human_label`` (required for merge)
    - ``review_status`` (``labeled`` when a choice was extracted)
    - ``ls_task_id``, ``ls_annotation_id``, ``ls_annotator_id``, ``ls_completed_at`` (metadata)

    Parameters
    ----------
    allowed_labels
        If set, any human label not in this set raises ``ValueError``.
    on_duplicate_id
        ``error`` (default) or ``last`` if the same ``annotation_id`` appears in multiple tasks.
    """

    records = _records_from_export(export)
    if not records:
        raise ValueError("Label Studio export is empty (no tasks).")

    if on_duplicate_id not in {"error", "last"}:
        raise ValueError("on_duplicate_id must be 'error' or 'last'.")

    by_id: dict[str, dict[str, Any]] = {}
    missing_id_tasks = 0

    for item in records:
        aid = _resolve_annotation_id(item)
        if aid is None:
            missing_id_tasks += 1
            continue

        chosen = _pick_best_annotation(list(item.get("annotations") or []))
        if chosen is None:
            continue

        label = _extract_choice_label(chosen.get("result"))
        if not label:
            continue

        if allowed_labels is not None and label not in allowed_labels:
            raise ValueError(
                f"Unknown human label {label!r} for annotation_id={aid!r}. "
                f"Allowed: {sorted(allowed_labels)}. Fix LS choices or annotation.labels in config."
            )

        row = {
            "annotation_id": aid,
            "human_label": label,
            "review_status": "labeled",
            "ls_task_id": _json_value(item.get("id")),
            "ls_annotation_id": _json_value(chosen.get("id")),
            "ls_annotator_id": _json_value(chosen.get("completed_by")),
            "ls_completed_at": _json_value(chosen.get("created_at")),
        }

        if aid in by_id:
            if on_duplicate_id == "error":
                raise ValueError(
                    f"Duplicate annotation_id {aid!r} in Label Studio export "
                    "(multiple tasks map to the same pipeline id)."
                )
        by_id[aid] = row

    if not by_id:
        msg = (
            "No completed annotations with choices found in Label Studio export. "
            "Export tasks after labeling (JSON); ensure the tagging interface uses "
            "Choices with from_name 'label' matching label_config.xml."
        )
        if missing_id_tasks:
            msg += f" Tasks skipped (missing annotation_id/source_id in data): {missing_id_tasks}."
        raise ValueError(msg)

    return pd.DataFrame(list(by_id.values()))


def read_labelstudio_export_path(
    path: Path,
    *,
    allowed_labels: set[str] | frozenset[str] | None = None,
    on_duplicate_id: str = "error",
) -> pd.DataFrame:
    """Load a Label Studio ``.json`` export file and return human label corrections."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        if isinstance(raw, dict):
            return labelstudio_export_to_human_labels(
                raw,
                allowed_labels=allowed_labels,
                on_duplicate_id=on_duplicate_id,
            )
        raise ValueError(f"Unexpected Label Studio JSON root type: {type(raw).__name__}")
    return labelstudio_export_to_human_labels(
        raw,
        allowed_labels=allowed_labels,
        on_duplicate_id=on_duplicate_id,
    )


def _json_value(value: Any) -> Any:
    """Convert pandas and Python values into JSON-safe objects."""

    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value
