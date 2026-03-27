from __future__ import annotations

from typing import Any

import pandas as pd

from agents.annotation.schemas import AnnotationTaskConfig


def build_annotation_spec(
    df: pd.DataFrame,
    task_config: AnnotationTaskConfig,
    text_column: str,
    threshold: float,
    margin_threshold: float,
) -> str:
    """Build a markdown annotation spec from auto-labeled data."""

    sections = [
        f"# Annotation Spec: {task_config.name}",
        "",
        "## Task Name",
        task_config.name,
        "",
        "## Unit Of Annotation",
        task_config.unit_of_annotation,
        "",
        "## Classes And Definitions",
    ]

    for label in task_config.labels:
        definition = task_config.label_definitions[label]
        sections.append(f"### {label}")
        sections.append(definition.description)
        sections.append("")
        sections.append("Decision rules:")
        for rule in definition.decision_rules:
            sections.append(f"- {rule}")
        sections.append("")
        sections.append("Examples:")
        for index, example in enumerate(
            _select_examples_for_label(df, label, text_column, task_config, threshold),
            start=1,
        ):
            sections.append(f"{index}. {example}")
        sections.append("")

    sections.extend(
        [
            "## Boundary Cases",
            "Use extra care for examples that look partly benign and partly risky.",
        ]
    )
    for bullet in task_config.boundary_case_guidance:
        sections.append(f"- {bullet}")
    for example in _select_boundary_examples(df, text_column, margin_threshold):
        sections.append(f"- Example: {example}")
    sections.append("")

    sections.append("## Typical Annotator Mistakes")
    for bullet in task_config.annotator_mistakes:
        sections.append(f"- {bullet}")
    sections.append("")

    sections.append("## What To Do When In Doubt")
    for bullet in task_config.doubt_guidance:
        sections.append(f"- {bullet}")
    sections.append("- Escalate unresolved items to the human review queue instead of forcing a confident label.")
    sections.append("")

    return "\n".join(sections).strip() + "\n"


def _select_examples_for_label(
    df: pd.DataFrame,
    label: str,
    text_column: str,
    task_config: AnnotationTaskConfig,
    threshold: float,
) -> list[str]:
    """Select at least three examples for a given class."""

    if df.empty or text_column not in df.columns or "auto_label" not in df.columns:
        return list(task_config.label_definitions[label].canonical_examples[:3])

    candidates = df.loc[df["auto_label"] == label].copy()
    if candidates.empty:
        return list(task_config.label_definitions[label].canonical_examples[:3])

    if "confidence" in candidates.columns:
        candidates = candidates.sort_values("confidence", ascending=False)
    if "needs_review" in candidates.columns:
        preferred = candidates.loc[
            (~candidates["needs_review"].fillna(False))
            & (candidates.get("confidence", pd.Series(index=candidates.index, dtype=float)).fillna(0.0) >= threshold)
        ]
        if not preferred.empty:
            dedupe_subset = [text_column] if text_column in candidates.columns else None
            candidates = pd.concat([preferred, candidates]).drop_duplicates(subset=dedupe_subset)

    examples = [
        _format_example_row(row, text_column)
        for _, row in candidates.head(3).iterrows()
    ]
    fallback_examples = task_config.label_definitions[label].canonical_examples
    for fallback in fallback_examples:
        if len(examples) >= 3:
            break
        examples.append(fallback)
    return examples[:3]


def _select_boundary_examples(
    df: pd.DataFrame,
    text_column: str,
    margin_threshold: float,
) -> list[str]:
    """Select boundary examples from review-worthy rows."""

    if df.empty or text_column not in df.columns:
        return []

    boundary_df = df.copy()
    if "needs_review" in boundary_df.columns:
        boundary_df = boundary_df.loc[boundary_df["needs_review"].fillna(False)]
    elif "margin" in boundary_df.columns:
        boundary_df = boundary_df.loc[boundary_df["margin"].fillna(1.0) < margin_threshold]

    if boundary_df.empty and "margin" in df.columns:
        boundary_df = df.loc[df["margin"].fillna(1.0).sort_values().index].head(5)

    if boundary_df.empty:
        return []

    ordered = boundary_df.sort_values(
        by=["confidence", "margin"],
        ascending=[True, True],
    ) if {"confidence", "margin"}.issubset(boundary_df.columns) else boundary_df
    return [
        _format_example_row(row, text_column, include_meta=True)
        for _, row in ordered.head(6).iterrows()
    ]


def _format_example_row(
    row: pd.Series,
    text_column: str,
    include_meta: bool = False,
) -> str:
    """Render one dataset row as a human-readable example string."""

    text_value = _truncate_text(row.get(text_column))
    label = row.get("auto_label")
    confidence = row.get("confidence")
    margin = row.get("margin")
    if not include_meta:
        return f'"{text_value}"'

    meta_parts: list[str] = []
    if isinstance(label, str) and label:
        meta_parts.append(f"label={label}")
    if confidence is not None and not pd.isna(confidence):
        meta_parts.append(f"confidence={float(confidence):.2f}")
    if margin is not None and not pd.isna(margin):
        meta_parts.append(f"margin={float(margin):.2f}")
    meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
    return f'"{text_value}"{meta}'


def _truncate_text(value: Any, limit: int = 220) -> str:
    """Return a compact one-line snippet."""

    text = "" if value is None else " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
