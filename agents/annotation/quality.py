from __future__ import annotations

import math
from collections import Counter

import pandas as pd

from agents.annotation.schemas import AnnotationTaskConfig


def compute_quality_metrics(
    df: pd.DataFrame,
    task_config: AnnotationTaskConfig,
    human_label_column: str = "human_label",
    threshold: float | None = None,
) -> dict[str, object]:
    """Compute quality metrics for auto labels and optional human labels."""

    total_rows = len(df)
    confidence_series = _numeric_series(df, "confidence")
    review_series = _bool_series(df, "needs_review")
    label_dist = _distribution(df["auto_label"]) if "auto_label" in df.columns else {}
    active_threshold = float(task_config.threshold if threshold is None else threshold)
    low_conf_share = float((confidence_series < active_threshold).mean()) if not confidence_series.empty else 0.0

    metrics: dict[str, object] = {
        "row_count": int(total_rows),
        "label_dist": label_dist,
        "confidence_mean": round(float(confidence_series.mean()), 6) if not confidence_series.empty else None,
        "confidence_median": round(float(confidence_series.median()), 6) if not confidence_series.empty else None,
        "low_conf_share": round(low_conf_share, 6),
        "needs_review_count": int(review_series.sum()) if not review_series.empty else 0,
        "agreement": None,
        "cohen_kappa": None,
        "confusion_matrix": None,
        "per_class_agreement": None,
    }

    if human_label_column not in df.columns or "auto_label" not in df.columns:
        return metrics

    paired = df[["auto_label", human_label_column]].dropna()
    paired = paired.loc[
        paired["auto_label"].astype(str).str.len() > 0
    ]
    paired = paired.loc[
        paired[human_label_column].astype(str).str.len() > 0
    ]
    if paired.empty:
        return metrics

    metrics["agreement"] = round(float((paired["auto_label"] == paired[human_label_column]).mean()), 6)
    metrics["cohen_kappa"] = round(_cohen_kappa(paired["auto_label"], paired[human_label_column]), 6)
    metrics["confusion_matrix"] = _confusion_matrix(
        paired["auto_label"],
        paired[human_label_column],
        labels=_ordered_labels(task_config, paired, human_label_column),
    )
    metrics["per_class_agreement"] = _per_class_agreement(
        paired["auto_label"],
        paired[human_label_column],
        labels=_ordered_labels(task_config, paired, human_label_column),
    )
    return metrics


def render_annotation_report(
    metrics: dict[str, object],
    task_config: AnnotationTaskConfig,
) -> str:
    """Render markdown report for the annotation stage."""

    lines = [
        f"# Annotation Report: {task_config.name}",
        "",
        "## Dataset Summary",
        f"- rows: {metrics.get('row_count', 0)}",
        f"- labels: {task_config.labels}",
        "",
        "## Auto-Label Distribution",
    ]
    label_dist = metrics.get("label_dist") or {}
    if label_dist:
        for label, value in label_dist.items():
            lines.append(f"- {label}: {value:.4f}")
    else:
        lines.append("- No labels available.")
    lines.extend(
        [
            "",
            "## Confidence Stats",
            f"- mean: {_format_optional_float(metrics.get('confidence_mean'))}",
            f"- median: {_format_optional_float(metrics.get('confidence_median'))}",
            f"- low confidence share: {_format_optional_float(metrics.get('low_conf_share'))}",
            f"- needs review count: {metrics.get('needs_review_count', 0)}",
            "",
            "## Human Agreement",
            f"- agreement: {_format_optional_float(metrics.get('agreement'))}",
            f"- cohen_kappa: {_format_optional_float(metrics.get('cohen_kappa'))}",
        ]
    )

    per_class = metrics.get("per_class_agreement") or {}
    if per_class:
        lines.append("")
        lines.append("## Per-Class Agreement")
        for label, value in per_class.items():
            lines.append(f"- {label}: {_format_optional_float(value)}")

    confusion = metrics.get("confusion_matrix") or {}
    if confusion:
        lines.append("")
        lines.append("## Confusion Matrix")
        for human_label, auto_counts in confusion.items():
            lines.append(f"- human={human_label}: {auto_counts}")

    return "\n".join(lines).strip() + "\n"


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series or an empty series."""

    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a boolean series or an empty series."""

    if column not in df.columns:
        return pd.Series(dtype=bool)
    return df[column].fillna(False).astype(bool)


def _distribution(series: pd.Series) -> dict[str, float]:
    """Return normalized label distribution."""

    normalized = series.dropna().astype(str)
    if normalized.empty:
        return {}
    value_counts = normalized.value_counts(normalize=True)
    return {
        str(label): round(float(score), 6)
        for label, score in value_counts.items()
    }


def _ordered_labels(
    task_config: AnnotationTaskConfig,
    paired: pd.DataFrame,
    human_label_column: str,
) -> list[str]:
    """Return label order from task config plus unexpected labels."""

    labels = list(task_config.labels)
    observed = sorted(
        set(paired["auto_label"].astype(str)).union(set(paired[human_label_column].astype(str)))
    )
    for label in observed:
        if label not in labels:
            labels.append(label)
    return labels


def _cohen_kappa(auto_labels: pd.Series, human_labels: pd.Series) -> float:
    """Compute Cohen's kappa without external dependencies."""

    total = len(auto_labels)
    if total == 0:
        return 0.0
    agreement = float((auto_labels == human_labels).mean())
    auto_distribution = Counter(auto_labels.astype(str))
    human_distribution = Counter(human_labels.astype(str))
    labels = set(auto_distribution) | set(human_distribution)
    expected = 0.0
    for label in labels:
        expected += (auto_distribution[label] / total) * (human_distribution[label] / total)
    denominator = 1.0 - expected
    if math.isclose(denominator, 0.0):
        return 1.0 if math.isclose(agreement, 1.0) else 0.0
    return (agreement - expected) / denominator


def _confusion_matrix(
    auto_labels: pd.Series,
    human_labels: pd.Series,
    labels: list[str],
) -> dict[str, dict[str, int]]:
    """Build a confusion matrix keyed by human label then auto label."""

    matrix = {
        human_label: {auto_label: 0 for auto_label in labels}
        for human_label in labels
    }
    for auto_label, human_label in zip(auto_labels.astype(str), human_labels.astype(str)):
        matrix.setdefault(human_label, {})
        matrix[human_label].setdefault(auto_label, 0)
        matrix[human_label][auto_label] += 1
    return matrix


def _per_class_agreement(
    auto_labels: pd.Series,
    human_labels: pd.Series,
    labels: list[str],
) -> dict[str, float]:
    """Return agreement conditioned on the human label."""

    agreement: dict[str, float] = {}
    for label in labels:
        mask = human_labels.astype(str) == label
        if not mask.any():
            continue
        agreement[label] = round(float((auto_labels[mask].astype(str) == human_labels[mask].astype(str)).mean()), 6)
    return agreement


def _format_optional_float(value: object) -> str:
    """Format nullable floats for markdown."""

    if value is None:
        return "None"
    return f"{float(value):.4f}"
