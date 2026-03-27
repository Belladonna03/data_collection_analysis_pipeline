from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QualityReport:
    """Structured quality assessment report."""

    generated_at: str
    row_count: int
    column_count: int
    text_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    missing: dict[str, Any] = field(default_factory=dict)
    duplicates: dict[str, Any] = field(default_factory=dict)
    outliers: dict[str, Any] = field(default_factory=dict)
    class_imbalance: dict[str, Any] | None = None
    recommendations: list[str] = field(default_factory=list)
    # Text / forum-oriented checks (empty/short, PII, near-dup, language); empty dict when disabled.
    text_quality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass
class ComparisonReport:
    """Comparison of dataset state before and after cleaning."""

    generated_at: str
    before_rows: int
    after_rows: int
    metrics: list[dict[str, Any]] = field(default_factory=list)
    class_distribution_before: dict[str, float] = field(default_factory=dict)
    class_distribution_after: dict[str, float] = field(default_factory=dict)
    markdown_table: str = ""
    # Extended fields (pipeline / human review); defaults keep older callers compatible.
    rows_removed: int = 0
    per_column_missing: dict[str, dict[str, Any]] = field(default_factory=dict)
    duplicates_breakdown: dict[str, Any] = field(default_factory=dict)
    outliers_per_feature: dict[str, Any] = field(default_factory=dict)
    summary_markdown: str = ""
    duplicates_breakdown_before: dict[str, Any] = field(default_factory=dict)
    duplicates_breakdown_after: dict[str, Any] = field(default_factory=dict)
    per_feature_outliers_before: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_feature_outliers_after: dict[str, dict[str, Any]] = field(default_factory=dict)
    row_count_removed: int = 0
    kept_row_count: int | None = None
    removed_row_count: int | None = None
    markdown_sections: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass
class QualityStageResult:
    """Structured result for orchestration-level stage execution."""

    status: str
    artifacts: dict[str, str] = field(default_factory=dict)
    quality_stage_status_json: str = ""
    decision_template_json: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)
