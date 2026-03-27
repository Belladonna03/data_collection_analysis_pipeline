from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LabelDefinition:
    """Human-facing definition of one annotation class."""

    name: str
    description: str
    decision_rules: list[str] = field(default_factory=list)
    canonical_examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass
class AnnotationTaskConfig:
    """Task configuration for one annotation task."""

    name: str
    modality: str
    unit_of_annotation: str
    labels: list[str]
    label_definitions: dict[str, LabelDefinition]
    text_column: str | None = None
    id_column: str | None = None
    threshold: float = 0.70
    margin_threshold: float = 0.12
    text_column_candidates: list[str] = field(default_factory=list)
    safe_keywords: list[str] = field(default_factory=list)
    benign_patterns: list[str] = field(default_factory=list)
    borderline_patterns: list[str] = field(default_factory=list)
    harmful_keywords: list[str] = field(default_factory=list)
    harmful_intent_patterns: list[str] = field(default_factory=list)
    safe_context_patterns: list[str] = field(default_factory=list)
    noise_patterns: list[str] = field(default_factory=list)
    annotator_mistakes: list[str] = field(default_factory=list)
    doubt_guidance: list[str] = field(default_factory=list)
    boundary_case_guidance: list[str] = field(default_factory=list)
    model_version: str = "annotation-text-v1"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["label_definitions"] = {
            name: definition.to_dict()
            for name, definition in self.label_definitions.items()
        }
        return payload


@dataclass
class TextLabelingResult:
    """Output of a text auto-labeling backend."""

    label: str
    confidence: float
    class_scores: dict[str, float]
    margin: float
    entropy: float
    backend_name: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)
