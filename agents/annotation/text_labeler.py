from __future__ import annotations

import math
import re
from dataclasses import dataclass

from agents.annotation.schemas import AnnotationTaskConfig, TextLabelingResult
from agents.annotation.task_configs import LEGACY_SAFETY_LABELS


DEFAULT_ZERO_SHOT_MODEL = "facebook/bart-large-mnli"


class BaseTextLabeler:
    """Base class for text auto-labeling backends."""

    backend_name = "base"

    def __init__(self, task_config: AnnotationTaskConfig) -> None:
        self.task_config = task_config

    def label_text(self, text: object) -> TextLabelingResult:
        """Label one text sample."""

        raise NotImplementedError


class TransformersZeroShotTextLabeler(BaseTextLabeler):
    """Optional transformers-based zero-shot classifier."""

    backend_name = "transformers_zero_shot"

    def __init__(
        self,
        task_config: AnnotationTaskConfig,
        model_name: str = DEFAULT_ZERO_SHOT_MODEL,
    ) -> None:
        super().__init__(task_config)
        from transformers import pipeline

        self.model_name = model_name
        self.pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
        )
        self.candidate_labels = list(task_config.labels)
        self.hypothesis_template = "This request is {}."

    def label_text(self, text: object) -> TextLabelingResult:
        """Label text via a zero-shot classification pipeline."""

        normalized_text = _normalize_text(text)
        if not normalized_text:
            return _uncertain_result(
                self.task_config,
                self.backend_name,
                warnings=["empty_text"],
            )

        response = self.pipeline(
            normalized_text,
            candidate_labels=self.candidate_labels,
            hypothesis_template=self.hypothesis_template,
            multi_label=False,
        )
        scores = {
            label: float(score)
            for label, score in zip(response["labels"], response["scores"])
        }
        ordered_scores = _normalize_scores(
            {label: scores.get(label, 0.0) for label in self.task_config.labels}
        )
        return _build_result(
            class_scores=ordered_scores,
            backend_name=self.backend_name,
            warnings=_text_warnings(self.task_config, normalized_text),
        )


class RuleBasedTextLabeler(BaseTextLabeler):
    """Deterministic fallback labeler for text safety classification."""

    backend_name = "rule_based"

    def label_text(self, text: object) -> TextLabelingResult:
        """Label text using task-specific heuristics."""

        normalized_text = _normalize_text(text)
        warnings = _text_warnings(self.task_config, normalized_text)
        if not normalized_text:
            return _uncertain_result(
                self.task_config,
                self.backend_name,
                warnings=warnings or ["empty_text"],
            )

        if frozenset(self.task_config.labels) != LEGACY_SAFETY_LABELS:
            scores = {label: 1.0 for label in self.task_config.labels}
            return _build_result(
                _normalize_scores(scores),
                backend_name=self.backend_name,
                warnings=warnings,
            )

        lowered_text = normalized_text.lower()
        unsafe_hits = _count_keyword_hits(lowered_text, self.task_config.harmful_keywords)
        harmful_intent_hits = _count_regex_hits(
            lowered_text,
            self.task_config.harmful_intent_patterns,
        )
        safe_context_hits = _count_regex_hits(
            lowered_text,
            self.task_config.safe_context_patterns,
        )
        borderline_hits = _count_regex_hits(
            lowered_text,
            self.task_config.borderline_patterns,
        )
        benign_hits = _count_regex_hits(lowered_text, self.task_config.benign_patterns)
        safe_keyword_hits = _count_keyword_hits(lowered_text, self.task_config.safe_keywords)

        raw_scores = {
            "safe": 1.0,
            "borderline_safe": 1.0,
            "unsafe": 1.0,
        }

        raw_scores["unsafe"] += unsafe_hits * 1.35
        raw_scores["unsafe"] += harmful_intent_hits * 2.40
        raw_scores["unsafe"] += 0.50 if _looks_like_direct_request(lowered_text) and unsafe_hits else 0.0

        raw_scores["borderline_safe"] += borderline_hits * 1.65
        raw_scores["borderline_safe"] += safe_context_hits * 1.40
        raw_scores["borderline_safe"] += min(unsafe_hits, 2) * 0.55 if safe_context_hits else 0.0
        raw_scores["borderline_safe"] += 0.60 if "?" in normalized_text and unsafe_hits else 0.0

        raw_scores["safe"] += benign_hits * 1.25
        raw_scores["safe"] += safe_keyword_hits * 0.70
        raw_scores["safe"] += 1.05 if unsafe_hits == 0 and harmful_intent_hits == 0 else 0.0

        if safe_context_hits:
            raw_scores["unsafe"] = max(0.25, raw_scores["unsafe"] - safe_context_hits * 0.90)
        if warnings:
            raw_scores["borderline_safe"] += 0.75
            raw_scores["safe"] = max(0.25, raw_scores["safe"] - 0.30)

        class_scores = _normalize_scores(raw_scores)
        return _build_result(
            class_scores=class_scores,
            backend_name=self.backend_name,
            warnings=warnings,
        )


@dataclass
class _AutoBackend:
    """Wrapper that tries zero-shot first and falls back to rules."""

    primary: BaseTextLabeler | None
    fallback: BaseTextLabeler

    @property
    def backend_name(self) -> str:
        """Return the active backend label."""

        if self.primary is not None:
            return f"{self.primary.backend_name}+fallback"
        return self.fallback.backend_name

    def label_text(self, text: object) -> TextLabelingResult:
        """Label text using the available backend chain."""

        if self.primary is not None:
            try:
                return self.primary.label_text(text)
            except Exception:
                pass
        return self.fallback.label_text(text)


def build_text_labeler(
    task_config: AnnotationTaskConfig,
    backend: str = "auto",
    model_name: str = DEFAULT_ZERO_SHOT_MODEL,
) -> BaseTextLabeler | _AutoBackend:
    """Build a text labeler with an optional zero-shot backend."""

    normalized_backend = backend.strip().lower()
    fallback = RuleBasedTextLabeler(task_config)
    if normalized_backend == "rule_based":
        return fallback

    try:
        zero_shot = TransformersZeroShotTextLabeler(task_config, model_name=model_name)
    except Exception:
        if normalized_backend == "zero_shot":
            return fallback
        return _AutoBackend(primary=None, fallback=fallback)

    if normalized_backend == "zero_shot":
        return zero_shot
    return _AutoBackend(primary=zero_shot, fallback=fallback)


def _normalize_text(text: object) -> str:
    """Convert arbitrary input into normalized text."""

    if text is None:
        return ""
    value = str(text)
    normalized_value = re.sub(r"\s+", " ", value).strip()
    if normalized_value.lower() in {"nan", "none", "null"}:
        return ""
    return normalized_value


def _text_warnings(task_config: AnnotationTaskConfig, text: str) -> list[str]:
    """Return warnings for problematic text."""

    warnings: list[str] = []
    if not text:
        warnings.append("empty_text")
        return warnings
    if len(text) < 3:
        warnings.append("very_short_text")
    if len(text.split()) == 1 and len(text) > 24:
        warnings.append("single_token_text")
    alpha_ratio = sum(character.isalnum() for character in text) / max(len(text), 1)
    if alpha_ratio < 0.35:
        warnings.append("low_signal_text")
    for pattern in task_config.noise_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            warnings.append("noisy_text")
            break
    return sorted(set(warnings))


def _count_keyword_hits(text: str, keywords: list[str]) -> int:
    """Count exact keyword hits in text."""

    hits = 0
    for keyword in keywords:
        if re.search(rf"\b{re.escape(keyword.lower())}\b", text):
            hits += 1
    return hits


def _count_regex_hits(text: str, patterns: list[str]) -> int:
    """Count regex pattern hits in text."""

    return sum(1 for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE))


def _looks_like_direct_request(text: str) -> bool:
    """Heuristic for direct operational requests."""

    return bool(
        re.search(
            r"\b(how do i|how to|give me|show me|write me|build me|make me|help me)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Normalize raw scores into a probability-like distribution."""

    positive_scores = {label: max(float(value), 0.0001) for label, value in scores.items()}
    total = sum(positive_scores.values())
    return {
        label: round(value / total, 6)
        for label, value in positive_scores.items()
    }


def _build_result(
    class_scores: dict[str, float],
    backend_name: str,
    warnings: list[str] | None = None,
) -> TextLabelingResult:
    """Convert scores into a structured result."""

    ordered = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = round(float(top_score - second_score), 6)
    entropy = round(_normalized_entropy(class_scores), 6)
    return TextLabelingResult(
        label=top_label,
        confidence=round(float(top_score), 6),
        class_scores=class_scores,
        margin=margin,
        entropy=entropy,
        backend_name=backend_name,
        warnings=list(warnings or []),
    )


def _uncertain_result(
    task_config: AnnotationTaskConfig,
    backend_name: str,
    warnings: list[str] | None = None,
) -> TextLabelingResult:
    """Return a low-confidence default result for empty or broken text."""

    if frozenset(task_config.labels) == LEGACY_SAFETY_LABELS:
        class_scores = _normalize_scores(
            {
                "safe": 1.0,
                "borderline_safe": 1.15,
                "unsafe": 1.0,
            }
        )
    else:
        class_scores = _normalize_scores({label: 1.0 for label in task_config.labels})
    return _build_result(class_scores, backend_name=backend_name, warnings=warnings)


def _normalized_entropy(class_scores: dict[str, float]) -> float:
    """Return entropy normalized to the range [0, 1]."""

    probabilities = [max(float(score), 1e-12) for score in class_scores.values()]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    max_entropy = math.log(max(len(probabilities), 1))
    if max_entropy == 0:
        return 0.0
    return entropy / max_entropy
