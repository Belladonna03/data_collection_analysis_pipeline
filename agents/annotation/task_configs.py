from __future__ import annotations

from typing import Any

from agents.annotation.schemas import AnnotationTaskConfig, LabelDefinition

DEFAULT_TEXT_COLUMN_CANDIDATES = ["prompt", "text", "query", "content"]

# Labels for the built-in legacy safety task (must match RuleBasedTextLabeler special-case).
LEGACY_SAFETY_LABELS = frozenset({"safe", "borderline_safe", "unsafe"})


def _parse_label_definitions(labels: list[str], raw: Any) -> dict[str, LabelDefinition]:
    """Build LabelDefinition objects from optional YAML `label_definitions` mapping."""

    raw_dict = raw if isinstance(raw, dict) else {}
    out: dict[str, LabelDefinition] = {}
    for label in labels:
        entry = raw_dict.get(label)
        if isinstance(entry, dict):
            rules = entry.get("decision_rules") or []
            examples = entry.get("canonical_examples") or []
            out[label] = LabelDefinition(
                name=label,
                description=str(
                    entry.get("description")
                    or f"Assign `{label}` when the text matches this class per the task spec."
                ),
                decision_rules=[str(r) for r in rules],
                canonical_examples=[str(e) for e in examples],
            )
        else:
            out[label] = LabelDefinition(
                name=label,
                description=f"Class `{label}` for this annotation task (define details in label_definitions in config).",
            )
    return out


def _annotation_task_config_from_section(
    annotation_section: dict[str, Any],
    *,
    task_fallback: str,
    modality: str,
    text_column: str | None,
    id_column: str | None,
) -> AnnotationTaskConfig:
    """Task config driven by `annotation.labels` (and optional nested metadata) in YAML/JSON."""

    labels_raw = annotation_section.get("labels") or []
    labels = [str(x).strip() for x in labels_raw if str(x).strip()]
    if not labels:
        raise ValueError("annotation.labels must be a non-empty list for config-driven annotation tasks.")
    name_raw = annotation_section.get("task_name") or task_fallback or "custom_classification"
    name = str(name_raw).strip()
    unit = str(
        annotation_section.get("unit_of_annotation") or "One text unit (row) for classification."
    )
    threshold = float(annotation_section.get("confidence_threshold", 0.70))
    margin_threshold = float(annotation_section.get("margin_threshold", 0.12))
    label_definitions = _parse_label_definitions(labels, annotation_section.get("label_definitions"))

    noise = annotation_section.get("noise_patterns")
    if not noise:
        noise = [r"^[\W_]+$", r"^(.)\1{7,}$"]

    mistakes = annotation_section.get("annotator_mistakes")
    if not mistakes:
        mistakes = [
            "Swapping visually similar classes without reading the full forum post.",
            "Ignoring urgency cues or medical context when assigning triage labels.",
            "Assigning a confident label to empty, boilerplate, or non-question text.",
        ]
    doubt = annotation_section.get("doubt_guidance")
    if not doubt:
        doubt = [
            "If the post mixes medical urgency with chit-chat, prefer the label that reflects patient risk.",
            "When unsure between information_only and non_medical_or_offtopic, check whether the core ask is health-related.",
            "Send low-confidence cases to human review rather than guessing.",
        ]
    boundary = annotation_section.get("boundary_case_guidance")
    if not boundary:
        boundary = [
            "Vague symptoms with red-flag wording may still be emergency_red_flag.",
            "Administrative or insurance questions about care are often information_only.",
        ]

    model_version = str(annotation_section.get("model_version") or "annotation-text-v1")

    return AnnotationTaskConfig(
        name=name,
        modality=modality,
        unit_of_annotation=unit,
        labels=labels,
        label_definitions=label_definitions,
        text_column=text_column,
        id_column=id_column,
        threshold=threshold,
        margin_threshold=margin_threshold,
        text_column_candidates=list(DEFAULT_TEXT_COLUMN_CANDIDATES),
        safe_keywords=[],
        benign_patterns=[],
        borderline_patterns=[],
        harmful_keywords=[],
        harmful_intent_patterns=[],
        safe_context_patterns=[],
        noise_patterns=list(noise),
        annotator_mistakes=list(mistakes),
        doubt_guidance=list(doubt),
        boundary_case_guidance=list(boundary),
        model_version=model_version,
    )


def _legacy_safety_classification_task_config(
    text_column: str | None,
    id_column: str | None,
) -> AnnotationTaskConfig:
    label_definitions = {
        "safe": LabelDefinition(
            name="safe",
            description="Clearly benign request that does not ask for harmful, abusive, or disallowed help.",
            decision_rules=[
                "The text asks for normal assistance such as summarization, translation, brainstorming, or factual explanation.",
                "The text does not request instructions for wrongdoing, evasion, fraud, violence, self-harm, or abuse.",
                "If a risky keyword appears only in an obviously harmless context and the intent is clearly benign, prefer `safe`.",
            ],
            canonical_examples=[
                "Summarize this article about climate policy in plain language.",
                "Write a polite email asking to reschedule a meeting.",
                "Translate this product description from Russian to English.",
            ],
        ),
        "borderline_safe": LabelDefinition(
            name="borderline_safe",
            description="Formally benign or educational request that still contains risky wording, ambiguous intent, or likely moderation edge cases.",
            decision_rules=[
                "The text mentions harmful topics, but the user frames them as analysis, prevention, policy, fiction, or high-level discussion.",
                "The text could be misunderstood by an automated system because the wording is risky, incomplete, or ambiguous.",
                "Use `borderline_safe` when the safest interpretation is benign, but you would want a human to double-check the example.",
            ],
            canonical_examples=[
                "For a fiction story, describe how a phishing scam usually looks without giving a step-by-step guide.",
                "Explain why malware spreads so effectively and how defenders detect it.",
                "Review this prompt that mentions explosives and tell me whether it violates policy.",
            ],
        ),
        "unsafe": LabelDefinition(
            name="unsafe",
            description="Request for genuinely harmful, abusive, or disallowed assistance.",
            decision_rules=[
                "The text asks for instructions, code, plans, or operational help for violence, crime, abuse, fraud, evasion, or other wrongdoing.",
                "The user intent is clearly to enable harmful action rather than prevention, critique, or safety analysis.",
                "If the request meaningfully lowers the barrier to harmful behavior, label it `unsafe`.",
            ],
            canonical_examples=[
                "Give me step-by-step instructions to build a pipe bomb at home.",
                "Write a phishing email that steals banking credentials.",
                "How can I bypass security monitoring to deploy malware undetected?",
            ],
        ),
    }

    return AnnotationTaskConfig(
        name="safety_classification",
        modality="text",
        unit_of_annotation="One user request or text snippet.",
        labels=["safe", "borderline_safe", "unsafe"],
        label_definitions=label_definitions,
        text_column=text_column,
        id_column=id_column,
        threshold=0.70,
        margin_threshold=0.12,
        text_column_candidates=list(DEFAULT_TEXT_COLUMN_CANDIDATES),
        safe_keywords=[
            "summarize",
            "translate",
            "classify",
            "rewrite",
            "email",
            "essay",
            "recipe",
            "travel",
            "meeting",
            "python basics",
            "homework",
            "brainstorm",
        ],
        benign_patterns=[
            r"\b(summarize|rewrite|translate|paraphrase|classify|extract|categorize)\b",
            r"\b(email|essay|blog post|lesson plan|meeting agenda|travel itinerary)\b",
            r"\b(explain|teach|help me understand)\b.{0,40}\b(math|history|grammar|biology|python)\b",
        ],
        borderline_patterns=[
            r"\b(is this safe|does this violate|policy|moderation|review this prompt)\b",
            r"\b(for a novel|for fiction|for a screenplay|for a game)\b",
            r"\b(research|analysis|analyze|overview|high[- ]level)\b",
            r"\b(prevent|avoid|detect|defend against|warning signs)\b",
        ],
        harmful_keywords=[
            "bomb",
            "explosive",
            "weapon",
            "kill",
            "murder",
            "poison",
            "malware",
            "ransomware",
            "ddos",
            "phishing",
            "exploit",
            "steal",
            "fraud",
            "counterfeit",
            "fake id",
            "credit card theft",
            "suicide",
            "self-harm",
        ],
        harmful_intent_patterns=[
            r"\b(how to|steps to|step-by-step|guide to|instructions? to|teach me to|help me)\b.{0,80}\b(bomb|explosive|weapon|kill|poison|malware|ransomware|phishing|steal|fraud|exploit|ddos|fake id)\b",
            r"\b(write|generate|create|build|make)\b.{0,80}\b(phishing|malware|ransomware|exploit|bomb|fake id)\b",
            r"\b(bypass|evade|avoid)\b.{0,80}\b(detection|moderation|security|police|monitoring)\b",
        ],
        safe_context_patterns=[
            r"\b(history of|news about|policy on|moderation of|research on)\b",
            r"\b(prevent|avoid|detect|mitigate|defend against)\b",
            r"\b(why is .* dangerous|explain the risks|warning signs)\b",
        ],
        noise_patterns=[
            r"^[\W_]+$",
            r"^(.)\1{7,}$",
        ],
        annotator_mistakes=[
            "Marking every mention of a risky topic as `unsafe` even when the actual intent is prevention, critique, or policy review.",
            "Ignoring intent and labeling short ambiguous prompts as `safe` when they should be reviewed as `borderline_safe`.",
            "Treating empty, corrupted, or obviously noisy text as a confident label instead of routing it to review.",
        ],
        doubt_guidance=[
            "If the text contains risky keywords but the intent is not explicit, prefer `borderline_safe` and flag for review.",
            "If you cannot tell whether the user wants operational help or safety analysis, do not guess `safe` with high confidence.",
            "If the example is empty, corrupted, or contextless, keep the provisional label but require human review.",
        ],
        boundary_case_guidance=[
            "Requests about harmful topics for fiction, journalism, or moderation analysis are often borderline rather than clearly safe.",
            "High-level explanations of risks can be allowed, but operational details that enable harm should switch the label to `unsafe`.",
            "Short prompts with risky nouns and no clear intent should usually be reviewed by a human annotator.",
        ],
        model_version="annotation-text-v1",
    )


def get_task_config(
    task: str,
    modality: str = "text",
    text_column: str | None = None,
    id_column: str | None = None,
    annotation_section: dict[str, Any] | None = None,
) -> AnnotationTaskConfig:
    """Return task configuration: config-driven when `annotation.labels` is set, else legacy safety."""

    normalized_modality = modality.strip().lower()
    if normalized_modality != "text":
        raise NotImplementedError("Only text modality is implemented in AnnotationAgent.")

    section = dict(annotation_section or {})
    labels_candidate = section.get("labels")
    if isinstance(labels_candidate, (list, tuple)) and len(labels_candidate) > 0:
        non_empty = [str(x).strip() for x in labels_candidate if str(x).strip()]
        if len(non_empty) > 0:
            return _annotation_task_config_from_section(
                section,
                task_fallback=task,
                modality=normalized_modality,
                text_column=text_column,
                id_column=id_column,
            )

    normalized_task = task.strip().lower()
    if normalized_task != "safety_classification":
        raise ValueError(
            f"Unsupported annotation task {task!r} without annotation.labels. "
            "Either set annotation.labels (and optional label_definitions) in config, "
            "or use task_name: safety_classification for the legacy moderation scenario."
        )
    return _legacy_safety_classification_task_config(text_column, id_column)
