from __future__ import annotations

import re

from agents.data_collection.schemas import ProfileFieldSource, SessionStatus, TopicProfile
from agents.data_collection.session import (
    CollectionSessionState,
    append_message,
    create_empty_session,
    update_status,
)


FIELD_ORDER = [
    "topic",
    "modality",
    "language",
    "task_type",
    "size_target",
    "needs_labels",
]

FIELD_QUESTIONS = {
    "topic": "Какую тему или предметную область нужно собрать?",
    "modality": "Какая модальность данных нужна: text, image, audio, video, tabular или другая?",
    "language": "Какой язык данных нужен?",
    "task_type": "Какой тип задачи нужен: classification, NER, QA, summarization, regression или другой?",
    "size_target": "Какой желаемый размер датасета в записях?",
    "needs_labels": "Нужны ли уже готовые метки?",
}

MODALITY_KEYWORDS = {
    "text": ["text", "текст", "текстовая"],
    "image": ["image", "images", "изображ", "картин", "photo"],
    "audio": ["audio", "speech", "голос", "аудио", "звук"],
    "video": ["video", "видео"],
    "tabular": ["tabular", "table", "таблич", "csv", "rows"],
    "timeseries": ["time series", "timeseries", "временн", "series"],
    "code": ["code", "код"],
}

LANGUAGE_KEYWORDS = {
    "english": ["english", "en", "англ", "english-language"],
    "russian": ["russian", "ru", "рус", "русский"],
    "multilingual": ["multilingual", "multi-language", "multilang", "мультиязы", "многоязы"],
    "kazakh": ["kazakh", "kk", "казах"],
}

TASK_KEYWORDS = {
    "classification": ["classification", "classify", "классиф"],
    "ner": ["ner", "named entity", "сущност"],
    "qa": ["qa", "question answering", "вопрос", "ответ"],
    "summarization": ["summarization", "summary", "суммар", "резюм"],
    "translation": ["translation", "translate", "перевод"],
    "regression": ["regression", "регресс"],
    "generation": ["generation", "generate", "генерац"],
    "detection": ["detection", "detect", "детект"],
    "segmentation": ["segmentation", "сегмент"],
}

CONFIRMED_SOURCES = {
    ProfileFieldSource.USER_EXPLICIT.value,
    ProfileFieldSource.CONFIRMED_BY_USER.value,
}


class ConversationManager:
    """Rule-based manager for topic clarification."""

    def __init__(self, session: CollectionSessionState | None = None) -> None:
        self.session = session or create_empty_session()

    def handle_user_message(self, user_message: str) -> str | None:
        """Process a user message and return the next question."""

        append_message(self.session, "user", user_message)
        self.update_topic_profile(user_message)

        if self.is_ready_for_discovery():
            update_status(self.session, SessionStatus.DISCOVERING)
            return None

        update_status(self.session, SessionStatus.CLARIFYING)
        next_question = self.get_next_question()
        if next_question:
            append_message(self.session, "assistant", next_question)
        return next_question

    def update_topic_profile(self, user_message: str) -> TopicProfile:
        """Update the topic profile from the latest message."""

        text = user_message.strip()
        if not text:
            return self.session.topic_profile

        profile = self.session.topic_profile
        lowered = text.casefold()
        missing_fields = self.get_missing_fields()
        first_missing = missing_fields[0] if missing_fields else None

        if first_missing == "topic" and profile.topic is None:
            profile.topic = text
            profile.field_provenance["topic"] = ProfileFieldSource.USER_EXPLICIT.value
            profile.discovery_hints.update(self._extract_topic_hints(text))
            self._record_hint_provenance(profile)
        elif first_missing == "modality":
            value = self._extract_by_keywords(lowered, MODALITY_KEYWORDS)
            if value is None:
                value = self._normalize_free_text(text)
            self._set_confirmed_field(profile, "modality", value)
        elif first_missing == "language":
            value = self._extract_by_keywords(lowered, LANGUAGE_KEYWORDS)
            if value is None:
                value = self._normalize_free_text(text)
            self._set_confirmed_field(profile, "language", value)
        elif first_missing == "task_type":
            value = self._extract_by_keywords(lowered, TASK_KEYWORDS)
            if value is None:
                value = self._normalize_free_text(text)
            self._set_confirmed_field(profile, "task_type", value)
        elif first_missing == "size_target":
            value = self._extract_size_target(lowered)
            if value is not None:
                self._set_confirmed_field(profile, "size_target", value)
        elif first_missing == "needs_labels":
            value = self._extract_needs_labels(lowered)
            if value is not None:
                self._set_confirmed_field(profile, "needs_labels", value)

        return profile

    def get_missing_fields(self) -> list[str]:
        """Return required profile fields that are still missing."""

        profile = self.session.topic_profile
        missing_fields: list[str] = []
        for field_name in FIELD_ORDER:
            if not self._is_field_confirmed(profile, field_name):
                missing_fields.append(field_name)
        return missing_fields

    def get_next_question(self) -> str | None:
        """Return the next clarification question."""

        missing_fields = self.get_missing_fields()
        if not missing_fields:
            return None
        return FIELD_QUESTIONS[missing_fields[0]]

    def is_ready_for_discovery(self) -> bool:
        """Check whether the profile is complete enough for discovery."""

        return not self.get_missing_fields()

    @staticmethod
    def _set_confirmed_field(
        profile: TopicProfile,
        field_name: str,
        value: object,
    ) -> None:
        """Set a structured field as explicitly confirmed."""

        setattr(profile, field_name, value)
        profile.field_provenance[field_name] = ProfileFieldSource.CONFIRMED_BY_USER.value

    @staticmethod
    def _is_field_confirmed(profile: TopicProfile, field_name: str) -> bool:
        """Check whether a field is present and explicitly confirmed."""

        value = getattr(profile, field_name)
        if value is None:
            return False
        return profile.field_provenance.get(field_name) in CONFIRMED_SOURCES

    @staticmethod
    def _extract_by_keywords(
        text: str,
        keyword_map: dict[str, list[str]],
    ) -> str | None:
        for normalized_value, keywords in keyword_map.items():
            for keyword in keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", text):
                    return normalized_value
        return None

    @staticmethod
    def _extract_size_target(text: str) -> int | None:
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*(k|m|тыс|тысяч|млн)?", text.casefold())
        if not match:
            return None

        value = float(match.group(1).replace(",", "."))
        suffix = match.group(2)
        multiplier = 1
        if suffix in {"k", "тыс", "тысяч"}:
            multiplier = 1_000
        elif suffix in {"m", "млн"}:
            multiplier = 1_000_000
        return int(value * multiplier)

    @staticmethod
    def _extract_needs_labels(text: str) -> bool | None:
        negative_markers = [
            "не нужны",
            "без меток",
            "без разметки",
            "unlabeled",
            "raw data",
            "нет",
            "no",
        ]
        positive_markers = [
            "нужны",
            "с метками",
            "с разметкой",
            "готовые метки",
            "размеч",
            "labeled",
            "labels",
            "да",
            "yes",
        ]

        if any(marker in text for marker in negative_markers):
            return False
        if any(marker in text for marker in positive_markers):
            return True
        return None

    @staticmethod
    def _normalize_free_text(text: str) -> str:
        return " ".join(text.strip().split())

    def _extract_topic_hints(self, text: str) -> dict[str, object]:
        """Extract soft discovery hints from raw topic text."""

        lowered = text.casefold()
        hints: dict[str, object] = {
            "keywords": self._extract_topic_keywords(text),
        }

        modality_hint = self._extract_by_keywords(lowered, MODALITY_KEYWORDS)
        if modality_hint is not None:
            hints["modality"] = modality_hint

        language_hint = self._extract_by_keywords(lowered, LANGUAGE_KEYWORDS)
        if language_hint is not None:
            hints["language"] = language_hint

        task_type_hint = self._extract_by_keywords(lowered, TASK_KEYWORDS)
        if task_type_hint is not None:
            hints["task_type"] = task_type_hint

        needs_labels_hint = self._extract_needs_labels(lowered)
        if needs_labels_hint is not None:
            hints["needs_labels"] = needs_labels_hint

        history_window_years = self._extract_history_window_years(lowered)
        if history_window_years is not None:
            hints["history_window_years"] = history_window_years

        return hints

    @staticmethod
    def _record_hint_provenance(profile: TopicProfile) -> None:
        """Record inferred provenance for hint-only fields."""

        for field_name in ("modality", "language", "task_type", "needs_labels"):
            if field_name in profile.discovery_hints and field_name not in profile.field_provenance:
                profile.field_provenance[field_name] = ProfileFieldSource.INFERRED_HINT.value

    @staticmethod
    def _extract_topic_keywords(text: str) -> list[str]:
        """Extract lightweight keyword hints from topic text."""

        tokens = re.findall(r"[a-zA-Zа-яА-Я0-9_+-]{3,}", text.casefold())
        stopwords = {
            "and",
            "the",
            "for",
            "from",
            "with",
            "data",
            "dataset",
            "last",
            "years",
            "year",
            "analysis",
            "prediction",
            "risk",
            "historical",
        }
        unique_keywords: list[str] = []
        for token in tokens:
            if token in stopwords or token.isdigit():
                continue
            if token not in unique_keywords:
                unique_keywords.append(token)
        return unique_keywords[:12]

    @staticmethod
    def _extract_history_window_years(text: str) -> int | None:
        """Extract historical window hints without treating them as dataset size."""

        match = re.search(
            r"(?:last|past)\s+(\d+)\s+years|(\d+)\s+years",
            text,
        )
        if not match:
            return None

        value = match.group(1) or match.group(2)
        return int(value) if value is not None else None
