from __future__ import annotations

from agents.data_collection.schemas import TopicProfile


class DiscoveryQueryBuilder:
    """Build deterministic search queries for discovery providers."""

    def __init__(self, max_queries_per_provider: int = 4) -> None:
        self.max_queries_per_provider = max_queries_per_provider

    def build_queries(self, topic_profile: TopicProfile) -> list[str]:
        """Build deduplicated discovery queries."""

        topic = (topic_profile.topic or "").strip()
        if not topic:
            return []

        structured_terms = [
            topic_profile.modality,
            topic_profile.task_type,
            topic_profile.language,
        ]
        hint_keywords = topic_profile.discovery_hints.get("keywords", [])

        queries = [
            topic,
            self._join_non_empty([topic, *structured_terms]),
            self._join_non_empty([topic, *hint_keywords[:3]]),
            self._join_non_empty([topic, "dataset"]),
            self._join_non_empty([topic, "github repository"]),
        ]

        deduplicated: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized_query = " ".join(query.split())
            if not normalized_query:
                continue
            lowered = normalized_query.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduplicated.append(normalized_query)

        return deduplicated[: self.max_queries_per_provider]

    @staticmethod
    def _join_non_empty(parts: list[str | None]) -> str:
        """Join non-empty query parts."""

        return " ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())
