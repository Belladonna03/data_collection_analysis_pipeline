from __future__ import annotations

from agents.data_collection.schemas import SourceCandidate, SourceType, TopicProfile


class APIScout:
    """Discover demo-safe API candidates."""

    def discover(self, topic_profile: TopicProfile) -> list[SourceCandidate]:
        """Return API source candidates for the topic profile."""

        if not topic_profile.topic:
            return []

        return [
            SourceCandidate(
                source_type=SourceType.API,
                name="dummyjson_quotes",
                description="Public quotes API that is easy to test from the CLI.",
                endpoint="https://dummyjson.com/quotes",
                relevance_score=0.72,
                pros=["Public endpoint", "Predictable JSON schema"],
                cons=["Narrow domain coverage"],
                risks=["Demo-safe source, not domain-specific"],
            )
        ]
