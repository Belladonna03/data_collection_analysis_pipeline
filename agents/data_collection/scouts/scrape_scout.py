from __future__ import annotations

from agents.data_collection.schemas import SourceCandidate, SourceType, TopicProfile


class ScrapeScout:
    """Discover demo-safe scrape candidates."""

    def discover(self, topic_profile: TopicProfile) -> list[SourceCandidate]:
        """Return scrape source candidates for the topic profile."""

        if not topic_profile.topic:
            return []

        return [
            SourceCandidate(
                source_type=SourceType.SCRAPE,
                name="quotes_toscrape",
                description="Public quote pages suitable for deterministic scrape demos.",
                url="https://quotes.toscrape.com/",
                selector=".quote",
                relevance_score=0.61,
                pros=["Public site", "Stable pagination for demos"],
                cons=["Very small and generic corpus"],
                risks=["HTML selectors can still change"],
            )
        ]
