from __future__ import annotations

from agents.data_collection.schemas import SourceCandidate, SourceType, TopicProfile


class RepoScout:
    """Discover repository-style dataset candidates."""

    def discover(self, topic_profile: TopicProfile) -> list[SourceCandidate]:
        """Return repository candidates for the topic profile."""

        if not topic_profile.topic:
            return []

        topic_slug = "-".join(topic_profile.topic.lower().split())
        return [
            SourceCandidate(
                source_type=SourceType.REPOSITORY,
                name=f"{topic_profile.topic} Kaggle collection",
                description="Stub repository candidate for community-maintained datasets.",
                url=f"https://www.kaggle.com/search?q={topic_slug}",
                estimated_rows=topic_profile.size_target,
                supports_labels=topic_profile.needs_labels,
                relevance_score=0.5,
                is_executable=False,
                non_executable_reason="Repository-style discovery candidates require manual review before execution.",
                pros=["Open dataset repository", "Can provide alternative schemas"],
                cons=["Quality may vary across community datasets"],
                risks=["License and freshness require manual review"],
            )
        ]
