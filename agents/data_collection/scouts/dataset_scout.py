from __future__ import annotations

from dataclasses import dataclass

from agents.data_collection.schemas import SourceCandidate, SourceType, TopicProfile


@dataclass(frozen=True)
class DatasetCatalogEntry:
    """Simple static dataset metadata for MVP discovery."""

    name: str
    description: str
    url: str
    modality: str | None
    task_type: str | None
    language: str | None
    estimated_rows: int | None
    supports_labels: bool | None
    tags: tuple[str, ...] = ()


class DatasetScout:
    """Discover open dataset candidates from a small static catalog."""

    def __init__(self) -> None:
        self.catalog: list[DatasetCatalogEntry] = [
            DatasetCatalogEntry(
                name="imdb",
                description="Movie review sentiment dataset with binary labels.",
                url="https://huggingface.co/datasets/imdb",
                modality="text",
                task_type="classification",
                language="english",
                estimated_rows=50_000,
                supports_labels=True,
                tags=("movie", "review", "sentiment"),
            ),
            DatasetCatalogEntry(
                name="ag_news",
                description="News topic classification dataset.",
                url="https://huggingface.co/datasets/ag_news",
                modality="text",
                task_type="classification",
                language="english",
                estimated_rows=127_600,
                supports_labels=True,
                tags=("news", "classification"),
            ),
            DatasetCatalogEntry(
                name="xnli",
                description="Multilingual natural language inference dataset.",
                url="https://huggingface.co/datasets/xnli",
                modality="text",
                task_type="classification",
                language="multilingual",
                estimated_rows=750_000,
                supports_labels=True,
                tags=("multilingual", "nli", "classification"),
            ),
            DatasetCatalogEntry(
                name="squad",
                description="Question answering dataset with context and answer spans.",
                url="https://huggingface.co/datasets/squad",
                modality="text",
                task_type="qa",
                language="english",
                estimated_rows=100_000,
                supports_labels=True,
                tags=("qa", "question answering"),
            ),
            DatasetCatalogEntry(
                name="common_voice",
                description="Crowdsourced speech dataset for ASR tasks.",
                url="https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0",
                modality="audio",
                task_type="transcription",
                language="multilingual",
                estimated_rows=None,
                supports_labels=True,
                tags=("speech", "audio", "asr"),
            ),
        ]

    def discover(self, topic_profile: TopicProfile) -> list[SourceCandidate]:
        """Return ranked dataset candidates for the topic profile."""

        ranked_candidates: list[SourceCandidate] = []
        for entry in self.catalog:
            score = self._score_entry(topic_profile, entry)
            if score <= 0.2:
                continue
            ranked_candidates.append(
                SourceCandidate(
                    source_type=SourceType.HF_DATASET,
                    name=entry.name,
                    description=entry.description,
                    url=entry.url,
                    estimated_rows=entry.estimated_rows,
                    supports_labels=entry.supports_labels,
                    relevance_score=round(score, 3),
                    pros=["Open dataset", "Easy to bootstrap collection"],
                    cons=["May not fully match the target domain"],
                    risks=["Schema may need normalization"],
                )
            )

        ranked_candidates.sort(
            key=lambda candidate: candidate.relevance_score or 0.0,
            reverse=True,
        )
        return ranked_candidates

    def _score_entry(
        self,
        topic_profile: TopicProfile,
        entry: DatasetCatalogEntry,
    ) -> float:
        """Score how well a catalog entry matches the topic profile."""

        score = 0.2
        topic = (topic_profile.topic or "").casefold()

        if topic_profile.modality and topic_profile.modality == entry.modality:
            score += 0.25
        if topic_profile.task_type and topic_profile.task_type == entry.task_type:
            score += 0.25
        if topic_profile.language:
            if topic_profile.language == entry.language:
                score += 0.2
            elif entry.language == "multilingual":
                score += 0.1
        if topic and any(tag in topic for tag in entry.tags):
            score += 0.2
        if topic_profile.needs_labels is True and entry.supports_labels:
            score += 0.1
        if topic_profile.size_target and entry.estimated_rows:
            if entry.estimated_rows >= topic_profile.size_target:
                score += 0.1

        return min(score, 1.0)
