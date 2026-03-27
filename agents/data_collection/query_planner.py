from __future__ import annotations

import json
import logging
import re
from typing import Any

from agents.data_collection.schemas import QueryPlan, TopicProfile


LOGGER = logging.getLogger(__name__)

DEFAULT_ASSET_TERMS = [
    "dataset",
    "csv",
    "parquet",
    "match results",
    "rankings",
    "stats",
]


class DeterministicQueryPlanner:
    """Build bounded provider-specific queries without an LLM."""

    def __init__(
        self,
        max_queries_per_provider: int = 4,
        max_query_length: int = 80,
    ) -> None:
        self.max_queries_per_provider = max_queries_per_provider
        self.max_query_length = max_query_length

    def build_plan(self, topic_profile: TopicProfile) -> QueryPlan:
        """Build a deterministic query plan."""

        topic = " ".join((topic_profile.topic or "").split())
        normalized_goal = self._truncate(topic, self.max_query_length)
        domain_terms = self._extract_domain_terms(topic_profile)
        asset_terms = self._extract_asset_terms(topic_profile)

        provider_queries = {
            "huggingface": self._build_provider_queries(
                provider="huggingface",
                normalized_goal=normalized_goal,
                domain_terms=domain_terms,
                asset_terms=asset_terms,
            ),
            "github": self._build_provider_queries(
                provider="github",
                normalized_goal=normalized_goal,
                domain_terms=domain_terms,
                asset_terms=asset_terms,
            ),
            "kaggle": self._build_provider_queries(
                provider="kaggle",
                normalized_goal=normalized_goal,
                domain_terms=domain_terms,
                asset_terms=asset_terms,
            ),
            "web_forum": self._build_web_forum_queries(
                normalized_goal=normalized_goal,
                domain_terms=domain_terms,
                asset_terms=asset_terms,
            ),
            # DevtoolsHarDiscoveryClient emits configured hints once per discover() run (query text ignored).
            "devtools_har": ["__devtools_hints__"],
        }

        return QueryPlan(
            normalized_goal=normalized_goal,
            domain_terms=domain_terms,
            asset_terms=asset_terms,
            provider_queries=provider_queries,
        )

    def _extract_domain_terms(self, topic_profile: TopicProfile) -> list[str]:
        """Extract bounded domain terms from topic and hints."""

        hinted_keywords = topic_profile.discovery_hints.get("keywords", [])
        candidate_terms = hinted_keywords or re.findall(
            r"[a-zA-Zа-яА-Я0-9_+-]{3,}",
            (topic_profile.topic or "").casefold(),
        )
        stopwords = {
            "historical",
            "analysis",
            "prediction",
            "ready",
            "data",
            "dataset",
            "years",
            "year",
            "last",
            "from",
            "with",
            "without",
            "pre",
            "match",
        }

        terms: list[str] = []
        for term in candidate_terms:
            normalized_term = term.casefold()
            if normalized_term in stopwords:
                continue
            if normalized_term not in terms:
                terms.append(normalized_term)
        return terms[:6]

    def _extract_asset_terms(self, topic_profile: TopicProfile) -> list[str]:
        """Extract asset-first search terms."""

        asset_terms = list(DEFAULT_ASSET_TERMS)
        if topic_profile.modality == "tabular":
            asset_terms = ["dataset", "csv", "parquet", "table", "stats", "results"]

        hinted_terms = topic_profile.discovery_hints.get("asset_terms", [])
        for term in hinted_terms:
            normalized_term = term.casefold()
            if normalized_term not in asset_terms:
                asset_terms.append(normalized_term)

        history_window = topic_profile.discovery_hints.get("history_window_years")
        if history_window:
            asset_terms.insert(0, f"{history_window} years")

        return asset_terms[:6]

    def _build_provider_queries(
        self,
        provider: str,
        normalized_goal: str,
        domain_terms: list[str],
        asset_terms: list[str],
    ) -> list[str]:
        """Build deterministic queries for one provider."""

        joined_domain = " ".join(domain_terms[:3])
        joined_entities = " ".join(domain_terms[:2])
        queries = [
            joined_domain,
            self._join_and_truncate([joined_domain, "dataset"]),
            self._join_and_truncate([joined_entities, asset_terms[0] if asset_terms else "dataset"]),
            self._join_and_truncate(domain_terms[:3] + asset_terms[:2]),
        ]

        if "tennis" in domain_terms:
            tennis_queries = [
                "atp wta historical match results",
                "atp wta tennis dataset",
                "tennis betting odds atp wta",
                "jeff sackmann tennis",
            ]
            queries = tennis_queries + queries

        if provider == "github":
            queries.append(self._join_and_truncate(domain_terms[:3] + ["github repository"]))
        elif provider == "kaggle":
            queries.append(self._join_and_truncate(domain_terms[:3] + ["kaggle dataset"]))

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
            deduplicated.append(self._truncate(normalized_query, self.max_query_length))
        return deduplicated[: self.max_queries_per_provider]

    def _build_web_forum_queries(
        self,
        *,
        normalized_goal: str,
        domain_terms: list[str],
        asset_terms: list[str],
    ) -> list[str]:
        joined = " ".join(domain_terms[:3]) if domain_terms else normalized_goal
        entities = " ".join(domain_terms[:2]) if domain_terms else normalized_goal
        queries = [
            self._join_and_truncate([joined, "forum discussion"]),
            self._join_and_truncate([joined, "community questions"]),
            self._join_and_truncate([entities, "discussion board"]),
            normalized_goal,
            self._join_and_truncate([joined, asset_terms[0] if asset_terms else "Q&A", "forum"]),
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
            deduplicated.append(self._truncate(normalized_query, self.max_query_length))
        return deduplicated[: self.max_queries_per_provider]

    def _join_and_truncate(self, parts: list[str]) -> str:
        """Join parts and truncate to configured length."""

        return self._truncate(" ".join(part for part in parts if part), self.max_query_length)

    @staticmethod
    def _truncate(text: str, max_length: int) -> str:
        """Truncate long query text at word boundaries."""

        normalized_text = " ".join(text.split())
        if len(normalized_text) <= max_length:
            return normalized_text
        truncated = normalized_text[:max_length].rsplit(" ", 1)[0]
        return truncated or normalized_text[:max_length]


class LLMQueryPlanner:
    """Optional bounded LLM augmentation for provider queries."""

    def __init__(
        self,
        llm: Any,
        max_queries_per_provider: int = 4,
        max_query_length: int = 80,
    ) -> None:
        self.llm = llm
        self.max_queries_per_provider = max_queries_per_provider
        self.max_query_length = max_query_length

    def augment_plan(self, topic_profile: TopicProfile, base_plan: QueryPlan) -> QueryPlan:
        """Augment a deterministic plan while preserving bounded behavior."""

        prompt = self._build_prompt(topic_profile, base_plan)
        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", response)
            parsed = json.loads(self._extract_json(str(content)))
        except Exception as exc:
            LOGGER.warning("LLM query planner failed, using deterministic plan: %s", exc)
            return base_plan

        provider_queries = dict(base_plan.provider_queries)
        raw_provider_queries = parsed.get("provider_queries", {})
        for provider_name, base_queries in provider_queries.items():
            merged_queries = list(base_queries)
            for query in raw_provider_queries.get(provider_name, []):
                normalized_query = self._truncate(" ".join(str(query).split()))
                if not normalized_query:
                    continue
                if normalized_query.casefold() not in {
                    existing.casefold() for existing in merged_queries
                }:
                    merged_queries.append(normalized_query)
            provider_queries[provider_name] = merged_queries[: self.max_queries_per_provider]

        domain_terms = self._merge_terms(
            base_plan.domain_terms,
            parsed.get("domain_terms", []),
        )
        asset_terms = self._merge_terms(
            base_plan.asset_terms,
            parsed.get("asset_terms", []),
        )

        return QueryPlan(
            normalized_goal=base_plan.normalized_goal,
            domain_terms=domain_terms,
            asset_terms=asset_terms,
            provider_queries=provider_queries,
        )

    def _build_prompt(self, topic_profile: TopicProfile, base_plan: QueryPlan) -> str:
        """Build a constrained JSON-only prompt."""

        return (
            "You are a dataset search query planner.\n"
            "Return strict JSON only.\n"
            f"Max {self.max_queries_per_provider} queries per provider.\n"
            f"Max query length: {self.max_query_length} characters.\n"
            "Focus on asset-first data discovery, not ML task phrasing.\n"
            "Do not invent providers. Allowed providers: huggingface, kaggle, github.\n"
            "Schema:\n"
            '{"domain_terms": ["..."], "asset_terms": ["..."], '
            '"provider_queries": {"huggingface": ["..."], "kaggle": ["..."], "github": ["..."]}}\n'
            f"Topic: {topic_profile.topic}\n"
            f"Deterministic plan: {json.dumps(base_plan.provider_queries, ensure_ascii=False)}"
        )

    def _merge_terms(self, base_terms: list[str], new_terms: list[Any]) -> list[str]:
        """Merge and bound planner terms."""

        merged = list(base_terms)
        for term in new_terms:
            normalized_term = " ".join(str(term).casefold().split())
            if not normalized_term:
                continue
            if normalized_term not in merged:
                merged.append(normalized_term)
        return merged[:6]

    def _truncate(self, query: str) -> str:
        """Truncate an LLM-generated query."""

        if len(query) <= self.max_query_length:
            return query
        truncated = query[: self.max_query_length].rsplit(" ", 1)[0]
        return truncated or query[: self.max_query_length]

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract the first JSON object from model output."""

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM query planner did not return JSON.")
        return text[start : end + 1]
