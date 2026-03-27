from __future__ import annotations

import logging
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from agents.data_collection.query_planner import (
    DeterministicQueryPlanner,
    LLMQueryPlanner,
)
from agents.data_collection.scouts.api_scout import APIScout
from agents.data_collection.scouts.dataset_scout import DatasetScout
from agents.data_collection.scouts.repo_scout import RepoScout
from agents.data_collection.scouts.scrape_scout import ScrapeScout
from agents.data_collection.schemas import (
    DiscoveryCapability,
    DiscoveryJournal,
    DiscoveryProvider,
    QueryPlan,
    RawSearchHit,
    SearchEvidence,
    SourceCandidate,
    SourceType,
    TopicProfile,
)
from agents.data_collection.search_clients.base import (
    BaseSearchClient,
    SearchClientConfig,
    SearchClientError,
)
from agents.data_collection.search_clients.devtools_har_search import DevtoolsHarDiscoveryClient
from agents.data_collection.search_clients.github_search import GitHubRepoSearchClient
from agents.data_collection.search_clients.huggingface_search import HuggingFaceSearchClient
from agents.data_collection.search_clients.kaggle_search import KaggleSearchClient
from agents.data_collection.search_clients.web_forum_search import WebForumSearchClient


LOGGER = logging.getLogger(__name__)


class SourceDiscoveryService:
    """Orchestrate source discovery across scouts."""

    def __init__(
        self,
        config: dict | None = None,
        llm: object | None = None,
        deterministic_query_planner: DeterministicQueryPlanner | None = None,
        llm_query_planner: LLMQueryPlanner | None = None,
        search_clients: list[BaseSearchClient] | None = None,
        dataset_scout: DatasetScout | None = None,
        api_scout: APIScout | None = None,
        scrape_scout: ScrapeScout | None = None,
        repo_scout: RepoScout | None = None,
    ) -> None:
        self.config = config or {}
        self.llm = llm
        self.deterministic_query_planner = deterministic_query_planner or DeterministicQueryPlanner(
            max_queries_per_provider=self.config.get("max_queries_per_provider", 4)
        )
        self.llm_query_planner = llm_query_planner
        if self.llm_query_planner is None and llm is not None and self.config.get(
            "enable_llm_query_planner",
            False,
        ):
            self.llm_query_planner = LLMQueryPlanner(
                llm=llm,
                max_queries_per_provider=self.config.get("max_queries_per_provider", 4),
                max_query_length=self.config.get("max_query_length", 80),
            )
        self.search_clients = search_clients or self._build_search_clients(self.config)
        self.dataset_scout = dataset_scout or DatasetScout()
        self.api_scout = api_scout or APIScout()
        self.scrape_scout = scrape_scout or ScrapeScout()
        self.repo_scout = repo_scout or RepoScout()
        self.last_journal = DiscoveryJournal()

    def _merged_auto_scrape_config(self) -> dict:
        """``connectors.scrape.auto_scrape`` and optional top-level ``scrape.auto_scrape`` (agent-injected)."""

        cs = self.config.get("_connectors_scrape") or {}
        tops = self.config.get("_top_level_scrape") or {}
        a1 = cs.get("auto_scrape") if isinstance(cs.get("auto_scrape"), dict) else {}
        a2 = tops.get("auto_scrape") if isinstance(tops.get("auto_scrape"), dict) else {}
        merged: dict = {**a1, **a2}
        return merged

    def _auto_scrape_discovery_enabled(self) -> bool:
        """Run requests-html probe for HTML web_forum hits when enabled in config or legacy web_forum flag."""

        if bool(self._merged_auto_scrape_config().get("enabled")):
            return True
        wf = (self.config.get("providers") or {}).get("web_forum") or {}
        return bool(wf.get("auto_probe_executable"))

    def _auto_scrape_probe_params(self) -> tuple[float, bool]:
        auto = self._merged_auto_scrape_config()
        wf = (self.config.get("providers") or {}).get("web_forum") or {}
        timeout = float(auto.get("timeout", wf.get("timeout", 25.0)))
        render_fallback = bool(auto.get("probe_render_fallback", wf.get("probe_render_fallback", True)))
        return timeout, render_fallback

    def discover(self, topic_profile: TopicProfile) -> list[SourceCandidate]:
        """Collect, deduplicate, and rank candidates."""

        for client in self.search_clients:
            if hasattr(client, "_hints_emitted"):
                setattr(client, "_hints_emitted", False)

        journal = DiscoveryJournal()
        query_plan = self.deterministic_query_planner.build_plan(topic_profile)
        if self.llm_query_planner is not None:
            query_plan = self.llm_query_planner.augment_plan(topic_profile, query_plan)
        journal.query_plan = {
            "normalized_goal": query_plan.normalized_goal,
            "domain_terms": query_plan.domain_terms,
            "asset_terms": query_plan.asset_terms,
            "provider_queries": query_plan.provider_queries,
        }
        journal.queries = self._flatten_provider_queries(query_plan)
        LOGGER.info("Discovery query plan: %s", journal.query_plan)

        capabilities = self._check_capabilities()
        journal.provider_capabilities = capabilities

        strict_mode = bool(self.config.get("strict_provider_check", False))
        available_clients = [
            client
            for client, capability in zip(self.search_clients, capabilities)
            if capability.available
        ]
        if strict_mode:
            violations = self._required_provider_violations(capabilities)
            if violations:
                raise RuntimeError("Required discovery providers unavailable: " + "; ".join(violations))

        evidence: list[SearchEvidence] = []
        raw_hits: list[RawSearchHit] = []
        candidates: list[SourceCandidate] = []

        for client in available_clients:
            provider_log = {
                "provider": client.provider.value,
                "queries": [],
                "errors": [],
                "hits": 0,
            }
            provider_queries = query_plan.provider_queries.get(client.provider.value, [])
            if not provider_queries and getattr(client, "supports_empty_query_list", False):
                provider_queries = ["__devtools_hints__"]
            if not provider_queries:
                continue
            for query in provider_queries:
                provider_log["queries"].append(query)
                try:
                    LOGGER.info("Searching provider=%s query=%s", client.provider.value, query)
                    hits = client.search(query)
                    LOGGER.info(
                        "Provider=%s query=%s parsed_hits=%s",
                        client.provider.value,
                        query,
                        len(hits),
                    )
                    raw_hits.extend(hits)
                    provider_log["hits"] += len(hits)
                    for hit in hits:
                        evidence_item = self._build_evidence(hit)
                        evidence.append(evidence_item)
                        candidates.append(
                            self._normalize_hit_to_candidate(
                                topic_profile=topic_profile,
                                hit=hit,
                                evidence_item=evidence_item,
                            )
                        )
                except SearchClientError as exc:
                    LOGGER.warning(
                        "Provider=%s query=%s failed: %s",
                        client.provider.value,
                        query,
                        exc,
                    )
                    provider_log["errors"].append(str(exc))
                    continue
                time.sleep(1.0 / max(client.config.rate_limit_per_second, 0.1))
            journal.provider_logs.append(provider_log)

        journal.raw_hits = raw_hits
        journal.evidence = evidence

        unique_candidates = self._deduplicate_candidates(candidates)
        unique_candidates.sort(
            key=lambda candidate: candidate.relevance_score or 0.0,
            reverse=True,
        )

        unique_candidates = self.attach_scrape_fallback_for_http_landing_candidates(unique_candidates)

        if not unique_candidates and self.config.get("allow_demo_fallback", False):
            fallback_candidates = self._build_demo_fallback(topic_profile)
            journal.used_demo_fallback = True
            unique_candidates = fallback_candidates
            LOGGER.warning("Discovery used demo fallback candidates.")

        self.last_journal = journal
        return unique_candidates

    @staticmethod
    def attach_scrape_fallback_for_http_landing_candidates(
        candidates: list[SourceCandidate],
    ) -> list[SourceCandidate]:
        """Append a non-executable SCRAPE hint for http_file URLs whose path looks like HTML/landing pages.

        This makes the scraper / generated-scraper path visible in plans without pretending the HTTP file
        connector should succeed on that URL.
        """

        from agents.data_collection.connectors.http_file_connector import _LANDING_PATH_SUFFIXES

        extras: list[SourceCandidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in candidates:
            if candidate.source_type is not SourceType.HTTP_FILE or not candidate.url:
                continue
            path = urlsplit(candidate.url).path.casefold()
            if not any(path.endswith(suffix) for suffix in _LANDING_PATH_SUFFIXES):
                continue
            key = ("landing_scrape_hint", SourceDiscoveryService._canonicalize_url(candidate.url))
            if key in seen:
                continue
            seen.add(key)
            extras.append(
                SourceCandidate(
                    source_type=SourceType.SCRAPE,
                    name=f"{candidate.name} (HTML landing — scraper fallback)",
                    url=candidate.url,
                    selector="",
                    relevance_score=max((candidate.relevance_score or 0.0) - 0.05, 0.05),
                    is_executable=False,
                    non_executable_reason=(
                        "Companion to http_file: URL path suggests an HTML landing document, not a direct "
                        "downloadable table. Use DataCollectionAgent.auto_scrape() or manual scrape after "
                        "higher-priority sources; scraping is the last ingestion tier by design."
                    ),
                    selection_rationale="Auto-attached for HTML-like http_file paths (orchestration visibility).",
                )
            )
        return candidates + extras

    @staticmethod
    def _deduplicate_candidates(
        candidates: list[SourceCandidate],
    ) -> list[SourceCandidate]:
        """Remove obvious duplicates across scouts."""

        deduplicated: dict[tuple[str, str], SourceCandidate] = {}
        for candidate in candidates:
            key = (
                candidate.normalized_source_id or "",
                SourceDiscoveryService._canonicalize_url(candidate.url or candidate.endpoint or ""),
            )
            previous = deduplicated.get(key)
            if previous is None or (candidate.relevance_score or 0.0) > (
                previous.relevance_score or 0.0
            ):
                deduplicated[key] = candidate
        by_title: dict[tuple[str, str], SourceCandidate] = {}
        for candidate in deduplicated.values():
            fallback_key = (
                candidate.platform or "",
                candidate.name.casefold(),
            )
            previous = by_title.get(fallback_key)
            if previous is None or (candidate.relevance_score or 0.0) > (
                previous.relevance_score or 0.0
            ):
                by_title[fallback_key] = candidate
        return list(by_title.values())

    def _required_provider_violations(self, capabilities: list[DiscoveryCapability]) -> list[str]:
        """Enforce ``discovery.required_providers`` including synthetic ``scrape``.

        Pre-flight checks only whether scrape-oriented *clients* are available. Whether discovery
        produced an executable scrape/API outcome (including HTML after a successful auto-scrape
        probe) is reflected per-candidate; see :meth:`candidate_is_executable_scrape_web_outcome`.
        """

        raw = self.config.get("required_providers") or []
        req = [str(x).strip().lower() for x in raw if str(x).strip()]
        if not req:
            return []
        by_provider = {c.provider.value: c for c in capabilities}
        errors: list[str] = []
        if "scrape" in req:
            scrape_ok = any(
                by_provider.get(name) is not None and by_provider[name].available
                for name in ("web_forum", "devtools_har")
            )
            if not scrape_ok:
                wf = by_provider.get("web_forum")
                dh = by_provider.get("devtools_har")
                detail_parts = []
                if wf is not None:
                    detail_parts.append(f"web_forum available={wf.available}: {wf.reason or 'ok'}")
                if dh is not None:
                    detail_parts.append(f"devtools_har available={dh.available}: {dh.reason or 'ok'}")
                extra = "; ".join(detail_parts) if detail_parts else "no web_forum/devtools_har clients"
                errors.append(
                    "required_providers includes 'scrape' but no scrape/web discovery provider is usable "
                    f"({extra})."
                )
        for r in req:
            if r == "scrape":
                continue
            cap = by_provider.get(r)
            if cap is None:
                errors.append(f"Unknown required provider {r!r} (not in discovery search clients).")
                continue
            if not cap.available:
                errors.append(f"{r}: {cap.reason}".strip())
        return errors

    @staticmethod
    def candidate_is_executable_scrape_web_outcome(candidate: SourceCandidate) -> bool:
        """Return True if *candidate* is an executable ``web_forum`` hit (HTML scrape or JSON API).

        Used to treat successful HTML auto-scrape the same class of outcome as ``api_json`` for
        downstream filtering (catalogs, reporting) without weakening strict *client* checks.
        """

        if not candidate.is_executable:
            return False
        if (candidate.platform or "").lower() != DiscoveryProvider.WEB_FORUM.value:
            return False
        return candidate.source_type in (SourceType.SCRAPE, SourceType.API)

    def _check_capabilities(self) -> list[DiscoveryCapability]:
        """Check availability of all configured providers."""

        capabilities = [client.check_capability() for client in self.search_clients]
        for capability in capabilities:
            LOGGER.info(
                "Provider capability provider=%s available=%s reason=%s",
                capability.provider.value,
                capability.available,
                capability.reason,
            )
        return capabilities

    @staticmethod
    def _build_search_clients(config: dict) -> list[BaseSearchClient]:
        """Build the configured search clients."""

        provider_config = config.get("providers", {})
        clients: list[BaseSearchClient] = [
            HuggingFaceSearchClient(
                SearchClientConfig(
                    enabled=provider_config.get("huggingface", {}).get("enabled", True),
                    timeout=provider_config.get("huggingface", {}).get("timeout", 20.0),
                    retries=provider_config.get("huggingface", {}).get("retries", 1),
                    rate_limit_per_second=provider_config.get("huggingface", {}).get(
                        "rate_limit_per_second",
                        2.0,
                    ),
                    max_results_per_query=provider_config.get("huggingface", {}).get(
                        "max_results_per_query",
                        5,
                    ),
                    token=provider_config.get("huggingface", {}).get("token"),
                )
            ),
            GitHubRepoSearchClient(
                SearchClientConfig(
                    enabled=provider_config.get("github", {}).get("enabled", True),
                    timeout=provider_config.get("github", {}).get("timeout", 20.0),
                    retries=provider_config.get("github", {}).get("retries", 1),
                    rate_limit_per_second=provider_config.get("github", {}).get(
                        "rate_limit_per_second",
                        1.0,
                    ),
                    max_results_per_query=provider_config.get("github", {}).get(
                        "max_results_per_query",
                        5,
                    ),
                    token=provider_config.get("github", {}).get("token"),
                )
            ),
            KaggleSearchClient(
                SearchClientConfig(
                    enabled=provider_config.get("kaggle", {}).get("enabled", False),
                    timeout=provider_config.get("kaggle", {}).get("timeout", 20.0),
                    retries=provider_config.get("kaggle", {}).get("retries", 1),
                    rate_limit_per_second=provider_config.get("kaggle", {}).get(
                        "rate_limit_per_second",
                        1.0,
                    ),
                    max_results_per_query=provider_config.get("kaggle", {}).get(
                        "max_results_per_query",
                        5,
                    ),
                    username=provider_config.get("kaggle", {}).get("username"),
                    key=provider_config.get("kaggle", {}).get("key"),
                    command=provider_config.get("kaggle", {}).get("command"),
                )
            ),
        ]
        wf = dict(provider_config.get("web_forum") or {})
        if wf.get("enabled", False):
            clients.append(
                WebForumSearchClient(
                    SearchClientConfig(
                        enabled=True,
                        timeout=float(wf.get("timeout", 25.0)),
                        retries=int(wf.get("retries", 1)),
                        rate_limit_per_second=float(wf.get("rate_limit_per_second", 0.5)),
                        max_results_per_query=int(wf.get("max_results_per_query", 5)),
                        allow_domains=list(wf.get("allow_domains") or []),
                        deny_domains=list(wf.get("deny_domains") or []),
                        user_agent=wf.get("user_agent"),
                    )
                )
            )
        dh = dict(provider_config.get("devtools_har") or {})
        if dh.get("enabled", False):
            clients.append(
                DevtoolsHarDiscoveryClient(
                    SearchClientConfig(
                        enabled=True,
                        timeout=float(dh.get("timeout", 15.0)),
                        retries=int(dh.get("retries", 0)),
                        rate_limit_per_second=float(dh.get("rate_limit_per_second", 50.0)),
                        max_results_per_query=max(
                            20,
                            len(dh.get("hints") or []) * 5 or 1,
                        ),
                        metadata={"hints": list(dh.get("hints") or [])},
                    )
                )
            )
        return clients

    def _normalize_hit_to_candidate(
        self,
        topic_profile: TopicProfile,
        hit: RawSearchHit,
        evidence_item: SearchEvidence,
    ) -> SourceCandidate:
        """Convert a raw hit into a normalized source candidate."""

        if hit.provider is DiscoveryProvider.HUGGING_FACE:
            return self._build_hf_candidate(topic_profile, hit, evidence_item)
        if hit.provider is DiscoveryProvider.GITHUB:
            return self._build_github_candidate(topic_profile, hit, evidence_item)
        if hit.provider is DiscoveryProvider.KAGGLE:
            return self._build_kaggle_candidate(topic_profile, hit, evidence_item)
        if hit.provider is DiscoveryProvider.WEB_FORUM:
            return self._build_web_forum_candidate(topic_profile, hit, evidence_item)
        if hit.provider is DiscoveryProvider.DEVTOOLS_HAR:
            return self._build_devtools_har_candidate(topic_profile, hit, evidence_item)
        raise ValueError(f"Unsupported discovery provider: {hit.provider.value}")

    def _build_hf_candidate(
        self,
        topic_profile: TopicProfile,
        hit: RawSearchHit,
        evidence_item: SearchEvidence,
    ) -> SourceCandidate:
        """Build a SourceCandidate from a Hugging Face search hit."""

        payload = hit.raw_payload
        dataset_id = payload.get("id") or hit.title
        tags = list(payload.get("tags") or [])
        modality = self._infer_modality(tags)
        task_type = self._infer_task_type(tags)
        language = self._infer_language(tags)
        score_breakdown = self._score_candidate(
            topic_profile=topic_profile,
            title=dataset_id,
            description=payload.get("description") or hit.snippet,
            tags=tags,
            platform_trust=1.0,
            modality=modality,
            task_type=task_type,
            language=language,
        )

        return SourceCandidate(
            source_type=SourceType.HF_DATASET,
            name=dataset_id,
            dataset_id=dataset_id,
            normalized_source_id=f"huggingface:{dataset_id.casefold()}",
            description=payload.get("description") or hit.snippet,
            url=hit.url,
            platform=hit.provider.value,
            tags=tags,
            modality=modality,
            task_type=task_type,
            language=language,
            estimated_rows=payload.get("downloads"),
            supports_labels=None,
            relevance_score=score_breakdown["total"],
            score_breakdown=score_breakdown,
            evidence_refs=[evidence_item.id],
            is_executable=True,
            pros=["Public dataset hub"],
            cons=["Dataset card quality varies"],
            risks=["Fields may still need normalization"],
        )

    def _build_github_candidate(
        self,
        topic_profile: TopicProfile,
        hit: RawSearchHit,
        evidence_item: SearchEvidence,
    ) -> SourceCandidate:
        """Build a SourceCandidate from a GitHub repository search hit."""

        payload = hit.raw_payload
        tags = list(payload.get("topics") or [])
        description = payload.get("description") or hit.snippet
        modality = self._infer_modality(tags)
        task_type = self._infer_task_type(tags)
        language = None
        score_breakdown = self._score_candidate(
            topic_profile=topic_profile,
            title=payload.get("full_name") or hit.title,
            description=description,
            tags=tags,
            platform_trust=0.85,
            modality=modality,
            task_type=task_type,
            language=language,
        )

        return SourceCandidate(
            source_type=SourceType.GITHUB_DATASET,
            name=payload.get("full_name") or hit.title,
            normalized_source_id=f"github:{(payload.get('full_name') or hit.title).casefold()}",
            repo_url=hit.url,
            branch=payload.get("default_branch"),
            description=description,
            url=hit.url,
            platform=hit.provider.value,
            tags=tags,
            modality=modality,
            task_type=task_type,
            language=language,
            relevance_score=score_breakdown["total"],
            score_breakdown=score_breakdown,
            evidence_refs=[evidence_item.id],
            is_executable=True,
            pros=["Rich repo metadata", "Often includes ingestion scripts"],
            cons=["Repository may not contain ready-to-use datasets"],
            risks=["Repository inspection may still find no parsable dataset files"],
        )

    def _build_kaggle_candidate(
        self,
        topic_profile: TopicProfile,
        hit: RawSearchHit,
        evidence_item: SearchEvidence,
    ) -> SourceCandidate:
        """Build a SourceCandidate from a Kaggle search hit."""

        payload = hit.raw_payload
        title = payload.get("title") or hit.title
        subtitle = payload.get("subtitle") or payload.get("description") or hit.snippet
        score_breakdown = self._score_candidate(
            topic_profile=topic_profile,
            title=title,
            description=subtitle,
            tags=[],
            platform_trust=0.95,
            modality="tabular" if any(
                term in f"{title} {subtitle}".casefold() for term in ("csv", "table", "tabular")
            ) else None,
            task_type=None,
            language=None,
        )

        return SourceCandidate(
            source_type=SourceType.KAGGLE,
            name=title,
            dataset_ref=payload.get("ref"),
            normalized_source_id=f"kaggle:{(payload.get('ref') or title).casefold()}",
            description=subtitle,
            url=hit.url,
            platform=hit.provider.value,
            modality="tabular" if "csv" in f"{title} {subtitle}".casefold() else None,
            relevance_score=score_breakdown["total"],
            score_breakdown=score_breakdown,
            evidence_refs=[evidence_item.id],
            is_executable=bool(payload.get("ref")),
            non_executable_reason=None if payload.get("ref") else "Kaggle search hit did not expose a dataset reference.",
            pros=["Dataset-focused platform", "Often contains tabular public data"],
            cons=["Credentialed access may be required"],
            risks=["License and usage constraints require review"],
        )

    def _build_web_forum_candidate(
        self,
        topic_profile: TopicProfile,
        hit: RawSearchHit,
        evidence_item: SearchEvidence,
    ) -> SourceCandidate:
        """Build a SourceCandidate from DuckDuckGo-lite web search."""

        payload = hit.raw_payload or {}
        kind = str(payload.get("detected_kind") or "html_page")
        forum_score = float(payload.get("forum_score") or 0.0)
        url = str(payload.get("resolved_url") or hit.url)
        risks = list(payload.get("risks") or [])

        if kind == "api_json":
            source_type = SourceType.API
            endpoint = url
            page_url = None
            is_executable = True
            non_ex: str | None = None
            rationale = (
                "Web discovery hit looks JSON/API-like; APIConnector may work with default GET — "
                "add response_path/params if the payload is nested."
            )
        else:
            source_type = SourceType.SCRAPE
            endpoint = None
            page_url = url
            is_executable = False
            non_ex = (
                "MVP web discovery: HTML page needs selector / scrape recipe or auto_scrape — "
                "not auto-executable."
            )
            rationale = "Public HTML page from web search; confirm extractable content and ToS."

        score_breakdown = self._score_candidate(
            topic_profile=topic_profile,
            title=hit.title,
            description=hit.snippet,
            tags=[],
            platform_trust=0.35,
            modality="text" if forum_score >= 0.45 else None,
            task_type=None,
            language=None,
        )
        boosted = min(1.0, (score_breakdown["total"] or 0.0) + 0.12 * forum_score)
        score_breakdown = {**score_breakdown, "total": round(boosted, 4)}

        name = (hit.title or url)[:220]
        candidate = SourceCandidate(
            source_type=source_type,
            name=name,
            normalized_source_id=f"web_forum:{self._canonicalize_url(url)}",
            description=hit.snippet,
            url=page_url,
            endpoint=endpoint,
            platform=DiscoveryProvider.WEB_FORUM.value,
            tags=[],
            modality="text" if forum_score >= 0.45 else None,
            relevance_score=score_breakdown["total"],
            score_breakdown=score_breakdown,
            evidence_refs=[evidence_item.id],
            selection_rationale=rationale,
            is_executable=is_executable,
            non_executable_reason=non_ex,
            risks=risks,
            content_type_hint="application/json" if kind == "api_json" else "text/html",
        )
        if kind != "api_json" and url and self._auto_scrape_discovery_enabled():
            timeout, render_fallback = self._auto_scrape_probe_params()
            try:
                from agents.data_collection.auto_scrape_requests_html import (
                    apply_probe_to_source_candidate,
                    probe_url,
                )
            except ImportError as exc:
                candidate.execution_mode = "auto_scrape_requests_html"
                candidate.auto_scrape_success = False
                candidate.auto_scrape_reason = f"import failed: {exc}"
                candidate.auto_scrape_preview_count = 0
                candidate.non_executable_reason = (
                    f"{non_ex} Auto-scrape (requests-html) unavailable: {exc}"
                )
            else:
                try:
                    res = probe_url(url, timeout=timeout, render_fallback=render_fallback)
                except Exception as exc:  # noqa: BLE001 — surface probe failures on candidate
                    LOGGER.warning("auto_scrape probe_url raised for %s: %s", url, exc)
                    candidate.execution_mode = "auto_scrape_requests_html"
                    candidate.auto_scrape_success = False
                    candidate.auto_scrape_reason = str(exc)
                    candidate.auto_scrape_preview_count = 0
                    candidate.non_executable_reason = f"{non_ex} Auto-scrape probe error: {exc}"
                else:
                    candidate.execution_mode = "auto_scrape_requests_html"
                    candidate.auto_scrape_success = res.success
                    candidate.auto_scrape_reason = res.reason if not res.success else None
                    candidate.auto_scrape_preview_count = (
                        len(res.preview_rows) if res.success else 0
                    )
                    if res.success:
                        apply_probe_to_source_candidate(candidate, res)
                    else:
                        reason = res.reason or "probe failed"
                        candidate.non_executable_reason = f"{non_ex} Auto-scrape probe: {reason}"
        return candidate

    def _build_devtools_har_candidate(
        self,
        topic_profile: TopicProfile,
        hit: RawSearchHit,
        evidence_item: SearchEvidence,
    ) -> SourceCandidate:
        """Build candidate from manual DevTools / HAR hints."""

        payload = hit.raw_payload or {}
        json_url = str(payload.get("json_url") or "").strip()
        page_url = str(payload.get("page_url") or "").strip()
        method = str(payload.get("method") or "GET").upper()
        headers = dict(payload.get("api_headers") or {})
        kind = str(payload.get("detected_kind") or ("api_json" if json_url else "html_page"))
        risks = list(payload.get("risks") or [])

        if json_url:
            source_type = SourceType.API
            endpoint = json_url
            is_executable = True
            non_ex = None
            rationale = "Manual DevTools/HAR JSON endpoint hint — verify auth and response_path."
            url_field = page_url or json_url
            scraper_spec: dict[str, object] = {"http_method": method}
            if headers:
                scraper_spec["api_headers"] = headers
        else:
            source_type = SourceType.SCRAPE
            endpoint = None
            is_executable = False
            non_ex = "HTML hint only — add selector / scraper_spec or use auto_scrape."
            rationale = "Manual page URL from DevTools — scraping recipe still required."
            url_field = page_url or hit.url
            scraper_spec = {"http_method": method} if method != "GET" else {}

        score_breakdown = self._score_candidate(
            topic_profile=topic_profile,
            title=str(payload.get("label") or hit.title),
            description=hit.snippet,
            tags=[],
            platform_trust=0.7,
            modality="text",
            task_type=None,
            language=None,
        )

        return SourceCandidate(
            source_type=source_type,
            name=str(payload.get("label") or hit.title)[:220],
            normalized_source_id=f"devtools_har:{self._canonicalize_url(url_field)}",
            description=hit.snippet,
            url=url_field if source_type is SourceType.SCRAPE else (page_url or None),
            endpoint=endpoint,
            platform=DiscoveryProvider.DEVTOOLS_HAR.value,
            relevance_score=score_breakdown["total"],
            score_breakdown=score_breakdown,
            evidence_refs=[evidence_item.id],
            selection_rationale=rationale,
            is_executable=is_executable,
            non_executable_reason=non_ex,
            risks=risks,
            scraper_spec=scraper_spec,
            content_type_hint="application/json" if kind == "api_json" else "text/html",
        )

    def _score_candidate(
        self,
        topic_profile: TopicProfile,
        title: str,
        description: str | None,
        tags: list[str],
        platform_trust: float,
        modality: str | None,
        task_type: str | None,
        language: str | None,
    ) -> dict[str, float]:
        """Compute deterministic candidate score and trace."""

        scoring = self.config.get("scoring", {})
        weights = {
            "topical_relevance": scoring.get("topical_relevance", 0.5),
            "platform_trust": scoring.get("platform_trust", 0.2),
            "metadata_richness": scoring.get("metadata_richness", 0.1),
            "modality_fit": scoring.get("modality_fit", 0.1),
            "task_fit": scoring.get("task_fit", 0.05),
            "language_fit": scoring.get("language_fit", 0.05),
        }

        topic_text = " ".join(
            [topic_profile.topic or "", *(topic_profile.discovery_hints.get("keywords", []))]
        ).casefold()
        searchable_text = " ".join([title, description or "", *tags]).casefold()
        topical_match = 1.0 if topic_text and any(token in searchable_text for token in topic_text.split()) else 0.0
        metadata_richness = min(1.0, (len(tags) + (1 if description else 0)) / 6.0)
        modality_fit = 1.0 if topic_profile.modality and topic_profile.modality == modality else 0.0
        task_fit = 1.0 if topic_profile.task_type and topic_profile.task_type == task_type else 0.0
        language_fit = 1.0 if topic_profile.language and topic_profile.language == language else 0.0

        total = (
            weights["topical_relevance"] * topical_match
            + weights["platform_trust"] * platform_trust
            + weights["metadata_richness"] * metadata_richness
            + weights["modality_fit"] * modality_fit
            + weights["task_fit"] * task_fit
            + weights["language_fit"] * language_fit
        )
        return {
            "topical_relevance": round(topical_match, 4),
            "platform_trust": round(platform_trust, 4),
            "metadata_richness": round(metadata_richness, 4),
            "modality_fit": round(modality_fit, 4),
            "task_fit": round(task_fit, 4),
            "language_fit": round(language_fit, 4),
            "total": round(total, 4),
        }

    @staticmethod
    def _build_evidence(hit: RawSearchHit) -> SearchEvidence:
        """Convert a raw search hit into serializable evidence."""

        return SearchEvidence(
            id=f"{hit.provider.value}:{abs(hash((hit.query, hit.url, hit.title)))}",
            provider=hit.provider,
            query=hit.query,
            source_url=hit.url,
            title=hit.title,
            snippet=hit.snippet,
            fetched_at=hit.fetched_at,
            metadata={
                key: value
                for key, value in hit.raw_payload.items()
                if key
                in {
                    "id",
                    "full_name",
                    "description",
                    "tags",
                    "topics",
                    "stargazers_count",
                    "license",
                    "updated_at",
                    "ref",
                    "subtitle",
                    "forum_score",
                    "detected_kind",
                    "risks",
                    "resolved_url",
                    "search_engine",
                    "label",
                    "page_url",
                    "json_url",
                    "method",
                    "api_headers",
                }
            },
        )

    def _build_demo_fallback(self, topic_profile: TopicProfile) -> list[SourceCandidate]:
        """Return explicit demo fallback candidates."""

        fallback_candidates = (
            self.dataset_scout.discover(topic_profile)
            + self.api_scout.discover(topic_profile)
            + self.scrape_scout.discover(topic_profile)
            + self.repo_scout.discover(topic_profile)
        )
        for candidate in fallback_candidates:
            candidate.is_demo_fallback = True
            if not candidate.platform:
                candidate.platform = DiscoveryProvider.DEMO.value
            if candidate.source_type is SourceType.REPOSITORY:
                candidate.is_executable = False
                if candidate.non_executable_reason is None:
                    candidate.non_executable_reason = (
                        "Demo fallback candidate has no execution path yet."
                    )
        return fallback_candidates

    @staticmethod
    def _canonicalize_url(url: str) -> str:
        """Canonicalize URLs for deduplication."""

        if not url:
            return ""
        parts = urlsplit(url)
        filtered_query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=False)))
        return urlunsplit(
            (
                parts.scheme.casefold(),
                parts.netloc.casefold(),
                parts.path.rstrip("/"),
                filtered_query,
                "",
            )
        )

    @staticmethod
    def _infer_modality(tags: list[str]) -> str | None:
        """Infer modality from provider tags."""

        lowered_tags = {tag.casefold() for tag in tags}
        if {"audio", "speech"} & lowered_tags:
            return "audio"
        if {"image", "vision"} & lowered_tags:
            return "image"
        if {"text", "nlp"} & lowered_tags:
            return "text"
        if {"tabular", "csv", "timeseries"} & lowered_tags:
            return "tabular"
        return None

    @staticmethod
    def _infer_task_type(tags: list[str]) -> str | None:
        """Infer task type from provider tags."""

        lowered_tags = {tag.casefold() for tag in tags}
        if {"classification", "text-classification"} & lowered_tags:
            return "classification"
        if {"question-answering", "qa"} & lowered_tags:
            return "qa"
        if {"translation"} & lowered_tags:
            return "translation"
        if {"summarization"} & lowered_tags:
            return "summarization"
        return None

    @staticmethod
    def _infer_language(tags: list[str]) -> str | None:
        """Infer language from provider tags."""

        lowered_tags = {tag.casefold() for tag in tags}
        if "english" in lowered_tags or "en" in lowered_tags:
            return "english"
        if "russian" in lowered_tags or "ru" in lowered_tags:
            return "russian"
        if "multilingual" in lowered_tags:
            return "multilingual"
        return None

    @staticmethod
    def _flatten_provider_queries(query_plan: QueryPlan) -> list[str]:
        """Flatten provider queries for logging convenience."""

        flattened_queries: list[str] = []
        for queries in query_plan.provider_queries.values():
            flattened_queries.extend(queries)
        return flattened_queries


# Example:
# service = SourceDiscoveryService()
# candidates = service.discover(topic_profile)
