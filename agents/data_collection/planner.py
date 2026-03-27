from __future__ import annotations

from agents.data_collection.schemas import (
    CollectionPlan,
    SourceCandidate,
    SourceSpec,
    SourceType,
    TopicProfile,
)

# Execution / planning tie-break: lower number = earlier in the ingestion fallback chain.
# Scraping (including generated scraper) is intentionally last among executable paths.
INGESTION_FALLBACK_PRIORITY: dict[SourceType, int] = {
    SourceType.HF_DATASET: 0,
    SourceType.KAGGLE: 1,
    SourceType.GITHUB_DATASET: 2,
    SourceType.HTTP_FILE: 3,
    SourceType.API: 4,
    SourceType.SCRAPE: 5,
    SourceType.REPOSITORY: 6,
}

INGESTION_FALLBACK_ORDER: tuple[SourceType, ...] = tuple(
    sorted(INGESTION_FALLBACK_PRIORITY, key=lambda t: INGESTION_FALLBACK_PRIORITY[t])
)

_OPEN_DATASET_TYPES = frozenset(
    {
        SourceType.HF_DATASET,
        SourceType.KAGGLE,
        SourceType.GITHUB_DATASET,
        SourceType.HTTP_FILE,
    }
)


class StrategyPlanner:
    """Build lightweight collection plans from discovered candidates."""

    @staticmethod
    def order_source_specs_for_execution(sources: list[SourceSpec]) -> list[SourceSpec]:
        """Stable sort of plan sources by :data:`INGESTION_FALLBACK_PRIORITY` (id as tie-breaker)."""

        return sorted(
            sources,
            key=lambda spec: (
                INGESTION_FALLBACK_PRIORITY.get(spec.type, 99),
                spec.id or "",
            ),
        )

    @staticmethod
    def _order_candidates_for_fallback(
        selected: list[SourceCandidate],
        ranked: list[SourceCandidate],
    ) -> list[SourceCandidate]:
        """Order selected candidates for execution: type fallback first, then discovery rank."""

        rank_pos: dict[tuple[str, str], int] = {}
        for position, candidate in enumerate(ranked):
            key = (candidate.source_type.value, candidate.name.casefold())
            if key not in rank_pos:
                rank_pos[key] = position
        return sorted(
            selected,
            key=lambda candidate: (
                INGESTION_FALLBACK_PRIORITY.get(candidate.source_type, 99),
                rank_pos.get((candidate.source_type.value, candidate.name.casefold()), 999),
            ),
        )

    @staticmethod
    def _planning_sort_key(candidate: SourceCandidate) -> tuple[float, int, str]:
        """Sort key: higher relevance first, then planner type priority, then name."""

        score = -(candidate.relevance_score or 0.0)
        priority = INGESTION_FALLBACK_PRIORITY.get(candidate.source_type, 99)
        return (score, priority, candidate.name.casefold())

    @staticmethod
    def _partition_tiers(candidates: list[SourceCandidate]) -> tuple[
        list[SourceCandidate],
        list[SourceCandidate],
        list[SourceCandidate],
    ]:
        """Split ordered candidates into open-dataset, API, and scrape lists (order preserved)."""

        open_data: list[SourceCandidate] = []
        apis: list[SourceCandidate] = []
        scrapes: list[SourceCandidate] = []
        for candidate in candidates:
            if candidate.source_type in _OPEN_DATASET_TYPES:
                open_data.append(candidate)
            elif candidate.source_type is SourceType.API:
                apis.append(candidate)
            elif candidate.source_type is SourceType.SCRAPE:
                scrapes.append(candidate)
        return open_data, apis, scrapes

    def build_plans(
        self,
        topic_profile: TopicProfile,
        candidates: list[SourceCandidate],
    ) -> list[CollectionPlan]:
        """Build up to three collection plans."""

        ranked_candidates = sorted(candidates, key=self._planning_sort_key)
        if not ranked_candidates:
            return [
                CollectionPlan(
                    topic_profile=topic_profile,
                    sources=[],
                    rationale="No candidates were discovered yet.",
                    expected_schema=self._build_expected_schema(topic_profile),
                    warnings=["No source candidates available for planning."],
                )
            ]

        plans: list[CollectionPlan] = []
        open_data, apis, scrapes = self._partition_tiers(ranked_candidates)

        plan_candidate_sets = [
            self._build_balanced_selection(ranked_candidates, open_data, apis, scrapes),
            self._build_open_plus_api_selection(open_data, apis),
            ranked_candidates[:3],
        ]

        seen_signatures: set[tuple[str, ...]] = set()
        for selected_candidates in plan_candidate_sets:
            unique_selection = self._unique_by_name(selected_candidates)
            if not unique_selection:
                continue

            ordered_selection = self._order_candidates_for_fallback(unique_selection, ranked_candidates)

            signature = tuple(
                f"{candidate.source_type.value}:{candidate.name}"
                for candidate in ordered_selection
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            warnings = self._validate_constraints(ordered_selection)
            plans.append(
                CollectionPlan(
                    topic_profile=topic_profile,
                    sources=[
                        self._candidate_to_source_spec(candidate)
                        for candidate in ordered_selection
                    ],
                    rationale=self._build_rationale(ordered_selection),
                    expected_schema=self._build_expected_schema(topic_profile),
                    warnings=warnings,
                )
            )
            if len(plans) >= 3:
                break

        return plans

    def _build_balanced_selection(
        self,
        ranked: list[SourceCandidate],
        open_data: list[SourceCandidate],
        apis: list[SourceCandidate],
        scrapes: list[SourceCandidate],
    ) -> list[SourceCandidate]:
        """Prefer one open-dataset source, then API, then scrape, then top-ranked fill-ins."""

        selection: list[SourceCandidate] = []
        if open_data:
            selection.append(open_data[0])
        if apis:
            selection.append(apis[0])
        if scrapes:
            selection.append(scrapes[0])
        selection.extend(ranked[:3])
        return self._unique_by_name(selection)[:3]

    @staticmethod
    def _build_open_plus_api_selection(
        open_data: list[SourceCandidate],
        apis: list[SourceCandidate],
    ) -> list[SourceCandidate]:
        """Up to two open-dataset sources plus one API (no scrape in this variant)."""

        chunk = open_data[:2]
        if apis:
            chunk = chunk + [apis[0]]
        return chunk

    @staticmethod
    def _validate_constraints(candidates: list[SourceCandidate]) -> list[str]:
        """Validate planning hints and return warnings (non-blocking)."""

        warnings: list[str] = []
        executable_count = sum(1 for candidate in candidates if candidate.is_executable)

        if len(candidates) < 2:
            warnings.append("Plan has fewer than 2 sources.")
        if executable_count == 0:
            warnings.append("Plan contains no executable sources.")
        elif executable_count == 1:
            warnings.append("Plan has only one executable source; source diversity may be limited.")

        return warnings

    @staticmethod
    def _source_spec_id(candidate: SourceCandidate) -> str:
        if candidate.normalized_source_id:
            safe = "".join(
                ch if ch.isalnum() or ch in "._:-" else "_"
                for ch in candidate.normalized_source_id
            ).strip("_")
            return safe or f"{candidate.source_type.value}:unnamed"
        slug = candidate.name.lower().replace(" ", "-")
        return f"{candidate.source_type.value}:{slug}"

    @staticmethod
    def _candidate_to_source_spec(candidate: SourceCandidate) -> SourceSpec:
        """Map discovery candidate fields into SourceSpec without source-specific demo logic."""

        is_executable = candidate.is_executable
        non_executable_reason = candidate.non_executable_reason
        if candidate.source_type is SourceType.REPOSITORY:
            is_executable = False
            non_executable_reason = non_executable_reason or (
                "Repository landing pages are discovery-only; promote to hf_dataset, github_dataset, "
                "or another executable source type before collection."
            )

        # Defaults — only override where the candidate carries real metadata.
        split: str | None = None
        subset: str | None = None
        sample_size: int | None = None
        params: dict = {}
        pagination: dict = {}
        response_path: str | None = None
        field_map: dict = {}
        label_map: dict = {}
        item_link_selector: str | None = None
        max_depth: int | None = None
        file_format: str | None = None
        spec_block = dict(candidate.scraper_spec or {})
        hint_headers = spec_block.get("api_headers")
        headers: dict[str, str] = {}
        if isinstance(hint_headers, dict):
            headers = {str(k): str(v) for k, v in hint_headers.items()}
        method = str(spec_block.get("http_method") or "GET")

        dataset_id = candidate.dataset_id
        dataset_ref = candidate.dataset_ref
        repo_url = candidate.repo_url
        branch = candidate.branch
        url = candidate.url
        endpoint = candidate.endpoint
        selector = candidate.selector
        files = list(candidate.files)
        file_patterns = list(candidate.file_patterns)

        return SourceSpec(
            id=StrategyPlanner._source_spec_id(candidate),
            type=candidate.source_type,
            name=candidate.name,
            dataset_id=dataset_id,
            dataset_ref=dataset_ref,
            repo_url=repo_url,
            branch=branch,
            split=split,
            subset=subset,
            files=files,
            file_patterns=file_patterns,
            max_depth=max_depth,
            sample_size=sample_size,
            estimated_rows=candidate.estimated_rows,
            url=url,
            endpoint=endpoint,
            file_format=file_format,
            method=method,
            response_path=response_path,
            params=params,
            headers=headers,
            selector=selector,
            item_link_selector=item_link_selector,
            field_map=field_map,
            label_map=label_map,
            pagination=pagination,
            is_executable=is_executable,
            non_executable_reason=non_executable_reason,
            enabled=True,
            revision=candidate.revision,
            streaming=candidate.streaming,
            subpath=candidate.subpath,
            scraper_runtime=candidate.scraper_runtime,
            scraper_spec=dict(candidate.scraper_spec),
            generated_code=candidate.generated_code,
            requires_js=candidate.requires_js,
            allowed_domains=list(candidate.allowed_domains),
            content_type_hint=candidate.content_type_hint,
        )

    @staticmethod
    def _build_rationale(candidates: list[SourceCandidate]) -> str:
        """Explain why these sources were selected together."""

        source_names = ", ".join(candidate.name for candidate in candidates)
        source_types = {candidate.source_type for candidate in candidates}

        if source_types.intersection(_OPEN_DATASET_TYPES) and source_types.intersection(
            {SourceType.API, SourceType.SCRAPE}
        ):
            return (
                f"Selected sources {source_names} to combine stable open datasets with fresher "
                "web or API signals. This mix gives a strong bootstrap corpus and a path to "
                "extend coverage beyond static repositories."
            )

        return (
            f"Selected sources {source_names} because they are the highest-ranked candidates for "
            "the current topic profile. They provide an MVP starting point, though source diversity "
            "may need improvement."
        )

    @staticmethod
    def _build_expected_schema(topic_profile: TopicProfile) -> dict[str, str]:
        """Build a lightweight expected schema."""

        schema = {
            "text": "str",
            "source": "str",
            "collected_at": "datetime",
        }
        if topic_profile.modality == "audio":
            schema["audio"] = "str"
        if topic_profile.modality == "image":
            schema["image"] = "str"
        if topic_profile.needs_labels:
            schema["label"] = "str"
        return schema

    @staticmethod
    def _unique_by_name(candidates: list[SourceCandidate]) -> list[SourceCandidate]:
        """Keep first occurrence of each candidate by type and name."""

        unique_candidates: list[SourceCandidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in candidates:
            key = (candidate.source_type.value, candidate.name.casefold())
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)
        return unique_candidates

    def build_plan_from_selected_candidates(
        self,
        topic_profile: TopicProfile,
        candidates_by_key: dict[str, SourceCandidate],
        selected_keys: list[str],
    ) -> CollectionPlan:
        """Build a single execution plan from explicit user-selected discovery candidates.

        *selected_keys* preserves caller order (e.g. ``collect select --ids 2,1``).
        Each key must exist in *candidates_by_key*. Every selected candidate must be
        marked ``is_executable``; otherwise :class:`ValueError` is raised.
        """

        if not selected_keys:
            raise ValueError("At least one candidate key is required.")

        ordered: list[SourceCandidate] = []
        seen_keys: set[str] = set()
        for key in selected_keys:
            if key not in candidates_by_key:
                raise ValueError(f"Unknown candidate_key: {key}")
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidate = candidates_by_key[key]
            if not candidate.is_executable:
                reason = candidate.non_executable_reason or "discovery-only or no connector"
                raise ValueError(
                    f"Candidate '{candidate.name}' ({key}) is not executable and cannot be collected: {reason}"
                )
            ordered.append(candidate)

        ranked_for_order = list(candidates_by_key.values())
        ordered_exec = self._order_candidates_for_fallback(ordered, ranked_for_order)
        warnings = self._validate_constraints(ordered_exec)
        return CollectionPlan(
            topic_profile=topic_profile,
            sources=[self._candidate_to_source_spec(candidate) for candidate in ordered_exec],
            rationale=self._build_rationale(ordered_exec),
            expected_schema=self._build_expected_schema(topic_profile),
            warnings=warnings,
        )


# Example:
# planner = StrategyPlanner()
# plans = planner.build_plans(topic_profile, candidates)
