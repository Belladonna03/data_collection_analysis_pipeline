from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


class SourceType(str, Enum):
    """Supported source kinds."""

    HF_DATASET = "hf_dataset"
    KAGGLE = "kaggle"
    GITHUB_DATASET = "github_dataset"
    HTTP_FILE = "http_file"
    REPOSITORY = "repository"
    API = "api"
    SCRAPE = "scrape"


class SessionStatus(str, Enum):
    """Session lifecycle states."""

    CLARIFYING = "clarifying"
    DISCOVERING = "discovering"
    AWAITING_APPROVAL = "awaiting_approval"
    COLLECTING = "collecting"
    DONE = "done"
    ERROR = "error"


class ProfileFieldSource(str, Enum):
    """Provenance for topic profile fields."""

    USER_EXPLICIT = "user_explicit"
    INFERRED_HINT = "inferred_hint"
    CONFIRMED_BY_USER = "confirmed_by_user"


class DiscoveryProvider(str, Enum):
    """Supported internet-backed discovery providers."""

    HUGGING_FACE = "huggingface"
    GITHUB = "github"
    KAGGLE = "kaggle"
    WEB_FORUM = "web_forum"
    DEVTOOLS_HAR = "devtools_har"
    EXTERNAL_TOOL = "external_tool"
    TAVILY = "tavily"
    SERPAPI = "serpapi"
    BRAVE = "brave"
    BING = "bing"
    DEMO = "demo"


@dataclass
class TopicProfile:
    """Normalized user requirements."""

    topic: str | None = None
    modality: str | None = None
    task_type: str | None = None
    language: str | None = None
    needs_labels: bool | None = None
    size_target: int | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    discovery_hints: dict[str, Any] = field(default_factory=dict)
    field_provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize provenance for explicitly provided fields."""

        if self.topic is not None and "topic" not in self.field_provenance:
            self.field_provenance["topic"] = ProfileFieldSource.USER_EXPLICIT.value

        for field_name in (
            "modality",
            "task_type",
            "language",
            "needs_labels",
            "size_target",
        ):
            if getattr(self, field_name) is not None and field_name not in self.field_provenance:
                self.field_provenance[field_name] = ProfileFieldSource.CONFIRMED_BY_USER.value


@dataclass
class SourceCandidate:
    """A possible source discovered for collection."""

    source_type: SourceType
    name: str
    normalized_source_id: str | None = None
    dataset_id: str | None = None
    dataset_ref: str | None = None
    repo_url: str | None = None
    branch: str | None = None
    files: list[str] = field(default_factory=list)
    file_patterns: list[str] = field(default_factory=list)
    description: str | None = None
    url: str | None = None
    endpoint: str | None = None
    selector: str | None = None
    platform: str | None = None
    tags: list[str] = field(default_factory=list)
    modality: str | None = None
    task_type: str | None = None
    language: str | None = None
    estimated_rows: int | None = None
    supports_labels: bool | None = None
    relevance_score: float | None = None
    score_breakdown: dict[str, float] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    selection_rationale: str | None = None
    is_executable: bool = True
    non_executable_reason: str | None = None
    is_demo_fallback: bool = False
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    # Optional ingestion / execution hints (mirrors SourceSpec for planner handoff).
    revision: str | None = None
    streaming: bool = False
    subpath: str | None = None
    scraper_runtime: str | None = None
    scraper_spec: dict[str, Any] = field(default_factory=dict)
    generated_code: str | None = None
    requires_js: bool = False
    allowed_domains: list[str] = field(default_factory=list)
    content_type_hint: str | None = None
    # Auto-scrape (requests-html heuristics) metadata; optional for snapshot compatibility.
    execution_mode: str | None = None
    auto_scrape_success: bool | None = None
    auto_scrape_reason: str | None = None
    auto_scrape_preview_count: int | None = None


@dataclass
class RawSearchHit:
    """Raw provider result before candidate normalization."""

    provider: DiscoveryProvider
    query: str
    url: str
    title: str
    snippet: str | None = None
    raw_payload: dict[str, Any] = field(default_factory=dict)
    fetched_at: str = ""


@dataclass
class SearchEvidence:
    """Normalized evidence describing one fetched search hit."""

    id: str
    provider: DiscoveryProvider
    query: str
    source_url: str
    title: str
    snippet: str | None = None
    fetched_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryCapability:
    """Capability status for one provider."""

    provider: DiscoveryProvider
    available: bool
    reason: str = ""


@dataclass
class DiscoveryJournal:
    """Detailed discovery trace for logging and artifact dumps."""

    queries: list[str] = field(default_factory=list)
    query_plan: dict[str, Any] = field(default_factory=dict)
    provider_capabilities: list[DiscoveryCapability] = field(default_factory=list)
    provider_logs: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[SearchEvidence] = field(default_factory=list)
    raw_hits: list[RawSearchHit] = field(default_factory=list)
    ranking_log: list[dict[str, Any]] = field(default_factory=list)
    used_demo_fallback: bool = False
    demo_fallback_reason: str = ""


@dataclass
class QueryPlan:
    """Structured provider-aware search plan."""

    normalized_goal: str
    domain_terms: list[str] = field(default_factory=list)
    asset_terms: list[str] = field(default_factory=list)
    provider_queries: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class SourceSpec:
    """Executable source description."""

    id: str
    type: SourceType
    name: str
    dataset_id: str | None = None
    dataset_ref: str | None = None
    repo_url: str | None = None
    branch: str | None = None
    split: str | None = None
    subset: str | None = None
    files: list[str] = field(default_factory=list)
    file_patterns: list[str] = field(default_factory=list)
    max_depth: int | None = None
    sample_size: int | None = None
    # Discovery / planner hint for proportional merge budgets (optional).
    estimated_rows: int | None = None
    url: str | None = None
    endpoint: str | None = None
    file_format: str | None = None
    compression: str | None = None
    sheet_name: str | int | None = None
    headers: dict[str, str] = field(default_factory=dict)
    method: str = "GET"
    response_path: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    selector: str | None = None
    attributes_to_extract: list[str] = field(default_factory=list)
    follow_links: bool = False
    item_link_selector: str | None = None
    field_map: dict[str, str] = field(default_factory=dict)
    label_map: dict[Any, str] = field(default_factory=dict)
    pagination: dict[str, Any] = field(default_factory=dict)
    is_executable: bool = True
    non_executable_reason: str | None = None
    enabled: bool = True
    # Hugging Face: dataset revision (git ref) and streaming mode for load_dataset.
    revision: str | None = None
    streaming: bool = False
    # GitHub: optional subdirectory under repo root to scope file discovery.
    subpath: str | None = None
    # Generated / configurable scraper pipeline (runtime is e.g. "python", "playwright").
    scraper_runtime: str | None = None
    scraper_spec: dict[str, Any] = field(default_factory=dict)
    generated_code: str | None = None
    requires_js: bool = False
    allowed_domains: list[str] = field(default_factory=list)
    # HTTP downloads: hint when URL has no file extension (e.g. Content-Type from HEAD).
    content_type_hint: str | None = None
    # ScrapeConnector: "html" = CSS *selector* on fetched HTML (default); "json" = URL returns JSON,
    # records via *response_path* (same dot syntax as API) or root list/object.
    # Future devtools/HAR-assisted flows can point at the same JSON endpoints without a separate scraper.
    scrape_content_mode: str = "html"
    # APIConnector: optional JSON serializable body for POST/PUT/PATCH (requests ``json=``).
    json_body: dict[str, Any] | None = None


@dataclass
class CollectionPlan:
    """Approved collection strategy."""

    topic_profile: TopicProfile
    sources: list[SourceSpec] = field(default_factory=list)
    rationale: str = ""
    expected_schema: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Lightweight collect-time checks (schema/nulls). Duplicate analysis is deferred to quality."""

    missing_columns: list[str] = field(default_factory=list)
    null_stats: dict[str, float] = field(default_factory=dict)
    duplicate_count: int = 0  # kept for compatibility; always 0 — use DataQualityAgent.detect_issues
    warnings: list[str] = field(default_factory=list)


@dataclass
class CollectionResult:
    """Collected artifacts and validation output."""

    dataframe: pd.DataFrame | None = None
    per_source_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    validation_report: ValidationReport = field(default_factory=ValidationReport)


@dataclass
class AutoScrapeResult:
    """Outcome of :meth:`DataCollectionAgent.auto_scrape` (inspect → spec → codegen → runner)."""

    success: bool
    dataframe: pd.DataFrame
    validation_report: ValidationReport = field(default_factory=ValidationReport)
    error: str | None = None
    scraper_spec: dict[str, Any] = field(default_factory=dict)
    generated_code: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
