from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from agents.data_collection.schemas import SourceSpec, SourceType
from agents.data_collection.scraper_spec_execution import source_spec_uses_structured_requests_html_scraper

if TYPE_CHECKING:
    import pandas as pd


class SourceSpecValidationError(ValueError):
    """Raised when a source spec is invalid for a connector."""


class BaseConnector(ABC):
    """Base interface for all data source connectors."""

    @property
    @abstractmethod
    def connector_name(self) -> str:
        """Return a human-readable connector name."""

    @abstractmethod
    def collect(self, source_spec: SourceSpec) -> pd.DataFrame:
        """Collect data for the given source spec."""

    def normalize_schema(self, df: pd.DataFrame, source_spec: SourceSpec) -> pd.DataFrame:
        """Normalize a collected dataframe schema."""

        return df

    def can_execute(self, source_spec: SourceSpec) -> tuple[bool, str | None]:
        """Return whether the source spec is executable by this connector."""

        try:
            self.validate_source_spec(source_spec)
        except SourceSpecValidationError as exc:
            return False, str(exc)
        return True, None

    def validate_source_spec(self, source_spec: SourceSpec) -> None:
        """Validate source spec fields required by the connector."""

        source_type = self._coerce_source_type(source_spec.type)

        if source_type is SourceType.HF_DATASET:
            if not (source_spec.dataset_id or source_spec.name.strip()):
                raise SourceSpecValidationError(
                    "Hugging Face source spec requires a non-empty 'name' or 'dataset_id'."
                )

        if source_type is SourceType.KAGGLE:
            if not (source_spec.dataset_ref or source_spec.name.strip()):
                raise SourceSpecValidationError(
                    "Kaggle source spec requires a non-empty 'dataset_ref' or 'name'."
                )

        if source_type is SourceType.GITHUB_DATASET:
            if not (source_spec.repo_url or source_spec.url):
                raise SourceSpecValidationError(
                    "GitHub dataset source spec requires a non-empty 'repo_url' or 'url'."
                )

        if source_type is SourceType.HTTP_FILE:
            if not source_spec.url:
                raise SourceSpecValidationError(
                    "HTTP file source spec requires a non-empty 'url'."
                )

        if source_type is SourceType.API and not source_spec.endpoint:
            raise SourceSpecValidationError(
                "API source spec requires a non-empty 'endpoint'."
            )

        if source_type is SourceType.SCRAPE:
            if not source_spec.url:
                raise SourceSpecValidationError(
                    "Scrape source spec requires a non-empty 'url'."
                )
            mode = (getattr(source_spec, "scrape_content_mode", None) or "html").strip().lower()
            if mode in {"json", "json_api", "application_json"}:
                mode = "json"
            if mode == "html":
                has_selector = bool((source_spec.selector or "").strip())
                has_structured = source_spec_uses_structured_requests_html_scraper(source_spec)
                if not has_selector and not has_structured:
                    raise SourceSpecValidationError(
                        "Scrape source spec in html mode requires a non-empty 'selector', "
                        "or a requests-html ``scraper_spec`` with ``item_selector`` and ``fields`` "
                        "(or use scrape_content_mode='json' for JSON URLs)."
                    )
            elif mode == "json":
                pass
            else:
                raise SourceSpecValidationError(
                    f"Unsupported scrape_content_mode {source_spec.scrape_content_mode!r}; "
                    "use 'html' or 'json'."
                )

    @staticmethod
    def _coerce_source_type(value: SourceType | str) -> SourceType:
        """Convert raw source type into SourceType."""

        if isinstance(value, SourceType):
            return value

        try:
            return SourceType(value)
        except ValueError as exc:
            raise SourceSpecValidationError(
                f"Unsupported source type: {value!r}."
            ) from exc
