from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from agents.data_collection.connectors.base import BaseConnector, SourceSpecValidationError


# Machine-detectable hint for orchestration/docs: last-resort path is scrape / generated scraper.
SCRAPER_FALLBACK_INGESTION_TOKEN = "[ingestion_fallback:scrape]"


class HTTPIngestionRoutingError(SourceSpecValidationError):
    """Raised when the URL is an HTML document or landing page, not a direct tabular file."""

    pass
from agents.data_collection.connectors.file_utils import (
    apply_light_mapping,
    apply_sample_size,
    collect_candidate_files,
    content_type_indicates_html,
    download_http_asset,
    file_snippet_looks_like_html,
    load_data_files,
    select_data_files,
)
from agents.data_collection.schemas import SourceSpec, SourceType

_CONTENT_TYPE_TO_FORMAT: dict[str, str] = {
    "text/csv": "csv",
    "application/csv": "csv",
    "text/tab-separated-values": "tsv",
    "application/json": "json",
    "application/x-ndjson": "jsonl",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.apache.parquet": "parquet",
}

_LANDING_PATH_SUFFIXES = (".html", ".htm", ".php", ".asp", ".aspx", ".jsp")


def _format_from_content_type_header(content_type: str | None) -> str | None:
    if not content_type:
        return None
    primary = content_type.split(";")[0].strip().casefold()
    if primary in {"application/zip", "application/x-zip-compressed"}:
        return None
    return _CONTENT_TYPE_TO_FORMAT.get(primary)


class HTTPFileConnector(BaseConnector):
    """Connector for direct public downloadable files (not HTML landing pages)."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    @property
    def connector_name(self) -> str:
        """Return the connector name."""

        return "http_file"

    def can_execute(self, source_spec: SourceSpec) -> tuple[bool, str | None]:
        """Reject obvious web-page URLs using path suffix and optional HEAD Content-Type."""

        valid, reason = super().can_execute(source_spec)
        if not valid:
            return valid, reason

        url = source_spec.url or ""
        parsed_path = urlparse(url).path.casefold()
        if any(parsed_path.endswith(suffix) for suffix in _LANDING_PATH_SUFFIXES):
            return (
                False,
                "URL path looks like a web page (HTML/PHP/ASP), not a direct downloadable data file. "
                "This is a scraper-path candidate (ingestion fallback after open datasets and API), "
                f"not a broken connector. {SCRAPER_FALLBACK_INGESTION_TOKEN}",
            )

        try:
            session = requests.Session()
            response = session.head(
                url,
                timeout=self.timeout,
                allow_redirects=True,
                headers=source_spec.headers or {},
            )
            if response.status_code == 200 and content_type_indicates_html(response.headers.get("Content-Type")):
                return (
                    False,
                    "URL points to an HTML page rather than a direct downloadable data file "
                    "(server Content-Type indicates HTML). Route to scrape / generated scraper as the last "
                    f"ingestion fallback, not the HTTP file connector. {SCRAPER_FALLBACK_INGESTION_TOKEN}",
                )
        except requests.RequestException:
            pass

        return True, None

    @staticmethod
    def _merged_file_patterns(source_spec: SourceSpec) -> list[str]:
        merged: list[str] = []
        for item in list(source_spec.files or []) + list(source_spec.file_patterns or []):
            if item and str(item).strip():
                merged.append(str(item).strip())
        return merged

    def collect(self, source_spec: SourceSpec) -> pd.DataFrame:
        """Download a public file URL and parse it into a DataFrame."""

        self.validate_source_spec(source_spec)
        if self._coerce_source_type(source_spec.type) is not SourceType.HTTP_FILE:
            raise SourceSpecValidationError(
                "HTTPFileConnector only supports source type 'http_file'."
            )
        if not source_spec.enabled:
            return pd.DataFrame()

        url = source_spec.url or ""

        with tempfile.TemporaryDirectory(prefix="http_file_") as temp_dir:
            temp_path = Path(temp_dir)
            try:
                download = download_http_asset(
                    url,
                    temp_path,
                    timeout=self.timeout,
                    headers=source_spec.headers,
                )
            except requests.HTTPError as exc:
                code = getattr(getattr(exc, "response", None), "status_code", None)
                raise SourceSpecValidationError(
                    f"HTTP download failed for '{url}' (HTTP status {code})."
                ) from exc
            except requests.RequestException as exc:
                raise SourceSpecValidationError(
                    f"HTTP download failed for '{url}': {exc}"
                ) from exc

            local_path = download.path
            response_ct = download.content_type

            if content_type_indicates_html(response_ct):
                raise HTTPIngestionRoutingError(
                    f"The URL '{url}' serves HTML (Content-Type: {response_ct!r}) rather than a direct "
                    "downloadable data file. Treat as a scraper-path / landing-page candidate (last ingestion "
                    f"fallback), not a tabular file download. {SCRAPER_FALLBACK_INGESTION_TOKEN}"
                )

            if local_path.suffix.casefold() in {".html", ".htm"}:
                raise HTTPIngestionRoutingError(
                    f"The downloaded resource at '{url}' was saved as '{local_path.name}', which looks like "
                    "an HTML landing page rather than a tabular data file. "
                    f"{SCRAPER_FALLBACK_INGESTION_TOKEN}"
                )

            effective_format = (
                source_spec.file_format
                or _format_from_content_type_header(source_spec.content_type_hint)
                or _format_from_content_type_header(response_ct)
            )

            candidates = collect_candidate_files(local_path)

            if (
                candidates == [local_path]
                and local_path.is_file()
                and file_snippet_looks_like_html(local_path)
            ):
                raise HTTPIngestionRoutingError(
                    f"The file from '{url}' looks like HTML (landing page) even though the name suggests "
                    f"tabular data. {SCRAPER_FALLBACK_INGESTION_TOKEN}"
                )

            if not candidates and local_path.is_file() and file_snippet_looks_like_html(local_path):
                raise HTTPIngestionRoutingError(
                    f"The downloaded bytes from '{url}' look like an HTML landing page, not tabular data. "
                    f"{SCRAPER_FALLBACK_INGESTION_TOKEN}"
                )

            if not candidates and local_path.is_file() and effective_format:
                candidates = [local_path]

            if not candidates:
                if local_path.is_file() and zipfile.is_zipfile(local_path):
                    raise SourceSpecValidationError(
                        f"The archive downloaded from '{url}' contains no parsable tabular files "
                        "(expected .csv, .tsv, .json, .jsonl, .parquet, .xls, or .xlsx inside the zip)."
                    )
                raise SourceSpecValidationError(
                    f"No parsable tabular files were found after downloading '{url}'. "
                    "The response may not be a direct dataset file."
                )

            pattern_list = self._merged_file_patterns(source_spec)
            selected = select_data_files(candidates, preferred_patterns=pattern_list if pattern_list else None)
            if not selected:
                raise SourceSpecValidationError(
                    f"Requested file patterns {pattern_list!r} did not match any parsable files "
                    f"under '{url}'."
                )

            dataframe = load_data_files(
                selected,
                file_format=effective_format,
                compression=source_spec.compression,
                sheet_name=source_spec.sheet_name,
                max_rows=source_spec.sample_size,
            )
            dataframe = apply_sample_size(dataframe, source_spec.sample_size)
            return self.normalize_schema(dataframe, source_spec)

    def normalize_schema(self, df: pd.DataFrame, source_spec: SourceSpec) -> pd.DataFrame:
        """Apply lightweight field and label mapping."""

        return apply_light_mapping(df, source_spec)
