from __future__ import annotations

from typing import Any

import pandas as pd

from agents.data_collection.connectors.base import BaseConnector, SourceSpecValidationError
from agents.data_collection.json_records import extract_json_records
from agents.data_collection.schemas import SourceSpec, SourceType


class APIConnector(BaseConnector):
    """Connector for paginated JSON APIs."""

    def __init__(self, timeout: float = 30.0, retries: int = 2) -> None:
        self.timeout = timeout
        self.retries = retries

    @property
    def connector_name(self) -> str:
        """Return the connector name."""

        return "api"

    def collect(self, source_spec: SourceSpec) -> pd.DataFrame:
        """Fetch records from an API endpoint and return a DataFrame."""

        self.validate_source_spec(source_spec)
        if self._coerce_source_type(source_spec.type) is not SourceType.API:
            raise SourceSpecValidationError(
                "APIConnector only supports source type 'api'."
            )
        if not source_spec.enabled:
            return pd.DataFrame()

        requests = self._import_requests()
        pagination = self._build_pagination_config(source_spec.pagination)

        all_records: list[dict[str, Any]] = []
        current_page = pagination["start_page"]
        cap = int(source_spec.sample_size) if source_spec.sample_size is not None else None

        for _ in range(pagination["max_pages"]):
            request_params = dict(source_spec.params)
            if pagination["page_param"]:
                request_params[pagination["page_param"]] = current_page
            if pagination["page_size_param"] and pagination["page_size"] is not None:
                request_params[pagination["page_size_param"]] = pagination["page_size"]

            payload = self._perform_request(
                requests=requests,
                method=source_spec.method,
                endpoint=source_spec.endpoint or "",
                params=request_params,
                headers=source_spec.headers,
                json_body=getattr(source_spec, "json_body", None),
            )
            page_records = self._extract_records(payload, source_spec.response_path)
            if not page_records:
                if pagination["stop_when_empty"]:
                    break
                current_page += 1
                continue

            all_records.extend(page_records)
            current_page += 1

            if cap is not None and len(all_records) >= cap:
                all_records = all_records[:cap]
                break

            if pagination["page_param"] is None:
                break

        if not all_records:
            raise ValueError(
                f"API source '{source_spec.name}' returned an empty result."
            )

        df = pd.DataFrame(all_records)
        if df.empty:
            raise ValueError(
                f"API source '{source_spec.name}' returned an empty result."
            )

        if source_spec.sample_size is not None:
            df = df.head(source_spec.sample_size).copy()

        return self.normalize_schema(df, source_spec)

    def normalize_schema(self, df: pd.DataFrame, source_spec: SourceSpec) -> pd.DataFrame:
        """Apply lightweight unified field and label mapping."""

        normalized_df = df.copy()

        for unified_field, source_field in source_spec.field_map.items():
            if source_field in normalized_df.columns:
                normalized_df[unified_field] = normalized_df[source_field]

        if "label" in normalized_df.columns and source_spec.label_map:
            normalized_df["label"] = normalized_df["label"].map(
                lambda value: source_spec.label_map.get(value, value)
            )

        return normalized_df

    def _perform_request(
        self,
        requests: Any,
        method: str,
        endpoint: str,
        params: dict[str, Any],
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        """Perform one HTTP request with simple retries."""

        last_error: Exception | None = None
        normalized_method = method.upper()

        for attempt in range(self.retries + 1):
            try:
                request_kw: dict[str, Any] = {
                    "method": normalized_method,
                    "url": endpoint,
                    "headers": headers,
                    "timeout": self.timeout,
                }
                if json_body is not None:
                    request_kw["json"] = json_body
                    request_kw["params"] = params
                else:
                    request_kw["params"] = params
                response = requests.request(**request_kw)
                if response.status_code >= 400:
                    raise ValueError(
                        f"API request failed with status code {response.status_code} "
                        f"for endpoint '{endpoint}'."
                    )
                try:
                    return response.json()
                except ValueError as exc:
                    raise ValueError(
                        f"API response from '{endpoint}' is not valid JSON."
                    ) from exc
            except Exception as exc:
                last_error = exc
                if attempt >= self.retries:
                    raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("API request failed without an explicit error.")

    def _extract_records(
        self,
        payload: Any,
        response_path: str | None,
    ) -> list[dict[str, Any]]:
        """Extract a list of records from JSON payload."""

        return extract_json_records(payload, response_path)

    @staticmethod
    def _build_pagination_config(pagination: dict[str, Any]) -> dict[str, Any]:
        """Build pagination settings with defaults."""

        return {
            "page_param": pagination.get("page_param"),
            "start_page": pagination.get("start_page", 1),
            "max_pages": pagination.get("max_pages", 1),
            "page_size_param": pagination.get("page_size_param"),
            "page_size": pagination.get("page_size"),
            "stop_when_empty": pagination.get("stop_when_empty", True),
        }

    @staticmethod
    def _import_requests():
        """Import requests lazily."""

        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "APIConnector requires the 'requests' package. "
                "Install it with: pip install requests"
            ) from exc
        return requests


# Example:
# source_spec = SourceSpec(
#     id="reviews-api",
#     type=SourceType.API,
#     name="reviews",
#     endpoint="https://api.example.com/reviews",
#     params={"lang": "en"},
#     headers={"Authorization": "Bearer token"},
#     method="GET",
#     response_path="results.items",
#     field_map={"text": "review_text", "label": "sentiment"},
#     label_map={0: "negative", 1: "positive"},
#     pagination={
#         "page_param": "page",
#         "start_page": 1,
#         "max_pages": 5,
#         "page_size_param": "limit",
#         "page_size": 100,
#         "stop_when_empty": True,
#     },
# )
