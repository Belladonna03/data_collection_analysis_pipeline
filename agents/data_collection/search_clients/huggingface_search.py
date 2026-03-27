from __future__ import annotations

from datetime import datetime, timezone
import logging

import requests

from agents.data_collection.schemas import DiscoveryCapability, DiscoveryProvider, RawSearchHit
from agents.data_collection.search_clients.base import BaseSearchClient, SearchClientError


LOGGER = logging.getLogger(__name__)


class HuggingFaceSearchClient(BaseSearchClient):
    """Search Hugging Face datasets via the public API."""

    SEARCH_URL = "https://huggingface.co/api/datasets"

    @property
    def provider(self) -> DiscoveryProvider:
        """Return the provider identifier."""

        return DiscoveryProvider.HUGGING_FACE

    def check_capability(self) -> DiscoveryCapability:
        """Return whether Hugging Face search is enabled."""

        if not self.config.enabled:
            return DiscoveryCapability(
                provider=self.provider,
                available=False,
                reason="Provider disabled by config.",
            )
        return DiscoveryCapability(provider=self.provider, available=True)

    def search(self, query: str) -> list[RawSearchHit]:
        """Search Hugging Face datasets for a query."""

        headers = {}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"

        params = {
            "search": query,
            "limit": self.config.max_results_per_query,
            "full": "true",
        }

        LOGGER.info("Fetching Hugging Face datasets url=%s params=%s", self.SEARCH_URL, params)
        try:
            response = requests.get(
                self.SEARCH_URL,
                params=params,
                headers=headers,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise SearchClientError(f"Hugging Face search failed for query '{query}': {exc}") from exc

        fetched_at = datetime.now(timezone.utc).isoformat()
        hits: list[RawSearchHit] = []
        for item in payload[: self.config.max_results_per_query]:
            dataset_id = item.get("id") or item.get("_id")
            if not dataset_id:
                continue
            hits.append(
                RawSearchHit(
                    provider=self.provider,
                    query=query,
                    url=f"https://huggingface.co/datasets/{dataset_id}",
                    title=dataset_id,
                    snippet=item.get("description"),
                    raw_payload=item,
                    fetched_at=fetched_at,
                )
            )
        return hits
