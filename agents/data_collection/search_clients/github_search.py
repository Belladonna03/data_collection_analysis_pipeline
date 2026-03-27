from __future__ import annotations

from datetime import datetime, timezone
import logging

import requests

from agents.data_collection.schemas import DiscoveryCapability, DiscoveryProvider, RawSearchHit
from agents.data_collection.search_clients.base import BaseSearchClient, SearchClientError


LOGGER = logging.getLogger(__name__)


class GitHubRepoSearchClient(BaseSearchClient):
    """Search GitHub repositories via the official search API."""

    SEARCH_URL = "https://api.github.com/search/repositories"

    @property
    def provider(self) -> DiscoveryProvider:
        """Return the provider identifier."""

        return DiscoveryProvider.GITHUB

    def check_capability(self) -> DiscoveryCapability:
        """Return whether GitHub search is enabled."""

        if not self.config.enabled:
            return DiscoveryCapability(
                provider=self.provider,
                available=False,
                reason="Provider disabled by config.",
            )
        return DiscoveryCapability(provider=self.provider, available=True)

    def search(self, query: str) -> list[RawSearchHit]:
        """Search GitHub repositories for a query."""

        headers = {
            "Accept": "application/vnd.github+json",
        }
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"

        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": self.config.max_results_per_query,
        }

        LOGGER.info("Fetching GitHub repositories url=%s params=%s", self.SEARCH_URL, params)
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
            raise SearchClientError(f"GitHub repo search failed for query '{query}': {exc}") from exc

        fetched_at = datetime.now(timezone.utc).isoformat()
        hits: list[RawSearchHit] = []
        for item in payload.get("items", [])[: self.config.max_results_per_query]:
            html_url = item.get("html_url")
            full_name = item.get("full_name")
            if not html_url or not full_name:
                continue
            hits.append(
                RawSearchHit(
                    provider=self.provider,
                    query=query,
                    url=html_url,
                    title=full_name,
                    snippet=item.get("description"),
                    raw_payload=item,
                    fetched_at=fetched_at,
                )
            )
        return hits
