from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from agents.data_collection.schemas import DiscoveryCapability, DiscoveryProvider, RawSearchHit
from agents.data_collection.search_clients.base import BaseSearchClient


LOGGER = logging.getLogger(__name__)


class DevtoolsHarDiscoveryClient(BaseSearchClient):
    """Build candidates from manual hints (page URL, JSON endpoint from DevTools / HAR).

    Does not crawl the open web: it materializes configured hints into SourceCandidate rows
    via :meth:`agents.data_collection.discovery.SourceDiscoveryService._normalize_hit_to_candidate`.
    """

    supports_empty_query_list: bool = True

    @property
    def provider(self) -> DiscoveryProvider:
        return DiscoveryProvider.DEVTOOLS_HAR

    def _hints(self) -> list[dict[str, Any]]:
        raw = self.config.metadata.get("hints")
        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                out.append(dict(item))
        return out

    def check_capability(self) -> DiscoveryCapability:
        if not self.config.enabled:
            return DiscoveryCapability(
                provider=self.provider,
                available=False,
                reason="Provider disabled in discovery.providers.devtools_har.enabled.",
            )
        if not self._hints():
            return DiscoveryCapability(
                provider=self.provider,
                available=False,
                reason="No discovery.providers.devtools_har.hints configured.",
            )
        return DiscoveryCapability(provider=self.provider, available=True)

    def search(self, query: str) -> list[RawSearchHit]:
        # One batch per discover() run — discovery may call search() for multiple planned strings;
        # emit hints only once to avoid duplicates.
        if getattr(self, "_hints_emitted", False):
            return []
        self._hints_emitted = True

        fetched_at = datetime.now(timezone.utc).isoformat()
        hits: list[RawSearchHit] = []
        for hint in self._hints():
            label = str(hint.get("label") or hint.get("name") or "devtools_hint").strip()
            page_url = (hint.get("page_url") or hint.get("page") or "").strip()
            json_url = (hint.get("json_url") or hint.get("endpoint") or hint.get("api_url") or "").strip()
            method = str(hint.get("method") or "GET").upper()
            notes = str(hint.get("notes") or "").strip()
            headers = hint.get("headers") if isinstance(hint.get("headers"), dict) else {}

            primary = json_url or page_url
            if not primary:
                LOGGER.warning("Skipping devtools hint without page_url/json_url: %s", hint)
                continue

            detected_kind = "api_json" if json_url else "html_page"
            risks = [
                "Manual DevTools/HAR hint: verify CORS, auth cookies, and rate limits before production collection.",
                "response_path / field_map may still be required for API ingestion.",
            ]
            if notes:
                risks.append(f"Operator notes: {notes[:240]}")

            hits.append(
                RawSearchHit(
                    provider=self.provider,
                    query=str(query or "__devtools_hints__"),
                    url=primary,
                    title=label,
                    snippet=notes or None,
                    raw_payload={
                        "label": label,
                        "page_url": page_url or None,
                        "json_url": json_url or None,
                        "method": method,
                        "api_headers": headers,
                        "detected_kind": detected_kind,
                        "risks": risks,
                    },
                    fetched_at=fetched_at,
                )
            )
        return hits
