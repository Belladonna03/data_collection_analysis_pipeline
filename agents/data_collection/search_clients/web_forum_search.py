from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from urllib.parse import unquote, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

from agents.data_collection.schemas import DiscoveryCapability, DiscoveryProvider, RawSearchHit
from agents.data_collection.search_clients.base import BaseSearchClient, SearchClientError


LOGGER = logging.getLogger(__name__)

_DEFAULT_UA = (
    "Mozilla/5.0 (compatible; PipelineDataCollection/1.0; +https://github.com/) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Safari/537.36"
)

_DENY_PATH_SNIPPETS = (
    "/login",
    "/signin",
    "/sign-in",
    "/signup",
    "/register",
    "/account",
    "/auth/",
    "/oauth",
    "/private",
    "/cart",
    "/checkout",
)

_DENY_HOST_PREFIXES = ("login.", "accounts.", "id.", "signin.")

_FORUM_PATH_HINTS = re.compile(
    r"forum|thread|topic|discussion|community|board|message|posts?/|f=\d+|t=\d+|showthread|viewforum",
    re.I,
)


def _resolve_duckduckgo_redirect(href: str) -> str:
    if not href:
        return href
    try:
        parsed = urlparse(href)
        if "duckduckgo.com" in (parsed.netloc or "").lower() and parsed.path.startswith("/l/"):
            qs = parse_qs(parsed.query)
            inner = (qs.get("uddg") or [None])[0]
            if inner:
                return unquote(inner)
    except Exception:
        pass
    return href


class WebForumSearchClient(BaseSearchClient):
    """MVP web discovery via DuckDuckGo Lite HTML (forum-friendly ranking heuristics)."""

    LITE_URL = "https://lite.duckduckgo.com/lite/"

    @property
    def provider(self) -> DiscoveryProvider:
        return DiscoveryProvider.WEB_FORUM

    def check_capability(self) -> DiscoveryCapability:
        if not self.config.enabled:
            return DiscoveryCapability(
                provider=self.provider,
                available=False,
                reason="Provider disabled in discovery.providers.web_forum.enabled.",
            )
        return DiscoveryCapability(provider=self.provider, available=True)

    def search(self, query: str) -> list[RawSearchHit]:
        q = (query or "").strip()
        if not q:
            return []

        ua = self.config.user_agent or _DEFAULT_UA
        session = requests.Session()
        headers = {"User-Agent": ua, "Accept": "text/html"}
        try:
            response = session.get(
                self.LITE_URL,
                params={"q": q},
                headers=headers,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
        except Exception as exc:
            raise SearchClientError(f"Web forum discovery (DDG lite) failed for query {query!r}: {exc}") from exc

        soup = BeautifulSoup(response.text, "lxml")
        fetched_at = datetime.now(timezone.utc).isoformat()
        hits: list[RawSearchHit] = []
        seen: set[str] = set()

        for anchor in soup.select('a[href^="http"], a[href^="//"]'):
            href = anchor.get("href") or ""
            if href.startswith("//"):
                href = "https:" + href
            href = _resolve_duckduckgo_redirect(href)
            if not href.startswith("http"):
                continue
            low_host = urlparse(href).netloc.casefold()
            if "duckduckgo" in low_host or "spreadprivacy" in low_host:
                continue
            if self._host_denied(low_host):
                continue
            if self._url_denied_path(href):
                continue
            if not self._host_allowed(low_host):
                continue

            key = href.split("#", 1)[0].rstrip("/")
            if key in seen:
                continue
            seen.add(key)

            title = (anchor.get_text() or "").strip() or key
            snippet = None
            row = anchor.find_parent("tr")
            if row:
                tds = row.find_all("td")
                if len(tds) > 1:
                    snippet = tds[-1].get_text(" ", strip=True)[:500] or None

            forum_score = self._forum_likelihood(href, title, snippet or "")
            detected_kind = self._classify_kind(href)
            risks = self._risks_for(href, forum_score, detected_kind)

            hits.append(
                RawSearchHit(
                    provider=self.provider,
                    query=query,
                    url=href,
                    title=title[:500],
                    snippet=snippet,
                    raw_payload={
                        "resolved_url": href,
                        "forum_score": forum_score,
                        "detected_kind": detected_kind,
                        "risks": risks,
                        "search_engine": "duckduckgo_lite",
                    },
                    fetched_at=fetched_at,
                )
            )
            if len(hits) >= self.config.max_results_per_query:
                break

        hits.sort(key=lambda h: float((h.raw_payload or {}).get("forum_score", 0.0)), reverse=True)
        return hits[: self.config.max_results_per_query]

    def _host_denied(self, host: str) -> bool:
        deny = [d.strip().casefold() for d in (self.config.deny_domains or []) if d.strip()]
        for d in deny:
            if d in host or host.endswith("." + d):
                return True
        for prefix in _DENY_HOST_PREFIXES:
            if host.startswith(prefix):
                return True
        return False

    def _host_allowed(self, host: str) -> bool:
        allow = [a.strip().casefold() for a in (self.config.allow_domains or []) if a.strip()]
        if not allow:
            return True
        return any(host == a or host.endswith("." + a) for a in allow)

    @staticmethod
    def _url_denied_path(url: str) -> bool:
        path = urlparse(url).path.casefold()
        return any(snippet in path for snippet in _DENY_PATH_SNIPPETS)

    @staticmethod
    def _forum_likelihood(url: str, title: str, snippet: str) -> float:
        text = f"{url} {title} {snippet}".lower()
        score = 0.2
        if _FORUM_PATH_HINTS.search(url) or _FORUM_PATH_HINTS.search(title):
            score += 0.5
        if any(x in text for x in ("question", "answers", "discussion", "thread", "community")):
            score += 0.15
        return min(1.0, score)

    @staticmethod
    def _classify_kind(url: str) -> str:
        u = url.lower()
        if u.endswith(".json") or "/api/" in u or "format=json" in u or "/v1/" in u or "/v2/" in u:
            return "api_json"
        if "rss" in u or u.endswith(".xml") or "/feed" in u:
            return "feed"
        return "html_page"

    @staticmethod
    def _risks_for(url: str, forum_score: float, kind: str) -> list[str]:
        risks = [
            "Respect robots.txt and site Terms of Use; discovery does not check ToS automatically.",
            "DuckDuckGoLite HTML layout may change and affect parsing.",
        ]
        if forum_score < 0.35:
            risks.append("Low forum-pattern score — may be a generic landing page, not a discussion thread listing.")
        if kind == "api_json":
            risks.append("JSON endpoint may require auth, pagination, or custom response_path for ingestion.")
        return risks
