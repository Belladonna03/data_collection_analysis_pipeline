from __future__ import annotations

import json
from typing import Any
from urllib.parse import urljoin, urlparse

import pandas as pd

from agents.data_collection.connectors.base import BaseConnector, SourceSpecValidationError
from agents.data_collection.json_records import extract_json_records
from agents.data_collection.schemas import SourceSpec, SourceType
from agents.data_collection.scraper_spec import ScraperSpec
from agents.data_collection.scraper_spec_execution import (
    enrich_structured_rows,
    extract_rows_from_document,
    infer_extraction_mode,
    source_spec_uses_structured_requests_html_scraper,
)


class ScrapeConnector(BaseConnector):
    """Connector for simple list-page scraping."""

    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout

    @property
    def connector_name(self) -> str:
        """Return the connector name."""

        return "scrape"

    def collect(self, source_spec: SourceSpec) -> pd.DataFrame:
        """Scrape HTML via CSS selector, or fetch a JSON document from *url* (``scrape_content_mode='json'``)."""

        self.validate_source_spec(source_spec)
        if self._coerce_source_type(source_spec.type) is not SourceType.SCRAPE:
            raise SourceSpecValidationError(
                "ScrapeConnector only supports source type 'scrape'."
            )
        if not source_spec.enabled:
            return pd.DataFrame()

        if self._normalized_scrape_mode(source_spec) == "json":
            return self._collect_json_document(source_spec)

        if source_spec_uses_structured_requests_html_scraper(source_spec):
            return self._collect_structured_requests_html(source_spec)

        pagination = self._build_pagination_config(source_spec.pagination)
        current_url = source_spec.url or ""
        collected_items: list[dict[str, Any]] = []
        cap = int(source_spec.sample_size) if source_spec.sample_size is not None else None

        for _ in range(pagination["max_pages"]):
            html = self.fetch_page(current_url)
            page_items = self.extract_items(html, current_url, source_spec)
            if not page_items:
                raise ValueError(
                    f"Selector '{source_spec.selector}' returned no items for '{current_url}'."
                )

            if cap is not None:
                need = cap - len(collected_items)
                if need <= 0:
                    break
                collected_items.extend(page_items[:need])
                if len(collected_items) >= cap:
                    break
            else:
                collected_items.extend(page_items)
            next_url = self.extract_next_page_url(html, current_url, source_spec)
            if not next_url:
                break
            current_url = next_url

        if not collected_items:
            raise ValueError(
                f"Scrape source '{source_spec.name}' returned an empty result."
            )

        df = pd.DataFrame(collected_items)
        if df.empty:
            raise ValueError(
                f"Scrape source '{source_spec.name}' returned an empty result."
            )

        if source_spec.sample_size is not None:
            df = df.head(source_spec.sample_size).copy()

        return self.normalize_schema(df, source_spec)

    @staticmethod
    def _host_in_allowed(url: str, allowed_hosts: list[str]) -> bool:
        if not allowed_hosts:
            return True
        host = urlparse(url).netloc.casefold()
        if host.startswith("www."):
            host = host[4:]
        allowed = {h.casefold() for h in allowed_hosts}
        return host in allowed

    def _fetch_requests_html_live_document(self, url: str, requires_js: bool):
        """Return ``(response.html, final_url_str)``."""

        html_session_cls, _ = self._import_requests_html()
        session = html_session_cls()
        response = session.get(url, timeout=self.timeout)
        if response.status_code >= 400:
            raise ValueError(
                f"Failed to load page {url!r} with status code {response.status_code}."
            )
        if requires_js:
            try:
                response.html.render(timeout=int(min(self.timeout, 45)), sleep=0.5)
            except Exception as exc:
                raise ValueError(f"JS render failed for {url!r}: {exc}") from exc
        return response.html, str(response.url)

    def _collect_structured_requests_html(self, source_spec: SourceSpec) -> pd.DataFrame:
        """Collect rows using ``scraper_spec`` field definitions (auto-scrape / requests-html)."""

        raw = dict(source_spec.scraper_spec or {})
        base_url = (source_spec.url or raw.get("entry_url") or "").strip()
        if not base_url:
            raise ValueError("Structured scrape requires a non-empty source_spec.url or scraper_spec.entry_url.")

        merged = {**raw, "entry_url": base_url}
        sel = (raw.get("item_selector") or source_spec.selector or "").strip()
        if sel:
            merged["item_selector"] = sel

        spec = ScraperSpec.from_dict(merged)
        if not (spec.item_selector or "").strip():
            raise ValueError("Structured scrape requires scraper_spec.item_selector (or source_spec.selector).")

        requires_js = bool(spec.requires_js or source_spec.requires_js)
        allowed_hosts = list(spec.allowed_domains or source_spec.allowed_domains or [])
        if not allowed_hosts and base_url:
            host = urlparse(base_url).netloc.casefold()
            if host.startswith("www."):
                host = host[4:]
            if host:
                allowed_hosts = [host]

        fields_payload: list[dict[str, Any]] = [
            {
                "name": f.name,
                "selector": f.selector,
                "extract": f.extract,
                "attribute": f.attribute,
            }
            for f in spec.fields
        ]

        mode = spec.extraction_mode or infer_extraction_mode(merged)

        all_rows: list[dict[str, Any]] = []
        current_url = base_url
        max_pages = max(1, int(spec.max_pages or 1))

        for page_idx in range(max_pages):
            if allowed_hosts and not self._host_in_allowed(current_url, allowed_hosts):
                break
            doc, final_url = self._fetch_requests_html_live_document(current_url, requires_js)
            page_rows = extract_rows_from_document(
                doc,
                item_selector=spec.item_selector,
                fields=fields_payload,
                base_url=final_url,
                extraction_mode=mode,
            )
            if not page_rows:
                if page_idx == 0:
                    raise ValueError(
                        f"Structured scrape: item_selector {spec.item_selector!r} matched no nodes on {final_url!r}."
                    )
                break
            all_rows.extend(enrich_structured_rows(page_rows, landing_url=final_url))

            if spec.pagination_strategy != "link_next" or not spec.pagination_selector:
                break
            nxt_el = doc.find(spec.pagination_selector, first=True)
            if nxt_el is None:
                break
            href = nxt_el.attrs.get("href") if hasattr(nxt_el, "attrs") else None
            next_url = self._resolve_href(final_url, href)
            if not next_url or next_url == current_url:
                break
            if allowed_hosts and not self._host_in_allowed(next_url, allowed_hosts):
                break
            current_url = next_url

        if not all_rows:
            raise ValueError(f"Structured scrape source '{source_spec.name}' returned an empty result.")

        df = pd.DataFrame(all_rows)
        if source_spec.sample_size is not None:
            df = df.head(source_spec.sample_size).copy()

        return self.normalize_schema(df, source_spec)

    @staticmethod
    def _normalized_scrape_mode(source_spec: SourceSpec) -> str:
        raw = (getattr(source_spec, "scrape_content_mode", None) or "html").strip().lower()
        if raw in {"json", "json_api", "application_json"}:
            return "json"
        return "html"

    def _collect_json_document(self, source_spec: SourceSpec) -> pd.DataFrame:
        """GET single URL; parse JSON body; rows via :func:`extract_json_records` (``response_path`` optional)."""

        url = source_spec.url or ""
        body = self.fetch_page(url)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"scrape_content_mode=json expected a JSON body from {url!r}."
            ) from exc

        records = extract_json_records(payload, source_spec.response_path)
        if not records:
            raise ValueError(
                f"JSON scrape '{source_spec.name}' returned no object rows "
                f"(response_path={source_spec.response_path!r})."
            )

        df = pd.DataFrame(records)
        if df.empty:
            raise ValueError(f"JSON scrape '{source_spec.name}' returned an empty result.")

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

    def fetch_page(self, url: str) -> str:
        """Fetch one page and return raw HTML."""

        try:
            html_session_cls, _ = self._import_requests_html()
            session = html_session_cls()
            response = session.get(url, timeout=self.timeout)
            if response.status_code >= 400:
                raise ValueError(
                    f"Failed to load page '{url}' with status code {response.status_code}."
                )
            return response.text
        except ImportError:
            requests = self._import_requests()
            try:
                response = requests.get(url, timeout=self.timeout)
            except Exception as exc:
                raise ValueError(f"Failed to load page '{url}'.") from exc
            if response.status_code >= 400:
                raise ValueError(
                    f"Failed to load page '{url}' with status code {response.status_code}."
                )
            return response.text
        except Exception as exc:
            raise ValueError(f"Failed to load page '{url}'.") from exc

    def extract_items(
        self,
        html: str,
        base_url: str,
        source_spec: SourceSpec,
    ) -> list[dict[str, Any]]:
        """Extract tabular items from HTML."""

        try:
            return self._extract_with_requests_html(html, base_url, source_spec)
        except ImportError:
            return self._extract_with_bs4(html, base_url, source_spec)
        except ValueError:
            return self._extract_with_bs4(html, base_url, source_spec)

    def extract_next_page_url(
        self,
        html: str,
        base_url: str,
        source_spec: SourceSpec,
    ) -> str | None:
        """Extract the next pagination URL."""

        next_page_selector = source_spec.pagination.get("next_page_selector")
        if not next_page_selector:
            return None

        try:
            return self._extract_next_page_url_with_requests_html(
                html,
                base_url,
                next_page_selector,
            )
        except ImportError:
            return self._extract_next_page_url_with_bs4(
                html,
                base_url,
                next_page_selector,
            )
        except ValueError:
            return self._extract_next_page_url_with_bs4(
                html,
                base_url,
                next_page_selector,
            )

    def _extract_with_requests_html(
        self,
        html: str,
        base_url: str,
        source_spec: SourceSpec,
    ) -> list[dict[str, Any]]:
        """Extract items using requests-html."""

        _, html_cls = self._import_requests_html()
        document = html_cls(html=html, url=base_url)
        elements = document.find(source_spec.selector or "")
        if not elements:
            raise ValueError(
                f"Selector '{source_spec.selector}' was not found in scraped page."
            )

        items: list[dict[str, Any]] = []
        for element in elements:
            row: dict[str, Any] = {
                "text": self._clean_text(element.text),
                "href": self._resolve_href(base_url, element.attrs.get("href")),
                "source": base_url,
            }
            row.update(self._extract_attrs(element.attrs, source_spec.attributes_to_extract))

            link_url = self._extract_item_link_from_requests_html(
                element,
                base_url,
                source_spec.item_link_selector,
            )
            if link_url:
                row["item_url"] = link_url
                if source_spec.follow_links:
                    row["follow_url"] = link_url
                    # TODO: Add Playwright/detail-page fallback for dynamic content.

            items.append(row)
        return items

    def _extract_with_bs4(
        self,
        html: str,
        base_url: str,
        source_spec: SourceSpec,
    ) -> list[dict[str, Any]]:
        """Extract items using BeautifulSoup fallback."""

        soup = self._build_bs4_soup(html)
        elements = soup.select(source_spec.selector or "")
        if not elements:
            raise ValueError(
                f"Selector '{source_spec.selector}' was not found in scraped page."
            )

        items: list[dict[str, Any]] = []
        for element in elements:
            row: dict[str, Any] = {
                "text": self._clean_text(element.get_text(" ", strip=True)),
                "href": self._resolve_href(base_url, element.get("href")),
                "source": base_url,
            }
            row.update(self._extract_attrs(element.attrs, source_spec.attributes_to_extract))

            link_url = self._extract_item_link_from_bs4(
                element,
                base_url,
                source_spec.item_link_selector,
            )
            if link_url:
                row["item_url"] = link_url
                if source_spec.follow_links:
                    row["follow_url"] = link_url
                    # TODO: Add Playwright/detail-page fallback for dynamic content.

            items.append(row)
        return items

    def _extract_next_page_url_with_requests_html(
        self,
        html: str,
        base_url: str,
        next_page_selector: str,
    ) -> str | None:
        """Extract next page link using requests-html."""

        _, html_cls = self._import_requests_html()
        document = html_cls(html=html, url=base_url)
        element = document.find(next_page_selector, first=True)
        if element is None:
            return None
        return self._resolve_href(base_url, element.attrs.get("href"))

    def _extract_next_page_url_with_bs4(
        self,
        html: str,
        base_url: str,
        next_page_selector: str,
    ) -> str | None:
        """Extract next page link using BeautifulSoup."""

        soup = self._build_bs4_soup(html)
        element = soup.select_one(next_page_selector)
        if element is None:
            return None
        return self._resolve_href(base_url, element.get("href"))

    @staticmethod
    def _extract_attrs(
        attrs: dict[str, Any],
        attributes_to_extract: list[str],
    ) -> dict[str, Any]:
        """Extract additional element attributes."""

        extracted: dict[str, Any] = {}
        for attr_name in attributes_to_extract:
            if attr_name in attrs:
                extracted[attr_name] = attrs.get(attr_name)
        return extracted

    @staticmethod
    def _extract_item_link_from_requests_html(
        element: Any,
        base_url: str,
        item_link_selector: str | None,
    ) -> str | None:
        """Extract nested item link with requests-html."""

        if not item_link_selector:
            return None
        link_element = element.find(item_link_selector, first=True)
        if link_element is None:
            return None
        return ScrapeConnector._resolve_href(base_url, link_element.attrs.get("href"))

    @staticmethod
    def _extract_item_link_from_bs4(
        element: Any,
        base_url: str,
        item_link_selector: str | None,
    ) -> str | None:
        """Extract nested item link with BeautifulSoup."""

        if not item_link_selector:
            return None
        link_element = element.select_one(item_link_selector)
        if link_element is None:
            return None
        return ScrapeConnector._resolve_href(base_url, link_element.get("href"))

    @staticmethod
    def _resolve_href(base_url: str, href: str | None) -> str | None:
        """Resolve a relative href against the page URL."""

        if not href:
            return None
        return urljoin(base_url, href)

    @staticmethod
    def _clean_text(value: str | None) -> str | None:
        """Normalize extracted text."""

        if value is None:
            return None
        cleaned = " ".join(value.split())
        return cleaned or None

    @staticmethod
    def _build_pagination_config(pagination: dict[str, Any]) -> dict[str, Any]:
        """Build pagination settings with defaults."""

        return {
            "next_page_selector": pagination.get("next_page_selector"),
            "max_pages": pagination.get("max_pages", 1),
        }

    @staticmethod
    def _import_requests_html():
        """Import requests-html lazily."""

        try:
            from requests_html import HTML, HTMLSession
        except ImportError as exc:
            raise ImportError(
                "ScrapeConnector prefers the 'requests-html' package. "
                "Install it with: pip install requests-html"
            ) from exc
        return HTMLSession, HTML

    @staticmethod
    def _import_requests():
        """Import requests lazily."""

        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "ScrapeConnector fallback requires the 'requests' package. "
                "Install it with: pip install requests"
            ) from exc
        return requests

    @staticmethod
    def _build_bs4_soup(html: str):
        """Build a BeautifulSoup parser."""

        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "ScrapeConnector fallback requires 'beautifulsoup4'. "
                "Install it with: pip install beautifulsoup4"
            ) from exc
        try:
            return BeautifulSoup(html, "lxml")
        except Exception:
            return BeautifulSoup(html, "html.parser")


# Example:
# source_spec = SourceSpec(
#     id="news-list",
#     type=SourceType.SCRAPE,
#     name="news-list",
#     url="https://example.com/news",
#     selector=".article-card",
#     attributes_to_extract=["data-id"],
#     item_link_selector="a",
#     follow_links=False,
#     field_map={"text": "text", "source_url": "item_url"},
#     pagination={"next_page_selector": ".pagination-next a", "max_pages": 3},
# )
