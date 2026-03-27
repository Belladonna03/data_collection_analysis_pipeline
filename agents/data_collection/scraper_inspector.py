from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests


@dataclass
class SelectorCandidate:
    """One CSS selector hypothesis with a rough repeat count on the page."""

    selector: str
    count: int
    source: str = ""


@dataclass
class PageInspection:
    """Structured snapshot after parsing HTML (no execution)."""

    base_url: str
    title: str | None
    raw_html_length: int
    layout_hints: list[str] = field(default_factory=list)
    candidate_item_selectors: list[SelectorCandidate] = field(default_factory=list)
    candidate_pagination_selectors: list[SelectorCandidate] = field(default_factory=list)
    sample_item_link_hrefs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def fetch_html(url: str, *, timeout: float = 20.0, headers: dict[str, str] | None = None) -> str:
    """Download page HTML with plain ``requests`` (no JS rendering)."""

    response = requests.get(url, timeout=timeout, headers=headers or {})
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding or "utf-8"
    return response.text


def inspect_url(url: str, *, timeout: float = 20.0, headers: dict[str, str] | None = None) -> PageInspection:
    """Fetch and inspect a URL."""

    html = fetch_html(url, timeout=timeout, headers=headers)
    return inspect_html(html, base_url=url)


def inspect_html(html: str, *, base_url: str = "") -> PageInspection:
    """Parse HTML and collect heuristic signals for scraper planning."""

    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "scraper_inspector requires beautifulsoup4. Install with: pip install beautifulsoup4"
        ) from exc

    soup = BeautifulSoup(html, "lxml")
    hints: list[str] = []
    notes: list[str] = []
    item_candidates: list[SelectorCandidate] = []
    pagination_candidates: list[SelectorCandidate] = []
    links_sample: list[str] = []

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        title = og["content"].strip() or title

    # --- Tables
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) >= 3:
            if table.find("tbody"):
                sel = "table tbody tr"
                count = len(table.find("tbody").find_all("tr"))
            else:
                sel = "table tr"
                count = len(rows)
            item_candidates.append(SelectorCandidate(selector=sel, count=count, source="table"))
            hints.append("table")
            break

    # --- Repeating class blocks (div/article/li)
    tag_names = ("div", "article", "li", "section")
    class_counts: Counter[tuple[str, str]] = Counter()
    for tag in soup.find_all(tag_names):
        cls = tag.get("class")
        if not cls or not isinstance(cls, list):
            continue
        for c in cls:
            cnorm = str(c).strip()
            if not cnorm or len(cnorm) > 80:
                continue
            class_counts[(tag.name, cnorm)] += 1

    for (tag_name, class_name), cnt in class_counts.most_common(40):
        if cnt < 3:
            break
        sel = f"{tag_name}.{_css_class_token(class_name)}"
        item_candidates.append(SelectorCandidate(selector=sel, count=cnt, source="repeated_class"))

    if any(h == "table" for h in hints) and any(c.source == "repeated_class" for c in item_candidates):
        hints.append("mixed_table_and_cards")

    # --- Pagination heuristics
    if soup.select('a[rel="next"]'):
        pagination_candidates.append(
            SelectorCandidate(selector='a[rel="next"]', count=len(soup.select('a[rel="next"]')), source="rel_next")
        )
    for a in soup.find_all("a", href=True):
        t = a.get_text(strip=True).casefold()
        if t in {"next", "older", "more", "→", "»"}:
            if a.get("class"):
                cls = a["class"][0]
                pagination_candidates.append(
                    SelectorCandidate(
                        selector=f"a.{_css_class_token(cls)}",
                        count=1,
                        source="link_text",
                    )
                )
            break
    nav_next = soup.select(".pagination a, nav.pagination a, ul.pagination a")
    if nav_next:
        pagination_candidates.append(
            SelectorCandidate(
                selector=".pagination a",
                count=len(nav_next),
                source="pagination_nav",
            )
        )

    # --- Sample in-content links (skip obvious global nav)
    main = soup.find("main") or soup.find("article") or soup.body
    if main:
        from urllib.parse import urljoin as _urljoin

        for a in main.find_all("a", href=True, limit=25):
            href = a["href"].strip()
            if href.startswith("#") or href.casefold().startswith("javascript:"):
                continue
            if base_url:
                href = _urljoin(base_url, href)
            links_sample.append(href)
            if len(links_sample) >= 8:
                break

    if not item_candidates:
        hints.append("no_clear_repeating_items")
        notes.append("No repeating blocks met the minimum count threshold.")

    deduped_items = _dedupe_selector_candidates(item_candidates)
    deduped_items.sort(key=lambda c: (-c.count, len(c.selector)))

    # --- Ambiguity: top two *deduped* item candidates with similar counts
    if len(deduped_items) >= 2:
        top, second = deduped_items[0], deduped_items[1]
        if second.count >= max(3, int(top.count * 0.75)):
            hints.append("ambiguous_item_selectors")

    return PageInspection(
        base_url=base_url,
        title=title,
        raw_html_length=len(html),
        layout_hints=hints,
        candidate_item_selectors=deduped_items[:12],
        candidate_pagination_selectors=_dedupe_selector_candidates(pagination_candidates)[:8],
        sample_item_link_hrefs=links_sample,
        notes=notes,
    )


def _css_class_token(class_name: str) -> str:
    """Escape a single class token for a simple CSS class selector."""

    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", class_name)
    return safe or "x"


def _dedupe_selector_candidates(candidates: list[SelectorCandidate]) -> list[SelectorCandidate]:
    seen: set[str] = set()
    out: list[SelectorCandidate] = []
    for c in candidates:
        if c.selector in seen:
            continue
        seen.add(c.selector)
        out.append(c)
    return out


def allowed_domains_for_url(url: str) -> list[str]:
    """Single-host allowlist from entry URL netloc."""

    parsed = urlparse(url)
    host = parsed.netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    return [host] if host else []
