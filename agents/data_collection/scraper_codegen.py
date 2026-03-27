from __future__ import annotations

from typing import Callable

from agents.data_collection.scraper_spec import ScraperField, ScraperSpec
from agents.data_collection.scraper_spec_execution import infer_extraction_mode


def _emit_field_dict(f: ScraperField) -> str:
    return (
        f'{{"name": {f.name!r}, "selector": {f.selector!r}, '
        f'"extract": {f.extract!r}, "attribute": {f.attribute!r}}}'
    )


def _allowed_domains_literal(spec: ScraperSpec) -> str:
    domains = list(spec.allowed_domains)
    if not domains and spec.entry_url:
        from urllib.parse import urlparse

        host = urlparse(spec.entry_url).netloc.casefold()
        if host.startswith("www."):
            host = host[4:]
        if host:
            domains = [host]
    return repr(domains)


def generate_debug_scraper_py(
    spec: ScraperSpec,
    *,
    source_url: str | None = None,
) -> str:
    """Return a **human-readable** Python script for demos and debugging only.

    The collection **pipeline does not execute** this file. Production runs use
    :class:`~agents.data_collection.connectors.scrape_connector.ScrapeConnector`
    with the JSON ``scraper_spec`` (same field definitions as embedded below).
    """

    if not (spec.item_selector or "").strip():
        raise ValueError("ScraperSpec.item_selector is required for debug code generation.")
    if not spec.fields:
        raise ValueError("ScraperSpec.fields must be non-empty for debug code generation.")

    entry = (source_url or spec.entry_url or "").strip() or spec.entry_url
    mode = spec.extraction_mode or infer_extraction_mode(spec.to_dict()) or "repeated_items"
    mode_lit = repr(mode)

    fields_lines = []
    for f in spec.fields:
        fields_lines.append(
            f"#   - {f.name!r}: selector={f.selector!r}, extract={f.extract!r}, attribute={f.attribute!r}"
        )
    fields_comment = "\n".join(fields_lines)

    header = f'''"""
DEBUG / DEMO ONLY — NOT EXECUTED BY THE DATA COLLECTION PIPELINE
================================================================

The pipeline collects data via ScrapeConnector + structured ``scraper_spec`` JSON
(item_selector, fields, extraction_mode). This script mirrors that logic using
``requests-html`` so you can run, tweak, and inspect selectors in isolation.

Origin: auto-scrape probe (heuristic) → ``scraper_spec`` saved as ``*_scraper_spec.json``.

Source URL (entry): {entry!r}
Extraction mode: {mode!r}
  - repeated_items: one DataFrame row per matched item (list/card/post).
  - forum_first_post / main_text: only the first matched node → one row (when multiple match).

Chosen item selector: {spec.item_selector!r}

Per-field selectors (relative to each item node):
{fields_comment}

Requires JS render: {spec.requires_js!r}
Runtime reference: {spec.runtime!r} (pipeline uses JSON spec, not this file).
Confidence (probe): {spec.confidence!r}
"""


'''

    fields_literal = "[\n        " + ",\n        ".join(_emit_field_dict(f) for f in spec.fields) + "\n    ]"
    allowed_domains = _allowed_domains_literal(spec)
    requires_js = "True" if spec.requires_js else "False"

    body = (
        _DEBUG_SCRAPER_TEMPLATE.replace("<<<ENTRY_URL>>>", repr(entry))
        .replace("<<<ITEM_SELECTOR>>>", repr(spec.item_selector))
        .replace("<<<EXTRACTION_MODE>>>", mode_lit)
        .replace("<<<MAX_PAGES>>>", str(int(spec.max_pages)))
        .replace("<<<ALLOWED_DOMAINS>>>", allowed_domains)
        .replace("<<<PAGINATION_STRATEGY>>>", repr(spec.pagination_strategy))
        .replace("<<<PAGINATION_SELECTOR>>>", repr(spec.pagination_selector))
        .replace("<<<FIELDS_LITERAL>>>", fields_literal)
        .replace("<<<REQUIRES_JS>>>", requires_js)
    )
    return header + body


def generate_scraper_code(spec: ScraperSpec) -> str:
    """Return readable Python source for a ``requests_html`` + ``pandas`` scraper.

    Contract: define ``run(timeout=20.0) -> pandas.DataFrame``.
    """

    if not (spec.item_selector or "").strip():
        raise ValueError("ScraperSpec.item_selector is required for code generation.")
    if not spec.fields:
        raise ValueError("ScraperSpec.fields must be non-empty for code generation.")

    fields_literal = "[\n        " + ",\n        ".join(_emit_field_dict(f) for f in spec.fields) + "\n    ]"
    allowed_domains = _allowed_domains_literal(spec)
    requires_js = "True" if spec.requires_js else "False"

    header = (
        "# Auto-generated scraper from ScraperSpec (MVP). Review before running.\n"
        f"# confidence={spec.confidence!r} runtime={spec.runtime!r}\n"
    )

    body = (
        _SCRAPER_TEMPLATE.replace("<<<ENTRY_URL>>>", repr(spec.entry_url))
        .replace("<<<ITEM_SELECTOR>>>", repr(spec.item_selector))
        .replace("<<<MAX_PAGES>>>", str(int(spec.max_pages)))
        .replace("<<<ALLOWED_DOMAINS>>>", allowed_domains)
        .replace("<<<PAGINATION_STRATEGY>>>", repr(spec.pagination_strategy))
        .replace("<<<PAGINATION_SELECTOR>>>", repr(spec.pagination_selector))
        .replace("<<<FIELDS_LITERAL>>>", fields_literal)
        .replace("<<<REQUIRES_JS>>>", requires_js)
    )
    return header + body


_SCRAPER_TEMPLATE = """
from urllib.parse import urljoin, urlparse

import pandas as pd
from requests_html import HTMLSession


def _clean_text(value):
    if value is None:
        return None
    cleaned = " ".join(str(value).split())
    return cleaned or None


def _resolve_href(base_url, href):
    if not href:
        return None
    return urljoin(base_url, href)


def _host_allowed(url, allowed_hosts):
    if not allowed_hosts:
        return True
    host = urlparse(url).netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    allowed = {h.casefold() for h in allowed_hosts}
    return host in allowed


def _field_value(item, base_url, field):
    selector = field["selector"] or ""
    extract = field["extract"]
    attribute = field["attribute"]
    node = item.find(selector, first=True) if selector else item
    if node is None:
        return None
    if extract == "text":
        return _clean_text(node.text)
    if extract == "html":
        return node.html
    if extract == "attr":
        if not attribute:
            return None
        raw = node.attrs.get(attribute)
        if attribute == "href" and raw is not None:
            return _resolve_href(base_url, raw)
        return raw
    return None


def run(timeout=20.0):
    '''Scrape according to the embedded spec; return a DataFrame.'''

    entry_url = <<<ENTRY_URL>>>
    item_selector = <<<ITEM_SELECTOR>>>
    max_pages = <<<MAX_PAGES>>>
    allowed_domains = <<<ALLOWED_DOMAINS>>>
    pagination_strategy = <<<PAGINATION_STRATEGY>>>
    pagination_selector = <<<PAGINATION_SELECTOR>>>
    fields = <<<FIELDS_LITERAL>>>
    requires_js = <<<REQUIRES_JS>>>

    session = HTMLSession()
    rows = []
    current_url = entry_url

    for _page_idx in range(max_pages):
        if not _host_allowed(current_url, allowed_domains):
            break
        response = session.get(current_url, timeout=timeout)
        if response.status_code >= 400:
            raise ValueError(
                "HTTP {} for {!r}".format(response.status_code, current_url)
            )
        if requires_js:
            response.html.render(timeout=int(timeout), sleep=0.5)
        doc = response.html
        items = doc.find(item_selector)
        if not items:
            break

        for item in items:
            row = {}
            for field in fields:
                row[field["name"]] = _field_value(item, current_url, field)
            rows.append(row)

        if pagination_strategy != "link_next" or not pagination_selector:
            break
        nxt_el = doc.find(pagination_selector, first=True)
        if nxt_el is None:
            break
        href = nxt_el.attrs.get("href")
        next_url = _resolve_href(current_url, href)
        if not next_url or next_url == current_url:
            break
        if not _host_allowed(next_url, allowed_domains):
            break
        current_url = next_url

    return pd.DataFrame(rows)
""".lstrip()


_DEBUG_SCRAPER_TEMPLATE = """
# -----------------------------------------------------------------------------
# Implementation sketch (requests-html). Safe to run locally; not used by pipeline.
# -----------------------------------------------------------------------------
from urllib.parse import urljoin, urlparse

import pandas as pd
from requests_html import HTMLSession


def _clean_text(value):
    if value is None:
        return None
    cleaned = " ".join(str(value).split())
    return cleaned or None


def _resolve_href(base_url, href):
    if not href:
        return None
    return urljoin(base_url, href)


def _host_allowed(url, allowed_hosts):
    if not allowed_hosts:
        return True
    host = urlparse(url).netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    allowed = {h.casefold() for h in allowed_hosts}
    return host in allowed


def _field_value(item, base_url, field):
    selector = field["selector"] or ""
    extract = field["extract"]
    attribute = field["attribute"]
    node = item.find(selector, first=True) if selector else item
    if node is None:
        return None
    if extract == "text":
        return _clean_text(node.text)
    if extract == "html":
        return node.html
    if extract == "attr":
        if not attribute:
            return None
        raw = node.attrs.get(attribute)
        if attribute == "href" and raw is not None:
            return _resolve_href(base_url, raw)
        return raw
    return None


def run(timeout=20.0):
    '''Fetch the page, match items, extract fields — same shape as pipeline internals.'''

    # --- Values copied from scraper_spec (see docstring above) ---
    entry_url = <<<ENTRY_URL>>>
    item_selector = <<<ITEM_SELECTOR>>>
    extraction_mode = <<<EXTRACTION_MODE>>>
    max_pages = <<<MAX_PAGES>>>
    allowed_domains = <<<ALLOWED_DOMAINS>>>
    pagination_strategy = <<<PAGINATION_STRATEGY>>>
    pagination_selector = <<<PAGINATION_SELECTOR>>>
    fields = <<<FIELDS_LITERAL>>>
    requires_js = <<<REQUIRES_JS>>>

    session = HTMLSession()
    rows = []
    current_url = entry_url

    for _page_idx in range(max_pages):
        if not _host_allowed(current_url, allowed_domains):
            break
        # Load page (same family of calls as ScrapeConnector)
        response = session.get(current_url, timeout=timeout)
        if response.status_code >= 400:
            raise ValueError(
                "HTTP {} for {!r}".format(response.status_code, current_url)
            )
        if requires_js:
            response.html.render(timeout=int(timeout), sleep=0.5)
        doc = response.html
        items = doc.find(item_selector)
        # Match pipeline: forum_first_post / main_text → first item only
        if extraction_mode in ("forum_first_post", "main_text") and items:
            items = items[:1]
        if not items:
            break

        for item in items:
            row = {}
            for field in fields:
                row[field["name"]] = _field_value(item, current_url, field)
            rows.append(row)

        if pagination_strategy != "link_next" or not pagination_selector:
            break
        nxt_el = doc.find(pagination_selector, first=True)
        if nxt_el is None:
            break
        href = nxt_el.attrs.get("href")
        next_url = _resolve_href(current_url, href)
        if not next_url or next_url == current_url:
            break
        if not _host_allowed(next_url, allowed_domains):
            break
        current_url = next_url

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run()
    print(df.head())
""".lstrip()


def save_generated_scraper_artifact(
    spec: ScraperSpec,
    save_text: Callable[..., str],
    artifact_name: str,
) -> tuple[str, str]:
    """Generate code and persist via ``save_text(name, content, *, suffix=...)``.

    Returns ``(code, path)``. Compatible with :meth:`_InMemoryArtifactStorage.save_text`.
    """

    code = generate_scraper_code(spec)
    path = save_text(artifact_name, code, suffix=".py")
    return code, path
