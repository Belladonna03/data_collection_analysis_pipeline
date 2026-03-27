"""Execute structured :class:`~agents.data_collection.scraper_spec.ScraperSpec` with requests-html.

Used by :class:`~agents.data_collection.connectors.scrape_connector.ScrapeConnector` for auto-scrape
recipes (no execution of LLM-generated code — only ``fields`` / ``item_selector`` JSON).
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal
from urllib.parse import urlparse, urljoin

from agents.data_collection.schemas import SourceSpec

EXTRACTION_MODES = ("repeated_items", "forum_first_post", "main_text")
ExtractionModeStr = Literal["repeated_items", "forum_first_post", "main_text"]


def source_spec_uses_structured_requests_html_scraper(source_spec: SourceSpec) -> bool:
    """True when ``scraper_spec`` carries an auto-scrape / requests-html field recipe."""

    spec = source_spec.scraper_spec if isinstance(source_spec.scraper_spec, dict) else {}
    raw_rt = str(spec.get("runtime") or source_spec.scraper_runtime or "requests_html").lower()
    if raw_rt.replace("-", "_") != "requests_html":
        return False
    fields = spec.get("fields")
    if not isinstance(fields, list) or len(fields) == 0:
        return False
    if not str(spec.get("item_selector") or source_spec.selector or "").strip():
        return False
    mode = (getattr(source_spec, "scrape_content_mode", None) or "html").strip().lower()
    return mode not in {"json", "json_api", "application_json"}


def infer_extraction_mode(spec_dict: dict[str, Any]) -> ExtractionModeStr | None:
    """Read ``extraction_mode`` or parse legacy ``planner_notes``."""

    raw = spec_dict.get("extraction_mode")
    if isinstance(raw, str) and raw in EXTRACTION_MODES:
        return raw  # type: ignore[return-value]
    for note in spec_dict.get("planner_notes") or []:
        if not isinstance(note, str):
            continue
        m = re.search(r"auto_probe:\s*mode=(\w+)", note)
        if m and m.group(1) in EXTRACTION_MODES:
            return m.group(1)  # type: ignore[return-value]
    return None


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(str(value).split())
    return cleaned or None


def resolve_href(base_url: str, href: str | None) -> str | None:
    if not href:
        return None
    return urljoin(base_url, href)


def field_value_from_item(item: Any, base_url: str, field: dict[str, Any]) -> Any:
    """Extract one field relative to an item node (requests-html Element)."""

    selector = str(field.get("selector") or "")
    extract = field.get("extract") or "text"
    attribute = field.get("attribute")
    node = item.find(selector, first=True) if selector else item
    if node is None:
        return None
    if extract == "text":
        return clean_text(node.text)
    if extract == "html":
        try:
            return node.html
        except Exception:
            return None
    if extract == "attr":
        if not attribute:
            return None
        raw = node.attrs.get(attribute) if hasattr(node, "attrs") else None
        if attribute == "href" and raw is not None:
            return resolve_href(base_url, str(raw))
        return raw
    return None


def extract_rows_from_document(
    doc: Any,
    *,
    item_selector: str,
    fields: list[dict[str, Any]],
    base_url: str,
    extraction_mode: ExtractionModeStr | None,
) -> list[dict[str, Any]]:
    """Pull row dicts from a parsed HTML document."""

    items = doc.find(item_selector)
    if not items:
        return []

    mode = extraction_mode
    if mode is None:
        mode = "repeated_items" if len(items) > 1 else "main_text"

    if mode in ("forum_first_post", "main_text"):
        items = [items[0]]

    rows: list[dict[str, Any]] = []
    for item in items:
        row: dict[str, Any] = {}
        for f in fields:
            name = str(f.get("name", ""))
            if not name:
                continue
            row[name] = field_value_from_item(item, base_url, f)
        rows.append(row)
    return rows


def thread_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        path = urlparse(url).path.strip("/")
        if not path:
            return None
        parts = path.split("/")
        return parts[-1][:128] or None
    except Exception:
        return None


def enrich_structured_rows(
    rows: list[dict[str, Any]],
    *,
    landing_url: str,
) -> list[dict[str, Any]]:
    """Add ``source_url``, optional ``thread_id`` / ``created_at`` / ``forum_section``, and ``metadata`` JSON."""

    preserved = {
        "title",
        "body",
        "text",
        "label",
        "link",
        "date",
        "section",
        "author_link",
        "source_url",
        "thread_id",
        "created_at",
        "forum_section",
        "tags",  # optional scrape field name
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        er = dict(row)
        link = er.get("link")
        if isinstance(link, str) and link.strip():
            src_url = link.split("#", 1)[0].strip()
        else:
            src_url = landing_url.split("#", 1)[0].strip()

        er["source_url"] = src_url
        tid = thread_id_from_url(src_url)
        if tid:
            er["thread_id"] = tid

        if er.get("date") is not None:
            er["created_at"] = er["date"]
        if er.get("section") is not None:
            er["forum_section"] = er["section"]

        meta_obj: dict[str, Any] = {}
        if er.get("link") is not None:
            meta_obj["item_href"] = er["link"]
        if er.get("author_link") is not None:
            meta_obj["author_link"] = er["author_link"]
        for k, v in er.items():
            if k in preserved or k == "metadata":
                continue
            if v is not None and str(v).strip() != "":
                meta_obj[k] = v

        er["metadata"] = (
            json.dumps(meta_obj, sort_keys=True, ensure_ascii=False, default=str) if meta_obj else "{}"
        )
        out.append(er)
    return out


def debug_snippet_for_spec(summary: dict[str, Any]) -> str:
    """Non-executable debug text (spec summary only)."""

    lines = [
        "# Debug: structured scraper_spec summary (not executed as code).",
        f"# extraction_mode={summary.get('extraction_mode')!r} item_selector={summary.get('item_selector')!r}",
        f"# fields={[f.get('name') for f in summary.get('fields') or []]}",
    ]
    return "\n".join(lines) + "\n"
