from __future__ import annotations

from agents.data_collection.scraper_inspector import PageInspection, SelectorCandidate, allowed_domains_for_url
from agents.data_collection.scraper_spec import ScraperField, ScraperSpec

DEFAULT_MAX_PAGES = 3


def _default_fields(chosen: SelectorCandidate) -> list[ScraperField]:
    """Minimal field set compatible with a list-extraction connector mindset."""

    if chosen.source == "table":
        return [ScraperField(name="row_text", selector="", extract="text")]

    return [
        ScraperField(name="text", selector="", extract="text"),
        ScraperField(name="link", selector="a", extract="attr", attribute="href"),
    ]


def _pick_pagination(inspection: PageInspection) -> tuple[str, str | None]:
    """Return (strategy, selector)."""

    for candidate in inspection.candidate_pagination_selectors:
        if candidate.selector:
            return "link_next", candidate.selector
    return "none", None


def plan_from_inspection(
    inspection: PageInspection,
    *,
    entry_url: str,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> ScraperSpec:
    """Build a single :class:`ScraperSpec` from a :class:`PageInspection`."""

    notes = list(inspection.notes)
    domains = allowed_domains_for_url(entry_url)

    if not inspection.candidate_item_selectors:
        return ScraperSpec(
            entry_url=entry_url,
            item_selector="",
            fields=[],
            pagination_strategy="none",
            pagination_selector=None,
            max_pages=max_pages,
            allowed_domains=domains,
            requires_js=False,
            runtime="requests_html",
            confidence=0.0,
            planner_notes=notes + ["No repeating item selector could be inferred."],
        )

    has_table_hint = "table" in inspection.layout_hints
    ambiguous = "ambiguous_item_selectors" in inspection.layout_hints
    chosen: SelectorCandidate | None = None
    if has_table_hint and not ambiguous:
        for candidate in inspection.candidate_item_selectors:
            if candidate.source == "table":
                chosen = candidate
                break
    if chosen is None:
        chosen = inspection.candidate_item_selectors[0]

    pag_strategy, pag_selector = _pick_pagination(inspection)

    if ambiguous:
        notes.append("Multiple repeating patterns had similar counts; using the highest-count selector.")
        confidence = 0.38
    elif chosen.source == "table":
        confidence = 0.74
    else:
        confidence = 0.58

    return ScraperSpec(
        entry_url=entry_url,
        item_selector=chosen.selector,
        fields=_default_fields(chosen),
        pagination_strategy=pag_strategy,
        pagination_selector=pag_selector,
        max_pages=max_pages,
        allowed_domains=domains,
        requires_js=False,
        runtime="requests_html",
        confidence=confidence,
        planner_notes=notes,
    )


def propose_scraper_spec(
    entry_url: str,
    *,
    html: str | None = None,
    timeout: float = 20.0,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> ScraperSpec:
    """Fetch (optional) HTML, inspect, and return a planned :class:`ScraperSpec`."""

    from agents.data_collection.scraper_inspector import inspect_html, inspect_url

    if html is None:
        inspection = inspect_url(entry_url, timeout=timeout)
    else:
        inspection = inspect_html(html, base_url=entry_url)
    return plan_from_inspection(inspection, entry_url=entry_url, max_pages=max_pages)


# Backward-compatible alias
propose_scraper_plan = propose_scraper_spec
