"""Heuristic HTML probing with requests-html: builds JSON ``scraper_spec`` (no exec of LLM code).

Public API:
- :func:`probe_html` — analyze HTML string + base URL (tests, offline).
- :func:`probe_url` — fetch with ``HTMLSession``, optional ``render`` retry.
- :func:`apply_probe_to_source_candidate` — promote a :class:`~agents.data_collection.schemas.SourceCandidate` when probe succeeds.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal

from agents.data_collection.schemas import SourceCandidate
from agents.data_collection.scraper_spec import ScraperField, ScraperSpec

ExtractionMode = Literal["repeated_items", "forum_first_post", "main_text"]

FORUM_LIKE = re.compile(
    r"post|thread|topic|discussion|question|article|message|reply|forum|comment",
    re.IGNORECASE,
)

# Candidate item selectors (CSS). Order: more specific-ish first.
_ITEM_SELECTOR_TRIES: tuple[str, ...] = (
    "li.topic-post",
    "li.discussion-item",
    "article.post",
    "[class*='topic-post']",
    "[class*='forum-post']",
    "[class*='thread-']",
    "[class*='discussion-']",
    "article",
    "[class*='post']",
    "[class*='comment']",
    "[class*='card']",
)


def _clean_text(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = " ".join(str(value).split())
    return cleaned or None


def _element_text_len(el: Any) -> int:
    try:
        return len((el.text or "").strip())
    except Exception:
        return 0


def _forum_kw_bonus(elements: list[Any], sample: int = 4) -> float:
    bonus = 0.0
    for el in elements[:sample]:
        try:
            html_snip = (el.html or "")[:2000]
        except Exception:
            html_snip = ""
        if FORUM_LIKE.search(html_snip):
            bonus += 5.0
    return min(bonus, 15.0)


def _prune_subsumed_elements(elements: list[Any]) -> list[Any]:
    """Drop small fragments that are clearly nested inside another matched node (reduces false repeated clusters)."""

    if len(elements) <= 1:
        return elements
    pairs: list[tuple[Any, str]] = []
    for e in elements:
        try:
            pairs.append((e, (e.html or "").strip()))
        except Exception:
            pairs.append((e, ""))
    out: list[Any] = []
    for i, (e, h) in enumerate(pairs):
        if len(h) < 30:
            continue
        subsumed = False
        for j, (_, h2) in enumerate(pairs):
            if i == j or len(h2) < len(h):
                continue
            if h and h2 and h in h2 and h != h2:
                subsumed = True
                break
        if not subsumed:
            out.append(e)
    return out if len(out) >= 2 else []


def _score_repeated_group(elements: list[Any]) -> float:
    n = len(elements)
    if n < 2:
        return 0.0
    lengths = [_element_text_len(e) for e in elements]
    avg_len = sum(lengths) / n
    if avg_len < 20.0:
        return 0.0
    link_hits = 0
    for e in elements:
        try:
            if e.find("a", first=True) is not None:
                link_hits += 1
        except Exception:
            pass
    link_ratio = link_hits / float(n)
    base = n * math.log1p(avg_len) + 20.0 * link_ratio
    return base + _forum_kw_bonus(elements)


def _preview_row_from_item(item: Any, base_url: str) -> dict[str, Any]:
    row: dict[str, Any] = {}
    try:
        title_el = item.find("h1, h2, h3, .title, .post-title", first=True)
        row["title"] = _clean_text(title_el.text if title_el else None)
    except Exception:
        row["title"] = None
    try:
        body_el = item.find(".body, .post-body, .preview, .content", first=True)
        if body_el is not None:
            row["body"] = _clean_text(body_el.text)
        else:
            ps = item.find("p")
            if ps:
                row["body"] = _clean_text(" ".join((p.text or "") for p in ps[:5]))
            else:
                row["body"] = _clean_text(item.text)
    except Exception:
        row["body"] = _clean_text(getattr(item, "text", None))
    try:
        a = item.find("a", first=True)
        if a is not None:
            raw_href = a.attrs.get("href")
            if raw_href:
                row["link"] = str(raw_href)
        auth = item.find("a.author-link, footer a", first=True)
        if auth is not None and auth.attrs.get("href"):
            row["author_link"] = str(auth.attrs.get("href"))
    except Exception:
        pass
    try:
        time_el = item.find("time", first=True)
        if time_el is not None:
            row["date"] = time_el.attrs.get("datetime") or _clean_text(time_el.text)
    except Exception:
        pass
    try:
        tag_el = item.find(".tag, .section-label, [data-section]", first=True)
        if tag_el is not None:
            row["section"] = tag_el.attrs.get("data-section") or _clean_text(tag_el.text)
    except Exception:
        pass
    return row


def _try_repeated_items(doc: Any) -> tuple[ExtractionMode, str, list[Any], float] | None:
    best_score = 0.0
    best_sel = ""
    best_els: list[Any] = []

    for sel in _ITEM_SELECTOR_TRIES:
        try:
            raw = doc.find(sel)
        except Exception:
            continue
        if len(raw) > 200:
            continue
        if len(raw) < 2:
            continue
        good = [e for e in raw if _element_text_len(e) >= 25]
        good = _prune_subsumed_elements(good)
        if len(good) < 2:
            continue
        sc = _score_repeated_group(good)
        if sc > best_score:
            best_score = sc
            best_sel = sel
            best_els = good

    if best_score < 12.0 or not best_sel:
        return None
    return "repeated_items", best_sel, best_els, best_score


def _try_forum_first_post(doc: Any) -> tuple[ExtractionMode, str, list[Any], float] | None:
    for sel in ("article.first-post", "article.post", "article", "[class*='first-post']"):
        try:
            els = doc.find(sel)
        except Exception:
            continue
        if not els:
            continue
        first = els[0]
        tl = _element_text_len(first)
        if tl < 80:
            continue
        if len(els) > 1:
            item_selector = f"{sel}:first-of-type"
        else:
            item_selector = sel
        score = 18.0 + math.log1p(tl) + _forum_kw_bonus([first])
        return "forum_first_post", item_selector, [first], score
    return None


def _try_main_text(doc: Any) -> tuple[ExtractionMode, str, list[Any], float] | None:
    for sel in ("main", "#content", ".content", "article", "body"):
        try:
            el = doc.find(sel, first=True)
        except Exception:
            el = None
        if el is None:
            continue
        t = (el.text or "").strip()
        if len(t) < 350:
            continue
        score = 10.0 + math.log1p(len(t))
        return "main_text", sel, [el], score
    return None


def _run_probe_on_document(doc: Any, url: str, *, used_render: bool) -> "AutoScrapeResult":
    metrics: dict[str, Any] = {"used_render": used_render}

    repeated = _try_repeated_items(doc)
    if repeated is not None:
        mode, sel, els, score = repeated
        metrics.update(mode=mode, item_selector=sel, score=score, item_count=len(els))
        previews = [_preview_row_from_item(e, url) for e in els[:8]]
        confidence = min(1.0, score / 45.0)
        spec = ScraperSpec(
            entry_url=url,
            item_selector=sel,
            fields=[
                ScraperField("title", "h1, h2, h3, .title, .post-title", "text"),
                ScraperField("body", ".body, .post-body, .preview, .content, p", "text"),
                ScraperField("link", "a", "attr", attribute="href"),
                ScraperField("date", "time", "attr", attribute="datetime"),
                ScraperField("section", ".tag, .section-label, [data-section]", "attr", attribute="data-section"),
            ],
            requires_js=used_render,
            runtime="requests_html",
            confidence=confidence,
            planner_notes=[f"auto_probe: mode={mode}", f"score={score:.2f}"],
            extraction_mode=mode,
        )
        return AutoScrapeResult(
            success=True,
            url=url,
            mode=mode,
            scraper_spec=spec.to_dict(),
            preview_rows=previews,
            reason=None,
            used_render=used_render,
            metrics=metrics,
        )

    forum = _try_forum_first_post(doc)
    if forum is not None:
        mode, sel, els, score = forum
        metrics.update(mode=mode, item_selector=sel, score=score, item_count=len(els))
        previews = [_preview_row_from_item(els[0], url)]
        confidence = min(1.0, score / 45.0)
        spec = ScraperSpec(
            entry_url=url,
            item_selector=sel,
            fields=[
                ScraperField("title", "h1, h2, h3, .title, .post-title", "text"),
                ScraperField("body", ".body, .post-body, p", "text"),
                ScraperField("link", "a", "attr", attribute="href"),
                ScraperField("date", "time", "attr", attribute="datetime"),
                ScraperField("section", ".section-label, [data-section]", "attr", attribute="data-section"),
            ],
            requires_js=used_render,
            runtime="requests_html",
            confidence=confidence,
            planner_notes=[f"auto_probe: mode={mode}", f"score={score:.2f}"],
            extraction_mode=mode,
        )
        return AutoScrapeResult(
            success=True,
            url=url,
            mode=mode,
            scraper_spec=spec.to_dict(),
            preview_rows=previews,
            reason=None,
            used_render=used_render,
            metrics=metrics,
        )

    main = _try_main_text(doc)
    if main is not None:
        mode, sel, els, score = main
        metrics.update(mode=mode, item_selector=sel, score=score)
        el = els[0]
        previews = [
            {
                "title": _clean_text(
                    el.find("h1, h2", first=True).text if el.find("h1, h2", first=True) else None
                ),
                "body": _clean_text(el.text),
            }
        ]
        confidence = min(1.0, score / 40.0)
        spec = ScraperSpec(
            entry_url=url,
            item_selector=sel,
            fields=[
                ScraperField("title", "h1, h2", "text"),
                ScraperField("body", "", "text"),
            ],
            requires_js=used_render,
            runtime="requests_html",
            confidence=confidence,
            planner_notes=[f"auto_probe: mode={mode}", f"score={score:.2f}"],
            extraction_mode=mode,
        )
        return AutoScrapeResult(
            success=True,
            url=url,
            mode=mode,
            scraper_spec=spec.to_dict(),
            preview_rows=previews,
            reason=None,
            used_render=used_render,
            metrics=metrics,
        )

    return AutoScrapeResult(
        success=False,
        url=url,
        mode=None,
        scraper_spec=None,
        preview_rows=[],
        reason="No repeatable items, substantive first post, or large main region found.",
        used_render=used_render,
        metrics=metrics,
    )


@dataclass
class AutoScrapeResult:
    success: bool
    url: str
    mode: ExtractionMode | None
    scraper_spec: dict[str, Any] | None
    preview_rows: list[dict[str, Any]]
    reason: str | None
    used_render: bool
    metrics: dict[str, Any]


def probe_html(html: str, url: str, *, used_render: bool = False) -> AutoScrapeResult:
    """Parse HTML with requests-html and run heuristics (offline-friendly for tests)."""

    try:
        from requests_html import HTML
    except ImportError as exc:
        return AutoScrapeResult(
            success=False,
            url=url,
            mode=None,
            scraper_spec=None,
            preview_rows=[],
            reason=f"requests-html is not installed: {exc}",
            used_render=used_render,
            metrics={},
        )

    doc = HTML(html=html, url=url)
    return _run_probe_on_document(doc, url, used_render=used_render)


def probe_url(
    url: str,
    *,
    timeout: float = 25.0,
    render_fallback: bool = True,
) -> AutoScrapeResult:
    """Fetch *url* with ``HTMLSession``; optional ``render`` only if the first pass fails."""

    try:
        from requests_html import HTMLSession
    except ImportError as exc:
        return AutoScrapeResult(
            success=False,
            url=url,
            mode=None,
            scraper_spec=None,
            preview_rows=[],
            reason=f"requests-html is not installed: {exc}",
            used_render=False,
            metrics={},
        )

    session = HTMLSession()
    try:
        response = session.get(url, timeout=timeout)
    except Exception as exc:
        return AutoScrapeResult(
            success=False,
            url=url,
            mode=None,
            scraper_spec=None,
            preview_rows=[],
            reason=f"fetch failed: {exc!r}",
            used_render=False,
            metrics={},
        )

    if response.status_code >= 400:
        return AutoScrapeResult(
            success=False,
            url=url,
            mode=None,
            scraper_spec=None,
            preview_rows=[],
            reason=f"HTTP {response.status_code}",
            used_render=False,
            metrics={"status_code": response.status_code},
        )

    first = _run_probe_on_document(response.html, url, used_render=False)
    if first.success or not render_fallback:
        return first

    # Fallback: try JS render when the static HTML looks too thin for body heuristics.
    skinny = len((response.html.text or "").strip()) < 400
    if not skinny:
        return first

    try:
        response.html.render(timeout=int(min(timeout, 45)), sleep=0.5)
    except Exception as exc:
        merged_metrics = {**first.metrics, "render_error": str(exc)}
        return AutoScrapeResult(
            success=False,
            url=url,
            mode=None,
            scraper_spec=None,
            preview_rows=[],
            reason=f"probe failed; render fallback failed: {exc!r}",
            used_render=True,
            metrics=merged_metrics,
        )

    second = _run_probe_on_document(response.html, url, used_render=True)
    if not second.success and first.reason:
        return AutoScrapeResult(
            success=False,
            url=url,
            mode=None,
            scraper_spec=None,
            preview_rows=[],
            reason=first.reason,
            used_render=True,
            metrics={**second.metrics, "render_attempted": True},
        )
    return second


def apply_probe_to_source_candidate(candidate: SourceCandidate, result: AutoScrapeResult) -> None:
    """If *result* succeeded, copy ``scraper_spec`` / ``selector`` onto *candidate* and mark executable."""

    if not result.success or not result.scraper_spec:
        return
    spec = result.scraper_spec
    candidate.selector = spec.get("item_selector") or candidate.selector
    merged = dict(candidate.scraper_spec or {})
    merged.update(spec)
    candidate.scraper_spec = merged
    candidate.scraper_runtime = str(spec.get("runtime") or "requests_html")
    candidate.requires_js = bool(result.used_render or spec.get("requires_js"))
    candidate.is_executable = True
    candidate.non_executable_reason = None
    candidate.execution_mode = "auto_scrape_requests_html"
    candidate.auto_scrape_success = True
    candidate.auto_scrape_reason = None
    candidate.auto_scrape_preview_count = len(result.preview_rows)


# Backward-friendly alias
apply_probe_to_candidate = apply_probe_to_source_candidate
