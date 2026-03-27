from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ExtractKind = Literal["text", "attr", "html"]


@dataclass
class ScraperField:
    """One column to extract relative to each matched item node.

    Empty ``selector`` means the item root element itself.
    """

    name: str
    selector: str = ""
    extract: ExtractKind = "text"
    attribute: str | None = None


@dataclass
class ScraperSpec:
    """Heuristic / planned scrape recipe (inspect → plan), separate from execution."""

    entry_url: str
    item_selector: str
    fields: list[ScraperField] = field(default_factory=list)
    pagination_strategy: Literal["none", "link_next"] = "none"
    pagination_selector: str | None = None
    max_pages: int = 3
    allowed_domains: list[str] = field(default_factory=list)
    requires_js: bool = False
    runtime: str = "requests_html"
    confidence: float = 0.0
    planner_notes: list[str] = field(default_factory=list)
    # Set by auto_scrape probe; drives one-row vs many-row extraction in ScrapeConnector.
    extraction_mode: Literal["repeated_items", "forum_first_post", "main_text"] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly structure (e.g. for ``SourceSpec.scraper_spec``)."""

        payload = asdict(self)
        payload["fields"] = [asdict(f) for f in self.fields]
        return payload

    def to_pagination_dict(self) -> dict[str, Any]:
        """Shape compatible with :class:`SourceSpec.pagination` / ScrapeConnector."""

        out: dict[str, Any] = {"max_pages": self.max_pages}
        if self.pagination_strategy == "link_next" and self.pagination_selector:
            out["next_page_selector"] = self.pagination_selector
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScraperSpec:
        """Rebuild from :meth:`to_dict` output."""

        fields_raw = data.get("fields") or []
        fields = [
            ScraperField(
                name=f["name"],
                selector=f.get("selector", ""),
                extract=f.get("extract", "text"),
                attribute=f.get("attribute"),
            )
            for f in fields_raw
        ]
        em = data.get("extraction_mode")
        if em is not None and em not in ("repeated_items", "forum_first_post", "main_text"):
            em = None
        entry_url = str(data.get("entry_url") or data.get("url") or "").strip()
        item_selector = str(data.get("item_selector") or data.get("itemSelector") or "").strip()
        return cls(
            entry_url=entry_url,
            item_selector=item_selector,
            fields=fields,
            pagination_strategy=data.get("pagination_strategy", "none"),
            pagination_selector=data.get("pagination_selector"),
            max_pages=int(data.get("max_pages", 3)),
            allowed_domains=list(data.get("allowed_domains") or []),
            requires_js=bool(data.get("requires_js", False)),
            runtime=str(data.get("runtime", "requests_html")),
            confidence=float(data.get("confidence", 0.0)),
            planner_notes=list(data.get("planner_notes") or []),
            extraction_mode=em,
        )
