from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SiteProbeResult:
    """Basic probe result for a remote site."""

    url: str
    reachable: bool
    content_type: str | None = None
    warnings: list[str] = field(default_factory=list)


class NetworkProbeSkill:
    """Very small MVP for probing URLs."""

    def probe(self, url: str) -> SiteProbeResult:
        """Return a basic, non-network probe result."""

        if not url.strip():
            return SiteProbeResult(
                url=url,
                reachable=False,
                warnings=["URL is empty."],
            )

        if url.startswith(("http://", "https://")):
            return SiteProbeResult(
                url=url,
                reachable=True,
                warnings=["TODO: replace stub probe with real network checks."],
            )

        return SiteProbeResult(
            url=url,
            reachable=False,
            warnings=["URL must start with http:// or https://."],
        )
