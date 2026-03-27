from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agents.data_collection.schemas import DiscoveryCapability, DiscoveryProvider, RawSearchHit


@dataclass
class SearchClientConfig:
    """Runtime config for search clients."""

    enabled: bool = True
    timeout: float = 20.0
    retries: int = 1
    rate_limit_per_second: float = 2.0
    max_results_per_query: int = 5
    token: str | None = None
    api_key: str | None = None
    username: str | None = None
    key: str | None = None
    command: str | None = None
    base_url: str | None = None
    domains: list[str] = field(default_factory=list)
    allow_domains: list[str] = field(default_factory=list)
    deny_domains: list[str] = field(default_factory=list)
    user_agent: str | None = None
    command_args: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SearchClientError(RuntimeError):
    """Raised when a search provider request fails."""


class BaseSearchClient(ABC):
    """Base interface for internet-backed search providers."""

    def __init__(self, config: SearchClientConfig | None = None) -> None:
        self.config = config or SearchClientConfig()

    @property
    @abstractmethod
    def provider(self) -> DiscoveryProvider:
        """Return the provider identifier."""

    @abstractmethod
    def check_capability(self) -> DiscoveryCapability:
        """Return provider availability for the current environment."""

    @abstractmethod
    def search(self, query: str) -> list[RawSearchHit]:
        """Search the provider and return raw hits."""

    @staticmethod
    def _drop_none(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop None values from a kwargs dictionary."""

        return {key: value for key, value in kwargs.items() if value is not None}
