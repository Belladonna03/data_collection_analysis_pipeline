from __future__ import annotations

from typing import Any

from agents.data_collection.connectors.base import BaseConnector
from agents.data_collection.schemas import SourceType


class ConnectorNotRegisteredError(LookupError):
    """Raised when a connector is requested but not registered."""


class ConnectorRegistry:
    """Registry for connector classes."""

    def __init__(self) -> None:
        self._connectors: dict[SourceType, type[BaseConnector]] = {}

    def register(
        self,
        source_type: SourceType | str,
        connector_cls: type[BaseConnector],
    ) -> None:
        """Register a connector class for a source type."""

        normalized_type = self._coerce_source_type(source_type)
        self._connectors[normalized_type] = connector_cls

    def get(self, source_type: SourceType | str) -> type[BaseConnector]:
        """Return the registered connector class."""

        normalized_type = self._coerce_source_type(source_type)
        try:
            return self._connectors[normalized_type]
        except KeyError as exc:
            available = ", ".join(self.available_source_types()) or "none"
            raise ConnectorNotRegisteredError(
                f"Connector for source type '{normalized_type.value}' is not registered. "
                f"Available types: {available}."
            ) from exc

    def create(self, source_type: SourceType | str, **kwargs: Any) -> BaseConnector:
        """Instantiate a registered connector."""

        connector_cls = self.get(source_type)
        return connector_cls(**kwargs)

    def available_source_types(self) -> list[str]:
        """Return registered source types."""

        return sorted(source_type.value for source_type in self._connectors)

    @staticmethod
    def _coerce_source_type(value: SourceType | str) -> SourceType:
        """Convert raw source type into SourceType."""

        if isinstance(value, SourceType):
            return value

        try:
            return SourceType(value)
        except ValueError as exc:
            raise ValueError(f"Unsupported source type: {value!r}.") from exc
