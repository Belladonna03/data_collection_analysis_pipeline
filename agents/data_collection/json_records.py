"""Shared JSON → list[dict] extraction for API and JSON URL (scrape) modes."""

from __future__ import annotations

from typing import Any


def extract_json_records(payload: Any, response_path: str | None) -> list[dict[str, Any]]:
    """Resolve *response_path* on *payload* (dot-separated keys); return dict rows.

    If *response_path* is None: accept a root JSON list of objects, a single object
    (wrapped as one row), or raise if the shape is not usable.
    """

    current: Any = payload
    if response_path:
        for part in response_path.split("."):
            if not isinstance(current, dict) or part not in current:
                raise ValueError(
                    f"Response path {response_path!r} was not found in JSON payload (failed at {part!r})."
                )
            current = current[part]

    if isinstance(current, list):
        return [item for item in current if isinstance(item, dict)]
    if isinstance(current, dict):
        return [current]
    raise ValueError("JSON payload does not contain record objects at the resolved path.")
