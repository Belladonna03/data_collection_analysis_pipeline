from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.data_collection.schemas import (
    CollectionPlan,
    SessionStatus,
    SourceCandidate,
    TopicProfile,
)


@dataclass
class CollectionSessionState:
    """Mutable state for one collection session."""

    topic_profile: TopicProfile = field(default_factory=TopicProfile)
    candidates: list[SourceCandidate] = field(default_factory=list)
    proposed_plans: list[CollectionPlan] = field(default_factory=list)
    selected_plan: CollectionPlan | None = None
    status: SessionStatus = SessionStatus.CLARIFYING
    messages: list[dict[str, str]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


def create_empty_session() -> CollectionSessionState:
    """Create a new empty session."""

    return CollectionSessionState()


def append_message(
    session: CollectionSessionState,
    role: str,
    content: str,
) -> CollectionSessionState:
    """Append a message to session history."""

    session.messages.append({"role": role, "content": content})
    return session


def update_status(
    session: CollectionSessionState,
    status: SessionStatus,
) -> CollectionSessionState:
    """Update the current session status."""

    session.status = status
    return session
