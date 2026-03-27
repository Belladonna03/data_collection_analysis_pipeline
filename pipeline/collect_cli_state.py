"""Persist DataCollectionAgent session fields between `collect discover|recommend|select|run` invocations."""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from agents.data_collection.schemas import CollectionPlan, SourceCandidate, SourceSpec, SourceType, TopicProfile

STATE_FILENAME = "collection_cli_state.json"

E = TypeVar("E", bound=Enum)


def collect_state_path(stage_dir: Path) -> Path:
    return stage_dir / STATE_FILENAME


def _json_prepare(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _json_prepare(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_prepare(x) for x in obj]
    return obj


def _enum_value(typ: Type[E], raw: str) -> E:
    return typ(raw)  # type: ignore[arg-type]


def topic_profile_from_dict(data: dict[str, Any]) -> TopicProfile:
    return TopicProfile(
        topic=data.get("topic"),
        modality=data.get("modality"),
        task_type=data.get("task_type"),
        language=data.get("language"),
        needs_labels=data.get("needs_labels"),
        size_target=data.get("size_target"),
        constraints=dict(data.get("constraints") or {}),
        discovery_hints=dict(data.get("discovery_hints") or {}),
        field_provenance=dict(data.get("field_provenance") or {}),
    )


def source_candidate_from_dict(data: dict[str, Any]) -> SourceCandidate:
    row = dict(data)
    row["source_type"] = _enum_value(SourceType, str(row["source_type"]))
    return SourceCandidate(**row)


def source_spec_from_dict(data: dict[str, Any]) -> SourceSpec:
    row = dict(data)
    row["type"] = _enum_value(SourceType, str(row["type"]))
    return SourceSpec(**row)


def collection_plan_from_dict(data: dict[str, Any]) -> CollectionPlan:
    return CollectionPlan(
        topic_profile=topic_profile_from_dict(dict(data["topic_profile"])),
        sources=[source_spec_from_dict(dict(x)) for x in data.get("sources", [])],
        rationale=str(data.get("rationale", "")),
        expected_schema=dict(data.get("expected_schema") or {}),
        warnings=list(data.get("warnings") or []),
    )


def snapshot_session(
    *,
    topic_profile: TopicProfile,
    candidates: list[SourceCandidate],
    proposed_plans: list[CollectionPlan],
    selected_plan: Optional[CollectionPlan],
) -> dict[str, Any]:
    selected_index: Optional[int] = None
    if selected_plan is not None and proposed_plans:
        for i, plan in enumerate(proposed_plans):
            if plan is selected_plan:
                selected_index = i
                break
    return _json_prepare(
        {
            "schema_version": 1,
            "topic_profile": asdict(topic_profile),
            "candidates": [asdict(c) for c in candidates],
            "proposed_plans": [asdict(p) for p in proposed_plans],
            "selected_plan_index": selected_index,
        }
    )


def load_snapshot(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported collection_cli_state schema: {data.get('schema_version')}")
    return data


def apply_snapshot_to_session_dict(data: dict[str, Any]) -> tuple[TopicProfile, list[SourceCandidate], list[CollectionPlan], Optional[CollectionPlan]]:
    profile = topic_profile_from_dict(dict(data["topic_profile"]))
    candidates = [source_candidate_from_dict(dict(x)) for x in data.get("candidates", [])]
    plans = [collection_plan_from_dict(dict(x)) for x in data.get("proposed_plans", [])]
    idx = data.get("selected_plan_index")
    selected: Optional[CollectionPlan] = None
    if isinstance(idx, int) and 0 <= idx < len(plans):
        selected = plans[idx]
    return profile, candidates, plans, selected


def save_collect_cli_state(
    stage_dir: Path,
    *,
    topic_profile: TopicProfile,
    candidates: list[SourceCandidate],
    proposed_plans: list[CollectionPlan],
    selected_plan: Optional[CollectionPlan],
) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload = snapshot_session(
        topic_profile=topic_profile,
        candidates=candidates,
        proposed_plans=proposed_plans,
        selected_plan=selected_plan,
    )
    out = collect_state_path(stage_dir)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def try_load_collect_cli_state(stage_dir: Path) -> Optional[tuple[TopicProfile, list[SourceCandidate], list[CollectionPlan], Optional[CollectionPlan]]]:
    path = collect_state_path(stage_dir)
    if not path.exists():
        return None
    try:
        data = load_snapshot(path)
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
        return None
    return apply_snapshot_to_session_dict(data)


def clear_collect_cli_state(stage_dir: Path) -> None:
    path = collect_state_path(stage_dir)
    if path.exists():
        path.unlink()

