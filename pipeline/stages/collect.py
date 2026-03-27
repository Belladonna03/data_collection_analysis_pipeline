from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from agents.data_collection.schemas import ProfileFieldSource, TopicProfile

from pipeline.collect_cli_state import clear_collect_cli_state, try_load_collect_cli_state
from pipeline.collect_snapshots import load_user_selection, try_load_discovery_snapshot

if TYPE_CHECKING:
    from agents.data_collection.schemas import CollectionResult
    from agents.data_collection_agent import DataCollectionAgent

REQUIRED_PROFILE_FIELDS = ("topic", "modality", "language", "task_type", "size_target", "needs_labels")


@dataclass
class CollectStageRunResult:
    merged_output_path: str
    artifact_paths: Dict[str, str]
    source_count: int
    row_count: int
    terminal_summary: str = ""


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "да"}:
        return True
    if text in {"0", "false", "no", "n", "нет"}:
        return False
    return None


def build_topic_profile(
    config: Dict[str, Any],
    *,
    topic: Optional[str] = None,
    modality: Optional[str] = None,
    language: Optional[str] = None,
    task_type: Optional[str] = None,
    size_target: Optional[int] = None,
    needs_labels: Optional[bool] = None,
) -> TopicProfile:
    cfg_profile = dict(config.get("topic_profile") or {})
    coll = config.get("collection") or {}
    cfg_profile.update(dict(coll.get("topic_profile") or {}))
    coll_topic = coll.get("topic")
    if coll_topic is not None and str(coll_topic).strip() and not (cfg_profile.get("topic") or "").strip():
        cfg_profile["topic"] = str(coll_topic).strip()
    coll_defaults = dict(coll.get("defaults") or {})
    for key in ("modality", "language", "task_type", "needs_labels"):
        if cfg_profile.get(key) is None and coll_defaults.get(key) is not None:
            cfg_profile[key] = coll_defaults[key]
    if cfg_profile.get("size_target") is None and coll_defaults.get("size_target") is not None:
        cfg_profile["size_target"] = coll_defaults["size_target"]
    st_raw = size_target if size_target is not None else cfg_profile.get("size_target")
    size_parsed: Optional[int] = None
    if st_raw is not None:
        size_parsed = int(st_raw)
    return TopicProfile(
        topic=topic if topic is not None else cfg_profile.get("topic"),
        modality=modality if modality is not None else cfg_profile.get("modality"),
        language=language if language is not None else cfg_profile.get("language"),
        task_type=task_type if task_type is not None else cfg_profile.get("task_type"),
        size_target=size_parsed,
        needs_labels=needs_labels if needs_labels is not None else _coerce_bool(cfg_profile.get("needs_labels")),
    )


def _default_size_target_from_config(config: Dict[str, Any]) -> int:
    c = config.get("collection") or {}
    defaults = c.get("defaults") or {}
    if defaults.get("size_target") is not None:
        return int(defaults["size_target"])
    cfg_profile = dict(config.get("topic_profile") or {})
    cfg_profile.update(dict(c.get("topic_profile") or {}))
    if cfg_profile.get("size_target") is not None:
        return int(cfg_profile["size_target"])
    return 10_000


_MODALITY_KEYWORDS: dict[str, list[str]] = {
    "text": ["text", "nlp", "nlp ", " caption", "transcript"],
    "image": ["image", "images", "photo", "vision"],
    "audio": ["audio", "speech", "sound"],
    "video": ["video"],
    "tabular": ["tabular", "table", " csv", "parquet", "rows", "database"],
}


def _infer_modality(topic: str) -> str:
    t = topic.casefold()
    for mod, keywords in _MODALITY_KEYWORDS.items():
        for kw in keywords:
            if kw.strip() and kw.casefold() in t:
                return mod if mod != "tabular" else "tabular"
    if re.search(r"\b(match|score|atp|wta|tennis|nba|stats|row)s?\b", t):
        return "tabular"
    return "tabular"


def _infer_language(topic: str) -> str:
    t = topic.casefold()
    if any(x in t for x in (" russian", " rus ", "русск", " казах", "kazakh")):
        return "russian" if "казах" not in t and "kazakh" not in t else "kazakh"
    if any(x in t for x in ("french", " german", "spanish ", "italian", "mandarin", "chinese text")):
        return "multilingual"
    return "english"


def infer_topic_profile(
    config: Dict[str, Any],
    *,
    topic: Optional[str] = None,
    modality: Optional[str] = None,
    language: Optional[str] = None,
    task_type: Optional[str] = None,
    size_target: Optional[int] = None,
    needs_labels: Optional[bool] = None,
) -> TopicProfile:
    """Build a full :class:`TopicProfile` with defaults; only *topic* (or config topic) is required."""

    merged = build_topic_profile(
        config,
        topic=topic,
        modality=modality,
        language=language,
        task_type=task_type,
        size_target=size_target,
        needs_labels=needs_labels,
    )
    t = merged.topic
    if t is None or not str(t).strip():
        cfg_t = (config.get("topic_profile") or {}).get("topic")
        c = config.get("collection") or {}
        cfg_t = cfg_t or (c.get("topic_profile") or {}).get("topic") or c.get("topic")
        if cfg_t:
            t = str(cfg_t).strip()
    if not t or not str(t).strip():
        raise ValueError(
            "Topic is required: pass --topic or set topic in config (topic_profile.topic or collection.topic)."
        )
    t = str(t).strip()

    inferred_modality = merged.modality or _infer_modality(t)
    inferred_language = merged.language or _infer_language(t)
    inferred_task = merged.task_type or "analysis"
    inferred_size: int = int(merged.size_target) if merged.size_target is not None else _default_size_target_from_config(config)
    inferred_labels = False if merged.needs_labels is None else bool(merged.needs_labels)

    user_explicit = ProfileFieldSource.USER_EXPLICIT.value
    confirmed = ProfileFieldSource.CONFIRMED_BY_USER.value
    inferred = ProfileFieldSource.INFERRED_HINT.value

    prov: Dict[str, str] = {}

    if topic is not None and str(topic).strip():
        prov["topic"] = user_explicit
    else:
        prov["topic"] = confirmed

    prov["modality"] = confirmed if merged.modality is not None else inferred
    prov["language"] = confirmed if merged.language is not None else inferred
    prov["task_type"] = confirmed if merged.task_type is not None else inferred
    prov["size_target"] = confirmed if merged.size_target is not None else inferred
    prov["needs_labels"] = confirmed if merged.needs_labels is not None else inferred

    return TopicProfile(
        topic=t,
        modality=inferred_modality,
        language=inferred_language,
        task_type=inferred_task,
        size_target=inferred_size,
        needs_labels=inferred_labels,
        constraints=dict(merged.constraints),
        discovery_hints=dict(merged.discovery_hints),
        field_provenance=prov,
    )


def validate_topic_profile(profile: TopicProfile) -> None:
    missing = [name for name in REQUIRED_PROFILE_FIELDS if getattr(profile, name) is None]
    if missing:
        raise ValueError(
            "Collect stage requires a complete topic profile. Missing fields: "
            + ", ".join(missing)
            + ". Provide them via --topic/--modality/--language/--task-type/--size-target/--needs-labels "
            + "or in config under topic_profile / collection.topic_profile."
        )


def topic_profiles_match_cli_session(a: TopicProfile, b: TopicProfile) -> bool:
    """Return True if persisted collection CLI state likely belongs to the same topic profile."""
    return (
        a.topic == b.topic
        and a.modality == b.modality
        and a.language == b.language
        and a.task_type == b.task_type
        and a.size_target == b.size_target
        and a.needs_labels == b.needs_labels
    )


def run_terminal_topic_clarification(agent: "DataCollectionAgent") -> None:
    """Ask required topic-profile questions in the terminal (customer / stakeholder)."""

    print(
        "Уточнение профиля сбора данных — ответьте в терминале (пустой ввод пропускается).",
        flush=True,
    )
    next_q = agent.conversation_manager.get_next_question()
    while next_q:
        print(next_q, flush=True)
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt) as exc:
            raise ValueError("Ввод прерван; профиль темы не заполнен.") from exc
        if not line:
            continue
        next_q = agent.conversation_manager.handle_user_message(line)
    if not agent.conversation_manager.is_ready_for_discovery():
        raise ValueError("Профиль темы неполон после опроса в терминале.")


def mark_profile_confirmed_for_interactive_run(profile: TopicProfile) -> None:
    """Mark all populated fields confirmed so :meth:`DataCollectionAgent.interactive_run` passes clarification gate."""

    confirmed = ProfileFieldSource.CONFIRMED_BY_USER.value
    for field in REQUIRED_PROFILE_FIELDS:
        if getattr(profile, field) is not None:
            profile.field_provenance[field] = confirmed


def run_collect_stage(
    *,
    config_payload: Dict[str, Any],
    stage_dir: Path,
    profile: Optional[TopicProfile] = None,
    interactive_terminal: bool = False,
) -> CollectStageRunResult:
    from agents.data_collection.planner import StrategyPlanner
    from agents.data_collection_agent import DataCollectionAgent

    cfg = dict(config_payload)
    storage_cfg = dict(cfg.get("storage", {}))
    storage_cfg["artifacts_dir"] = str(stage_dir.resolve())
    cfg["storage"] = storage_cfg

    selection_path = stage_dir / "selection" / "user_selection.json"
    if selection_path.exists():
        from pipeline.stages.collect_reports import finalize_collect_stage_artifacts

        discovery = try_load_discovery_snapshot(stage_dir)
        if discovery is None:
            raise ValueError(
                "user_selection.json exists but discovery_snapshot.json is missing. "
                "Run `python run_pipeline.py collect discover ...` for this run first."
            )
        selection = load_user_selection(stage_dir)
        planner = StrategyPlanner()
        plan = planner.build_plan_from_selected_candidates(
            discovery.topic_profile,
            discovery.candidates_by_key(),
            selection.candidate_keys,
        )
        agent = DataCollectionAgent(config=cfg)
        agent.session.topic_profile = discovery.topic_profile
        result: CollectionResult = agent.run_prepared_plan(plan)
        merged_path = result.artifacts.get("merged_dataframe")
        if not merged_path:
            raise ValueError("Collection completed without merged_dataframe artifact.")
        fin = finalize_collect_stage_artifacts(
            stage_dir=stage_dir,
            result=result,
            topic_profile=discovery.topic_profile,
            plan=plan,
            selection=selection,
            discovery=discovery,
        )
        merged_artifacts = {**result.artifacts, **fin.artifact_paths}
        return CollectStageRunResult(
            merged_output_path=fin.merged_parquet_path,
            artifact_paths=merged_artifacts,
            source_count=len(result.per_source_stats),
            row_count=fin.row_count,
            terminal_summary=fin.terminal_summary,
        )

    if try_load_discovery_snapshot(stage_dir) is not None:
        raise ValueError(
            "This run has a discovery snapshot but no source selection yet. "
            "Complete: `python run_pipeline.py collect recommend` (optional), "
            "then `collect select --ids ...`, then `collect run`. "
            "For legacy one-shot interactive collection, remove `01_collect/discovery/` first."
        )

    if interactive_terminal:
        agent = DataCollectionAgent(config=cfg)
        run_terminal_topic_clarification(agent)
        profile = agent.session.topic_profile
        validate_topic_profile(profile)
        reuse = False
        result: CollectionResult = agent.interactive_run(reuse_session_plans=reuse)
    else:
        if profile is None:
            raise ValueError(
                "Collect run needs a topic profile: set collection.topic (and optional fields) in config, "
                "pass --topic / other collect flags, or use `collect run --interactive` / collection.interactive "
                "to ask the customer in the terminal."
            )
        validate_topic_profile(profile)
        mark_profile_confirmed_for_interactive_run(profile)
        agent = DataCollectionAgent(config=cfg)
        agent.session.topic_profile = profile
        loaded = try_load_collect_cli_state(stage_dir)
        reuse = False
        if loaded is not None:
            persisted_profile, candidates, plans, selected = loaded
            if topic_profiles_match_cli_session(persisted_profile, profile):
                agent.session.candidates = candidates
                agent.session.proposed_plans = plans
                agent.session.selected_plan = selected
                reuse = bool(candidates or plans or selected is not None)
        result = agent.interactive_run(reuse_session_plans=reuse)
    clear_collect_cli_state(stage_dir)
    merged_path = result.artifacts.get("merged_dataframe")
    if not merged_path:
        raise ValueError("Collection completed without merged_dataframe artifact.")
    from pipeline.stages.collect_reports import finalize_collect_stage_artifacts

    fin = finalize_collect_stage_artifacts(
        stage_dir=stage_dir,
        result=result,
        topic_profile=profile,
        plan=None,
        selection=None,
        discovery=None,
    )
    merged_artifacts = {**result.artifacts, **fin.artifact_paths}
    return CollectStageRunResult(
        merged_output_path=fin.merged_parquet_path,
        artifact_paths=merged_artifacts,
        source_count=len(result.per_source_stats),
        row_count=fin.row_count,
        terminal_summary=fin.terminal_summary,
    )
