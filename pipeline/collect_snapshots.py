"""File-backed discovery / recommendations / selection artifacts for the collect stage CLI."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.data_collection.planner import StrategyPlanner
from agents.data_collection.schemas import CollectionPlan, SourceCandidate, TopicProfile

from pipeline.collect_cli_state import (
    collection_plan_from_dict,
    source_candidate_from_dict,
    topic_profile_from_dict,
)


def discovery_dir(stage_dir: Path) -> Path:
    return stage_dir / "discovery"


def recommendations_dir(stage_dir: Path) -> Path:
    return stage_dir / "recommendations"


def selection_dir(stage_dir: Path) -> Path:
    return stage_dir / "selection"


DISCOVERY_SNAPSHOT_FILE = "discovery_snapshot.json"
RECOMMENDATIONS_FILE = "recommendations.json"
USER_SELECTION_FILE = "user_selection.json"


def _slug_base(candidate: SourceCandidate) -> str:
    if candidate.normalized_source_id:
        safe = "".join(ch if ch.isalnum() or ch in "._:-" else "_" for ch in candidate.normalized_source_id).strip("_")
        return safe or "unnamed"
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", candidate.name.casefold()).strip("_")
    return slug or "unnamed"


def assign_stable_keys(candidates: Sequence[SourceCandidate]) -> List[Tuple[SourceCandidate, str]]:
    """Return (candidate, candidate_key) preserving discovery order; disambiguate duplicate base keys."""
    seen: Dict[str, int] = {}
    out: List[Tuple[SourceCandidate, str]] = []
    for c in candidates:
        base = f"{c.source_type.value}:{_slug_base(c)}"
        idx = seen.get(base, 0)
        seen[base] = idx + 1
        key = base if idx == 0 else f"{base}#{idx}"
        out.append((c, key))
    return out


def spec_id_for_numbered(candidate: SourceCandidate) -> str:
    """Match :meth:`StrategyPlanner._source_spec_id` without importing private patterns elsewhere."""
    return StrategyPlanner._source_spec_id(candidate)


@dataclass
class DiscoverySnapshot:
    """In-memory view of ``discovery_snapshot.json``."""

    topic_profile: TopicProfile
    ordered: List[Tuple[int, str, SourceCandidate]]  # (number, key, candidate)
    created_at: str

    def candidates_by_key(self) -> Dict[str, SourceCandidate]:
        return {key: c for _, key, c in self.ordered}

    def key_by_number(self) -> Dict[int, str]:
        return {num: key for num, key, _ in self.ordered}


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_discovery_snapshot(
    stage_dir: Path,
    *,
    topic_profile: TopicProfile,
    tagged: Sequence[Tuple[SourceCandidate, str]],
) -> Tuple[Path, List[Tuple[int, str, SourceCandidate]]]:
    rows: List[Dict[str, Any]] = []
    ordered: List[Tuple[int, str, SourceCandidate]] = []
    for i, (c, key) in enumerate(tagged, start=1):
        rows.append({"candidate_number": i, "candidate_key": key, "payload": _candidate_as_json_dict(c)})
        ordered.append((i, key, c))
    payload: Dict[str, Any] = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "topic_profile": _topic_as_json(topic_profile),
        "candidates": rows,
    }
    path = discovery_dir(stage_dir) / DISCOVERY_SNAPSHOT_FILE
    _json_dump(path, payload)
    return path, ordered


def _topic_as_json(tp: TopicProfile) -> Dict[str, Any]:
    from dataclasses import asdict

    return _json_prepare(asdict(tp))


def _candidate_as_json_dict(c: SourceCandidate) -> Dict[str, Any]:
    from dataclasses import asdict

    return _json_prepare(asdict(c))


def _json_prepare(obj: Any) -> Any:
    from enum import Enum

    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _json_prepare(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_prepare(x) for x in obj]
    return obj


def try_load_discovery_snapshot(stage_dir: Path) -> Optional[DiscoverySnapshot]:
    path = discovery_dir(stage_dir) / DISCOVERY_SNAPSHOT_FILE
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if int(raw.get("schema_version", 0)) != 2:
        return None
    tp = topic_profile_from_dict(dict(raw["topic_profile"]))
    ordered: List[Tuple[int, str, SourceCandidate]] = []
    for row in raw.get("candidates", []):
        num = int(row["candidate_number"])
        key = str(row["candidate_key"])
        cand = source_candidate_from_dict(dict(row["payload"]))
        ordered.append((num, key, cand))
    ordered.sort(key=lambda x: x[0])
    created_at = str(raw.get("created_at", ""))
    return DiscoverySnapshot(topic_profile=tp, ordered=ordered, created_at=created_at)


def save_recommendations(
    stage_dir: Path,
    *,
    combinations: List[Dict[str, Any]],
    plans_serialized: List[Dict[str, Any]],
) -> Path:
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "combinations": combinations,
        "plans": plans_serialized,
    }
    path = recommendations_dir(stage_dir) / RECOMMENDATIONS_FILE
    _json_dump(path, payload)
    return path


def try_load_recommendations(stage_dir: Path) -> Optional[Dict[str, Any]]:
    path = recommendations_dir(stage_dir) / RECOMMENDATIONS_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def plans_from_recommendations_file(data: Dict[str, Any]) -> List[CollectionPlan]:
    return [collection_plan_from_dict(dict(p)) for p in data.get("plans", [])]


def serialize_collection_plans(plans: Sequence[CollectionPlan]) -> List[Dict[str, Any]]:
    from dataclasses import asdict

    return [_json_prepare(asdict(p)) for p in plans]


@dataclass
class UserSelection:
    candidate_numbers: List[int]
    candidate_keys: List[str]
    entries: List[Dict[str, Any]]


def save_user_selection(stage_dir: Path, selection: UserSelection) -> Path:
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_numbers": selection.candidate_numbers,
        "candidate_keys": selection.candidate_keys,
        "entries": selection.entries,
    }
    path = selection_dir(stage_dir) / USER_SELECTION_FILE
    _json_dump(path, payload)
    return path


def load_user_selection(stage_dir: Path) -> UserSelection:
    path = selection_dir(stage_dir) / USER_SELECTION_FILE
    raw = json.loads(path.read_text(encoding="utf-8"))
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported user_selection schema_version: {raw.get('schema_version')}")
    return UserSelection(
        candidate_numbers=[int(x) for x in raw["candidate_numbers"]],
        candidate_keys=[str(x) for x in raw["candidate_keys"]],
        entries=[dict(e) for e in raw.get("entries", [])],
    )


def _truncate_visible(text: str, max_len: int) -> str:
    t = text.strip()
    if max_len <= 0 or len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def candidate_primary_resource(c: SourceCandidate) -> str:
    """Best-effort single line: where this candidate was found / how to reach it."""

    for attr in ("url", "repo_url", "endpoint"):
        v = getattr(c, attr, None)
        if v is not None and str(v).strip():
            return str(v).strip()
    if c.dataset_id is not None and str(c.dataset_id).strip():
        return f"dataset:{str(c.dataset_id).strip()}"
    if c.normalized_source_id is not None and str(c.normalized_source_id).strip():
        return str(c.normalized_source_id).strip()
    if c.dataset_ref is not None and str(c.dataset_ref).strip():
        return str(c.dataset_ref).strip()
    if c.platform is not None and str(c.platform).strip():
        return str(c.platform).strip()
    if getattr(c, "selector", None) and str(c.selector).strip():
        return str(c.selector).strip()
    return "—"


def print_discover_catalog(
    ordered: Sequence[Tuple[int, str, SourceCandidate]],
    *,
    verbose: bool = False,
    why_max_chars: int = 400,
) -> None:
    """Print discovery results. Default: name + resource + type only (readable console)."""

    if not verbose:
        print("Discovered sources (name — resource | type). Use --verbose for rationale, scores, risks.")
        for num, _key, c in ordered:
            res = candidate_primary_resource(c)
            demo = " [demo]" if c.is_demo_fallback else ""
            blocked = "" if c.is_executable else " [not executable]"
            print(f"  {num}. {c.name}{demo}{blocked} — {res} ({c.source_type.value})")
        return

    for num, _key, c in ordered:
        why_raw = (c.selection_rationale or c.description or "").strip() or "(no rationale text from discovery)"
        why = _truncate_visible(why_raw, why_max_chars)
        risk_parts = list(c.risks or [])
        if not c.is_executable and c.non_executable_reason:
            ne = _truncate_visible(str(c.non_executable_reason), 220)
            risk_parts.append(f"not executable: {ne}")
        if c.cons:
            risk_parts.append("cons: " + "; ".join(str(x) for x in c.cons[:3]))
        score = c.relevance_score
        score_s = f"{score:.3f}" if score is not None else "-"
        est = c.estimated_rows
        est_s = str(est) if est is not None else "-"
        print(f"{num}. {c.name}")
        print(f"   type: {c.source_type.value}")
        print(f"   resource: {candidate_primary_resource(c)}")
        print(f"   executable: {'yes' if c.is_executable else 'no'}")
        print(f"   score: {score_s}")
        print(f"   estimated_rows: {est_s}")
        print(f"   why: {why}")
        if risk_parts:
            print(f"   risks: {'; '.join(risk_parts)}")


def _executability_summary(plan: CollectionPlan) -> str:
    n = len(plan.sources)
    exe = sum(1 for s in plan.sources if s.is_executable)
    if exe == n:
        return f"all {n} source(s) marked executable in the plan"
    return f"{exe}/{n} sources marked executable"


def _coverage_summary(plan: CollectionPlan) -> str:
    names = [s.name for s in plan.sources]
    types = sorted({s.type.value for s in plan.sources})
    return f"sources: {', '.join(names)} | types: {', '.join(types)}"


def _merge_risk_summary(plan: CollectionPlan) -> str:
    hints = [w for w in plan.warnings if w]
    if not hints:
        return "no planner warnings"
    return "; ".join(hints[:4]) + ("…" if len(hints) > 4 else "")


def build_recommendation_combinations(
    plans: List[CollectionPlan],
    *,
    spec_id_to_number: Dict[str, int],
) -> List[Dict[str, Any]]:
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    combos: List[Dict[str, Any]] = []
    for i, plan in enumerate(plans):
        nums: List[int] = []
        for src in plan.sources:
            n = spec_id_to_number.get(src.id)
            if n is not None:
                nums.append(n)
        label = labels[i] if i < len(labels) else str(i + 1)
        combos.append(
            {
                "label": label,
                "candidate_numbers": nums,
                "why": plan.rationale,
                "warnings": list(plan.warnings),
                "coverage": _coverage_summary(plan),
                "merge_risk": _merge_risk_summary(plan),
                "executability": _executability_summary(plan),
            }
        )
    return combos


def print_recommendation_combinations(combos: Sequence[Dict[str, Any]]) -> None:
    print("Recommended combinations:")
    for combo in combos:
        nums = combo.get("candidate_numbers") or []
        joined = ",".join(str(n) for n in nums)
        print(f"{combo['label']}. [{joined}]")
        print(f"   why: {combo.get('why', '')}")
        if combo.get("warnings"):
            print(f"   warnings: {'; '.join(combo['warnings'])}")
        print(f"   coverage: {combo.get('coverage', '')}")
        print(f"   merge_risk: {combo.get('merge_risk', '')}")
        print(f"   executability: {combo.get('executability', '')}")


def spec_id_to_candidate_number(snapshot: DiscoverySnapshot) -> Dict[str, int]:
    m: Dict[str, int] = {}
    for num, _key, c in snapshot.ordered:
        m[spec_id_for_numbered(c)] = num
    return m


def format_collect_status_block(stage_dir: Path, stage_state: Any) -> str:
    """Human-readable collect UX summary for `collect status` (stage_state: StageState)."""
    lines: List[str] = ["collect workflow:"]
    snap = try_load_discovery_snapshot(stage_dir)
    topic = "(unknown)"
    if snap is not None:
        topic = snap.topic_profile.topic or "(unknown)"
    lines.append(f"  topic: {topic}")
    lines.append(f"  discovery_snapshot: {'yes' if snap is not None else 'no'}")
    if snap is None:
        lines.append("  candidates_count: 0")
    else:
        lines.append(f"  candidates_count: {len(snap.ordered)}")
    rec = try_load_recommendations(stage_dir)
    lines.append(f"  recommendations: {'yes' if rec is not None else 'no'}")
    sel_path = selection_dir(stage_dir) / USER_SELECTION_FILE
    if sel_path.exists():
        try:
            sel_sel = load_user_selection(stage_dir)
            lines.append(f"  selected_ids: {sel_sel.candidate_numbers}")
            lines.append(f"  selected_source_keys: {sel_sel.candidate_keys}")
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            lines.append("  selected_ids: (error reading selection file)")
    else:
        lines.append("  selected_ids: (none)")
    ready = snap is not None and sel_path.exists()
    lines.append(f"  ready_for_collect_run: {yesno(ready)}")
    art = stage_state.artifacts or {}
    lines.append(f"  merged_output: {art.get('collect_merged_output') or art.get('collect_merged_dataset_parquet') or '(not set)'}")
    lines.append(f"  data_card: {art.get('collect_data_card_md', '(not set)')}")
    lines.append(f"  eda_summary: {art.get('collect_eda_summary_md', '(not set)')}")
    lines.append(f"  source_summary: {art.get('collect_source_summary_json', '(not set)')}")
    lines.append(f"  eda_plots_dir: {art.get('collect_eda_plots_dir', '(not set)')}")
    return "\n".join(lines)


def yesno(b: bool) -> str:
    return "yes" if b else "no"


def parse_candidate_ids(arg: str) -> List[int]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if not parts:
        raise ValueError("--ids must contain at least one number (e.g. 1,2,4).")
    out: List[int] = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"Invalid id '{p}' in --ids (use integers like 1,2,4).")
        out.append(int(p))
    return out
