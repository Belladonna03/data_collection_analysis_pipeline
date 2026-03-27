from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pipeline.cli_syntax import (
    al_review_file_placeholder,
    annotate_review_file_placeholder,
    collect_discover,
    pipeline_artifacts,
    quality_review_decision_placeholder,
    stage_run,
)
from pipeline.registry import STAGE_REGISTRY, StageDefinition

StageStatus = Literal[
    "pending",
    "running",
    "awaiting_review",
    "approved",
    "completed",
    "failed",
    "skipped",
]
PipelineStatus = Literal["idle", "running", "awaiting_review", "completed", "failed"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StageState:
    stage_id: str
    short_name: str
    display_name: str
    ordinal: int
    review_supported: bool
    artifact_dir_name: str
    status: StageStatus = "pending"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    review_required: bool = False
    review_file: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    last_error: Optional[str] = None
    notes: str = ""


@dataclass
class PipelineState:
    schema_version: int
    run_id: str
    pipeline_status: PipelineStatus
    current_stage: str
    created_at: str
    updated_at: str
    run_dir: str
    stages: List[StageState]
    next_action: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "PipelineState":
        required = {"run_id", "pipeline_status", "current_stage", "created_at", "updated_at", "run_dir", "stages"}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"State file missing required keys: {', '.join(missing)}")
        stages = [StageState(**item) for item in payload["stages"]]
        return PipelineState(
            schema_version=int(payload.get("schema_version", 1)),
            run_id=str(payload["run_id"]),
            pipeline_status=payload["pipeline_status"],
            current_stage=str(payload["current_stage"]),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            run_dir=str(payload["run_dir"]),
            stages=stages,
            next_action=str(payload.get("next_action", "")),
        )


class PipelineStateManager:
    def __init__(self, runs_root: Path = Path("artifacts/runs")) -> None:
        self.runs_root = runs_root
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.active_run_file = self.runs_root / ".active_run"

    def create_run(self, run_id: Optional[str] = None) -> PipelineState:
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        for stage in STAGE_REGISTRY:
            (run_dir / stage.artifact_dir_name).mkdir(parents=True, exist_ok=True)
        stages = [self._init_stage_state(stage) for stage in STAGE_REGISTRY]
        state = PipelineState(
            schema_version=1,
            run_id=run_id,
            pipeline_status="idle",
            current_stage=STAGE_REGISTRY[0].stage_id,
            created_at=now_iso(),
            updated_at=now_iso(),
            run_dir=str(run_dir.resolve()),
            stages=stages,
            next_action=collect_discover(),
        )
        self.save(state)
        self.set_active_run(run_id)
        return state

    def load_run(self, run_id: str) -> PipelineState:
        path = self.runs_root / run_id / "pipeline_state.json"
        if not path.exists():
            raise ValueError(f"Run '{run_id}' does not exist or has no pipeline_state.json.")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Run '{run_id}' has corrupted pipeline_state.json: {exc}") from exc
        return PipelineState.from_dict(payload)

    def load_latest_run(self) -> Optional[PipelineState]:
        candidates = [d for d in self.runs_root.iterdir() if d.is_dir()]
        for run_dir in sorted(candidates, key=lambda p: (p.stat().st_mtime, p.name), reverse=True):
            path = run_dir / "pipeline_state.json"
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return PipelineState.from_dict(payload)
            except Exception:
                continue
        return None

    def load_active_or_latest(self) -> Optional[PipelineState]:
        active = self.get_active_run_id()
        if active:
            try:
                return self.load_run(active)
            except Exception:
                pass
        return self.load_latest_run()

    def set_active_run(self, run_id: str) -> None:
        self.active_run_file.write_text(run_id, encoding="utf-8")

    def get_active_run_id(self) -> Optional[str]:
        if not self.active_run_file.exists():
            return None
        value = self.active_run_file.read_text(encoding="utf-8").strip()
        return value or None

    def save(self, state: PipelineState) -> None:
        self._refresh_derived_fields(state)
        out_path = Path(state.run_dir) / "pipeline_state.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp_path.replace(out_path)

    def set_current_stage(self, state: PipelineState, stage_id: str) -> None:
        self._require_stage(state, stage_id)
        state.current_stage = stage_id

    def update_stage_status(self, state: PipelineState, stage_id: str, status: StageStatus, *, note: Optional[str] = None) -> None:
        stage = self._require_stage(state, stage_id)
        previous = stage.status
        stage.status = status
        ts = now_iso()
        if status == "running":
            stage.started_at = ts
        if status in {"awaiting_review", "approved", "completed", "failed", "skipped"}:
            stage.finished_at = ts
        if note:
            stage.notes = note
        elif previous != status:
            stage.notes = f"status changed: {previous} -> {status}"

    def record_artifact(self, state: PipelineState, stage_id: str, key: str, path: str) -> None:
        stage = self._require_stage(state, stage_id)
        stage.artifacts[key] = path

    def record_artifacts(self, state: PipelineState, stage_id: str, artifacts: Dict[str, str]) -> None:
        stage = self._require_stage(state, stage_id)
        stage.artifacts.update(artifacts)

    def record_review_file(self, state: PipelineState, stage_id: str, review_path: str) -> None:
        stage = self._require_stage(state, stage_id)
        stage.review_file = review_path
        stage.review_required = True

    def clear_review_requirement(self, state: PipelineState, stage_id: str) -> None:
        stage = self._require_stage(state, stage_id)
        stage.review_required = False

    def set_next_action(self, state: PipelineState, text: str) -> None:
        state.next_action = text.strip()

    def _refresh_derived_fields(self, state: PipelineState) -> None:
        state.updated_at = now_iso()
        statuses = [s.status for s in state.stages]
        state.pipeline_status = self._resolve_pipeline_status(statuses)
        if state.pipeline_status != "completed":
            for stage in sorted(state.stages, key=lambda s: s.ordinal):
                if stage.status not in {"completed", "skipped"}:
                    state.current_stage = stage.stage_id
                    break
        if not state.next_action:
            state.next_action = self._suggest_next_action(state)

    @staticmethod
    def _resolve_pipeline_status(statuses: List[str]) -> PipelineStatus:
        if any(status == "failed" for status in statuses):
            return "failed"
        if any(status == "awaiting_review" for status in statuses):
            return "awaiting_review"
        if all(status in {"completed", "skipped"} for status in statuses):
            return "completed"
        if any(status == "running" for status in statuses):
            return "running"
        return "idle"

    @staticmethod
    def _suggest_next_action(state: PipelineState) -> str:
        for stage in sorted(state.stages, key=lambda s: s.ordinal):
            if stage.status == "awaiting_review":
                if stage.short_name == "quality":
                    return quality_review_decision_placeholder()
                if stage.short_name == "annotate":
                    return annotate_review_file_placeholder()
                if stage.short_name == "al":
                    return al_review_file_placeholder()
                return f"Resolve human review for stage {stage.display_name}."
        for stage in sorted(state.stages, key=lambda s: s.ordinal):
            if stage.status in {"pending", "approved"}:
                if stage.short_name == "collect":
                    return collect_discover()
                return stage_run(stage.short_name)
        return pipeline_artifacts()

    @staticmethod
    def _init_stage_state(defn: StageDefinition) -> StageState:
        return StageState(
            stage_id=defn.stage_id,
            short_name=defn.short_name,
            display_name=defn.display_name,
            ordinal=defn.ordinal,
            review_supported=defn.review_supported,
            artifact_dir_name=defn.artifact_dir_name,
        )

    @staticmethod
    def _require_stage(state: PipelineState, stage_id: str) -> StageState:
        for stage in state.stages:
            if stage.stage_id == stage_id:
                return stage
        raise ValueError(f"Stage '{stage_id}' is not registered in this pipeline state.")

