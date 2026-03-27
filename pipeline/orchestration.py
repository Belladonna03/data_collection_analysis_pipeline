from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from pipeline.cli_syntax import stage_run
from pipeline.registry import STAGE_BY_SHORT_NAME
from pipeline.state import PipelineState, PipelineStateManager


def get_pipeline_orchestration_flags(cfg: Dict[str, Any]) -> Dict[str, bool]:
    """Defaults align with documented config.yaml (strict guards on by default)."""

    pipeline = dict(cfg.get("pipeline") or {})
    return {
        "skip_annotation_if_needs_labels_false": bool(
            pipeline.get("skip_annotation_if_needs_labels_false", True)
        ),
        "fail_if_text_missing": bool(pipeline.get("fail_if_text_missing", True)),
    }


def skip_annotation_policy(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (skip, reason) when annotate should not run. Empty reason when skip is False."""

    pipeline = dict(cfg.get("pipeline") or {})
    if not bool(pipeline.get("run_annotation", True)):
        return True, "ANNOTATE skipped: pipeline.run_annotation is false."
    if not get_pipeline_orchestration_flags(cfg)["skip_annotation_if_needs_labels_false"]:
        return False, ""
    from pipeline.stages.collect import build_topic_profile

    profile = build_topic_profile(cfg)
    if profile.needs_labels is False:
        return (
            True,
            "ANNOTATE skipped: topic profile has needs_labels=false and "
            "pipeline.skip_annotation_if_needs_labels_false is enabled.",
        )
    return False, ""


def should_skip_annotation_for_config(cfg: Dict[str, Any]) -> bool:
    """True when annotate should be skipped (needs_labels gate, run_annotation, etc.)."""

    return skip_annotation_policy(cfg)[0]


def resolve_annotation_modality(cfg: Dict[str, Any]) -> str:
    ann = dict(cfg.get("annotation") or {})
    if ann.get("modality"):
        return str(ann.get("modality")).strip().lower()
    from pipeline.stages.collect import build_topic_profile

    return str(build_topic_profile(cfg).modality or "text").strip().lower()


def require_text_modality_for_annotation(cfg: Dict[str, Any]) -> None:
    """AnnotationAgent is text-only; refuse non-text before any model work."""

    mod = resolve_annotation_modality(cfg)
    if mod != "text":
        raise ValueError(
            f"ANNOTATE: AnnotationAgent supports text modality only; resolved modality is {mod!r}. "
            "Use text data or set pipeline.run_annotation to false to skip this stage."
        )


def apply_annotation_stage_skip(
    manager: PipelineStateManager,
    state: PipelineState,
    *,
    passthrough_dataset_path: str,
    annotate_stage_dir: Path,
    reason: str,
) -> None:
    """Mark ANNOTATE as skipped, record passthrough for downstream resolvers, advance to AL."""

    ann_def = STAGE_BY_SHORT_NAME["annotate"]
    annotate_stage_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = annotate_stage_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    skip_report = reports_dir / "annotation_skipped.md"
    skip_report.write_text(
        "# Annotation stage skipped\n\n"
        f"{reason}\n\n"
        f"- passthrough_dataset: {passthrough_dataset_path}\n",
        encoding="utf-8",
    )
    manager.record_artifact(state, ann_def.stage_id, "annotate_skipped_passthrough", passthrough_dataset_path)
    manager.record_artifact(
        state, ann_def.stage_id, "annotate_skipped_report_md", str(skip_report.resolve())
    )
    manager.record_artifact(state, ann_def.stage_id, "skipped_reason", reason)
    manager.record_artifact(state, ann_def.stage_id, "annotate_skipped_reason", reason)
    manager.update_stage_status(state, ann_def.stage_id, "skipped", note=reason)
    manager.set_current_stage(state, STAGE_BY_SHORT_NAME["al"].stage_id)
    manager.set_next_action(state, stage_run("al"))
    manager.save(state)
