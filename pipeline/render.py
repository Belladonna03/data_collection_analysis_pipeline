from __future__ import annotations

from typing import List

from pipeline.state import PipelineState, StageState


def stage_status_line(stage: StageState) -> str:
    return f"[{stage.status}] {stage.stage_id} {stage.display_name}"


def render_pipeline_summary(state: PipelineState) -> str:
    current = next((s for s in state.stages if s.stage_id == state.current_stage), None)
    current_status = current.status if current else "unknown"
    lines = [
        f"run_id: {state.run_id}",
        f"pipeline_status: {state.pipeline_status}",
        f"current_stage: {state.current_stage}",
        f"current_stage_status: {current_status}",
        "stages:",
    ]
    lines.extend([f"  {stage_status_line(stage)}" for stage in sorted(state.stages, key=lambda s: s.ordinal)])
    lines.extend(["next action:", f"  {state.next_action}"])
    return "\n".join(lines)


def render_current_stage_summary(state: PipelineState) -> str:
    current = next((s for s in state.stages if s.stage_id == state.current_stage), None)
    if current is None:
        return f"current_stage: {state.current_stage} (not found in registry)"
    lines: List[str] = [
        f"stage: {current.stage_id} {current.display_name}",
        f"status: {current.status}",
        f"review_supported: {current.review_supported}",
        f"review_required: {current.review_required}",
    ]
    if current.review_file:
        lines.append(f"review_file: {current.review_file}")
    if current.notes:
        lines.append(f"notes: {current.notes}")
    if current.artifacts:
        lines.append("artifacts:")
        for key, value in sorted(current.artifacts.items()):
            lines.append(f"  - {key}: {value}")
    else:
        lines.append("artifacts: (none)")
    return "\n".join(lines)


def render_stage_list(state: PipelineState) -> str:
    return "\n".join(stage_status_line(stage) for stage in sorted(state.stages, key=lambda s: s.ordinal))

