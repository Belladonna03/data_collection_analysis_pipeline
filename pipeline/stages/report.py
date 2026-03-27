from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from pipeline.state import PipelineState


@dataclass
class ReportRunResult:
    report_path: str
    artifact_paths: Dict[str, str]


def _human_review_summary_lines(state: PipelineState) -> List[str]:
    lines: List[str] = []
    for stage in sorted(state.stages, key=lambda s: s.ordinal):
        if stage.review_supported:
            lines.append(
                f"- {stage.stage_id} {stage.display_name}: "
                f"review_required={stage.review_required}, review_file={stage.review_file or '-'}"
            )
    return lines


def _stage_summary_lines(state: PipelineState) -> List[str]:
    return [f"- [{stage.status}] {stage.stage_id} {stage.display_name}" for stage in sorted(state.stages, key=lambda s: s.ordinal)]


def run_report_stage(*, stage_dir: Path, state: PipelineState) -> ReportRunResult:
    stage_dir.mkdir(parents=True, exist_ok=True)
    report_path = stage_dir / "final_pipeline_report.md"
    annotate_stage = next((s for s in state.stages if s.short_name == "annotate"), None)
    annotate_line = (
        "- ANNOTATE: skipped (needs_labels=false; dataset passed through — see annotate_skipped_report_md / annotation_skipped.md)."
        if annotate_stage is not None and annotate_stage.status == "skipped"
        else "- ANNOTATE: auto-labeled samples and prepared low-confidence review queue."
    )
    lines = [
        "# Final Pipeline Report",
        "",
        "## Run Summary",
        f"- run_id: {state.run_id}",
        f"- pipeline_status: {state.pipeline_status}",
        f"- current_stage: {state.current_stage}",
        "",
        "## Stage Statuses",
        *_stage_summary_lines(state),
        "",
        "## What each agent did",
        "- COLLECT: discovered and collected candidate datasets.",
        "- QUALITY: detected issues, generated review artifacts, applied approved cleaning strategy.",
        annotate_line,
        "- AL: selected informative samples for additional human labeling.",
        "- TRAIN: produced baseline model and metrics artifacts.",
        "- REPORT: generated this retrospective summary.",
        "",
        "## Human Review Summary",
        *(_human_review_summary_lines(state) or ["- No review-enabled stages found."]),
        "",
        "## Metrics / key artifacts by stage",
    ]
    for stage in sorted(state.stages, key=lambda s: s.ordinal):
        if not stage.artifacts:
            continue
        lines.append(f"### {stage.stage_id} {stage.display_name}")
        for key, value in sorted(stage.artifacts.items()):
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(
        [
            "## Retrospective / limitations",
            "- Some stages use lightweight defaults suitable for demos (e.g. COLLECT/AL simulation).",
            "- TRAIN fits a TF-IDF + LogisticRegression baseline (DummyClassifier only if data are degenerate); see metrics.json, classification_report.txt, model.joblib/model.pkl.",
            "- Review quality depends on completeness and correctness of human-edited review files.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return ReportRunResult(
        report_path=str(report_path.resolve()),
        artifact_paths={"final_pipeline_report_md": str(report_path.resolve())},
    )

