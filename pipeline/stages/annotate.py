from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from agents.annotation.labelstudio import (
    build_labelstudio_config,
    build_labelstudio_tasks,
    read_labelstudio_export_path,
)
from agents.annotation_agent import AnnotationAgent

# At least this fraction of rows must have non-empty text when fail_if_text_missing is on.
_MIN_NONEMPTY_TEXT_FRACTION = 0.05


def validate_annotate_text_input(
    df: pd.DataFrame,
    config_payload: Dict[str, Any],
    *,
    fail_if_text_missing: bool,
) -> None:
    """Fail fast for text modality when the text column is missing or nearly all empty."""

    ann = dict(config_payload.get("annotation") or {})
    modality = str(ann.get("modality", "text")).strip().lower()
    if modality != "text":
        return
    text_column = ann.get("text_column") or "text"
    if text_column not in df.columns:
        raise ValueError(
            f"ANNOTATE (text): column {text_column!r} is missing. "
            f"Available columns: {list(df.columns)}. "
            "Fix the dataset or set annotation.text_column to an existing column."
        )
    n = len(df)
    if n == 0:
        raise ValueError("ANNOTATE (text): input dataframe has no rows.")
    if not fail_if_text_missing:
        return
    series = df[text_column]
    nonempty = 0
    for value in series.tolist():
        if pd.isna(value):
            continue
        if str(value).strip():
            nonempty += 1
    if nonempty == 0:
        raise ValueError(
            f"ANNOTATE (text): column {text_column!r} has no non-empty values; "
            "refusing to auto-label. If text lives in another column (e.g. body), set annotation.text_column. "
            "Otherwise fix upstream quality/collect, or set pipeline.fail_if_text_missing=false to skip this check."
        )
    ratio = nonempty / n
    if ratio < _MIN_NONEMPTY_TEXT_FRACTION:
        raise ValueError(
            f"ANNOTATE (text): column {text_column!r} is nearly empty "
            f"({nonempty}/{n} non-empty rows, {ratio:.1%}). "
            "Refusing to run annotation. Fix the dataset or set pipeline.fail_if_text_missing=false."
        )


@dataclass
class AnnotateRunResult:
    auto_labeled_path: str
    review_queue_path: str
    report_path: str
    review_needed: bool
    review_count: int
    artifact_paths: Dict[str, str]


def run_annotate_stage(*, config_payload: Dict[str, Any], stage_dir: Path, input_df: pd.DataFrame) -> AnnotateRunResult:
    from pipeline.orchestration import get_pipeline_orchestration_flags, require_text_modality_for_annotation

    cfg = dict(config_payload)
    require_text_modality_for_annotation(cfg)
    flags = get_pipeline_orchestration_flags(cfg)
    # Only ``pipeline.fail_if_text_missing`` controls strict non-empty text checks.
    fail_text = bool(flags["fail_if_text_missing"])
    validate_annotate_text_input(input_df, cfg, fail_if_text_missing=fail_text)
    ann_cfg = dict(cfg.get("annotation", {}))
    ann_cfg["project_root"] = str(stage_dir.resolve())
    cfg["annotation"] = ann_cfg
    agent = AnnotationAgent(config=cfg)

    labeled_df = agent.auto_label(input_df)
    agent.generate_spec(labeled_df)
    agent.check_quality(labeled_df, human_label_column="human_label")
    review_queue = agent.prepare_review_queue(labeled_df)
    rq_meta = getattr(agent, "last_review_queue_meta", {}) or {}

    auto_labeled_path = stage_dir / "data" / "annotated_auto.parquet"
    auto_labeled_path.parent.mkdir(parents=True, exist_ok=True)
    export_auto = labeled_df.copy()
    export_auto.attrs.clear()
    label_col = str(ann_cfg.get("label_column") or "label")
    if label_col not in export_auto.columns:
        export_auto[label_col] = export_auto["auto_label"]
    export_auto.to_parquet(auto_labeled_path, index=False)

    review_queue_path = stage_dir / "review" / "review_queue.csv"
    review_queue_path.parent.mkdir(parents=True, exist_ok=True)
    review_queue.to_csv(review_queue_path, index=False)

    report_path = stage_dir / "reports" / "annotation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "# Annotation Stage Report",
        "",
        f"- total_rows: {len(labeled_df)}",
        f"- review_rows_in_queue: {int(len(review_queue))}",
        f"- review_needed: {bool(len(review_queue) > 0)}",
    ]
    if rq_meta.get("eligible_for_review") is not None:
        report_lines.append(f"- rows_flagged_needs_review: {int(rq_meta['eligible_for_review'])}")
    if rq_meta.get("truncated"):
        report_lines.append(
            f"- review_queue_truncated: yes (cap `annotation.review_queue_max_rows`={rq_meta.get('cap')}; "
            "only hardest cases exported for HITL)"
        )
    report_path.write_text(
        "\n".join(
            report_lines
            + [
                "",
                "Expected review columns in `review_queue.csv`:",
                "- annotation_id",
                "- auto_label",
                "- confidence",
                "- review_reason",
                "- human_label (to be filled by reviewer)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    artifacts = dict(agent.last_artifacts)
    artifacts.update(
        {
            "annotate_auto_labeled": str(auto_labeled_path.resolve()),
            "annotate_review_queue_csv": str(review_queue_path.resolve()),
            "annotate_report_md": str(report_path.resolve()),
        }
    )

    ls_cfg = dict(cfg.get("label_studio") or {})
    if ls_cfg.get("enabled", False) and len(review_queue):
        text_col = agent._resolve_text_column(labeled_df)
        tasks = build_labelstudio_tasks(
            review_queue,
            agent.task_config,
            text_column=text_col,
            id_column="annotation_id",
            include_predictions=True,
        )
        payload = json.dumps(tasks, ensure_ascii=False, indent=2) + "\n"
        xml_text = build_labelstudio_config(agent.task_config)
        task_fn = Path(str(ls_cfg.get("task_file") or "labelstudio_import.json")).name
        cfg_fn = Path(str(ls_cfg.get("config_file") or "label_config.xml")).name

        rq_path = review_queue_path.parent / "review_queue_labelstudio.json"
        rq_path.write_text(payload, encoding="utf-8")
        cfg_path = review_queue_path.parent / "labelstudio_config.xml"
        cfg_path.write_text(xml_text, encoding="utf-8")
        import_path = review_queue_path.parent / task_fn
        if import_path.resolve() != rq_path.resolve():
            import_path.write_text(payload, encoding="utf-8")
        label_cfg_path = review_queue_path.parent / cfg_fn
        if label_cfg_path.resolve() != cfg_path.resolve():
            label_cfg_path.write_text(xml_text, encoding="utf-8")

        artifacts["annotate_review_queue_labelstudio_json"] = str(rq_path.resolve())
        artifacts["annotate_labelstudio_config_xml"] = str(cfg_path.resolve())
        artifacts["annotate_labelstudio_import_json"] = str(import_path.resolve())
        artifacts["annotate_label_config_xml"] = str(label_cfg_path.resolve())
        hint_path = review_queue_path.parent / "README_LABELSTUDIO.txt"
        hint_path.write_text(
            "\n".join(
                [
                    "Label Studio HITL (optional)",
                    "",
                    f"1. Create a project; open Labeling Interface > Code and paste {cfg_fn} "
                    "(copy of labelstudio_config.xml; same XML).",
                    f"2. Import tasks from {task_fn} or review_queue_labelstudio.json (JSON import).",
                    "3. After labeling: Export → JSON, then apply back into the pipeline:",
                    "   python run_pipeline.py annotate review --run-id <id> --file path/to/export.json",
                    "   Optional: CSV with columns annotation_id, human_label next to review_queue.csv.",
                    "",
                    "Set label_studio.task_file / config_file in config.yaml to rename the canonical files.",
                    "annotation.labelstudio_strict_labels (default true): reject LS labels outside annotation.labels.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        artifacts["annotate_labelstudio_readme"] = str(hint_path.resolve())

    return AnnotateRunResult(
        auto_labeled_path=str(auto_labeled_path.resolve()),
        review_queue_path=str(review_queue_path.resolve()),
        report_path=str(report_path.resolve()),
        review_needed=bool(len(review_queue) > 0),
        review_count=int(len(review_queue)),
        artifact_paths=artifacts,
    )


def apply_annotate_review(
    *,
    config_payload: Dict[str, Any],
    stage_dir: Path,
    auto_labeled_df: pd.DataFrame,
    reviewed_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    cfg = dict(config_payload)
    ann_cfg = dict(cfg.get("annotation", {}))
    ann_cfg["project_root"] = str(stage_dir.resolve())
    cfg["annotation"] = ann_cfg
    agent = AnnotationAgent(config=cfg)
    merged = agent.merge_human_annotations(auto_labeled_df, reviewed_df, human_label_column="human_label")
    label_col = str(ann_cfg.get("label_column") or "label")
    merged[label_col] = merged["final_label"]

    final_path = stage_dir / "data" / "annotated_reviewed.parquet"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    export_df = merged.copy()
    export_df.attrs.clear()
    export_df.to_parquet(final_path, index=False)
    artifacts = dict(agent.last_artifacts)
    artifacts["annotate_final_reviewed"] = str(final_path.resolve())
    return merged, artifacts


def load_annotate_review_corrections(
    path: Path,
    config_payload: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load reviewer output: CSV/Parquet row table or Label Studio JSON export."""

    suf = path.suffix.lower()
    if suf == ".json":
        allowed = None
        if config_payload is not None:
            ann = dict(config_payload.get("annotation") or {})
            if ann.get("labelstudio_strict_labels", True):
                strict_agent = AnnotationAgent(config=config_payload)
                allowed = frozenset(strict_agent.task_config.labels)
        return read_labelstudio_export_path(path, allowed_labels=allowed, on_duplicate_id="error")
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported review corrections format: {path.suffix}")

