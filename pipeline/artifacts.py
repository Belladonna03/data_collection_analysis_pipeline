from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from pipeline.registry import STAGE_BY_SHORT_NAME
from pipeline.state import PipelineState


def resolve_default_config_path(cwd: Optional[Path] = None) -> Optional[Path]:
    """If no path was passed, use ``config.yaml`` or ``config.yml`` in *cwd* when present."""

    root = cwd or Path.cwd()
    for name in ("config.yaml", "config.yml"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        resolved = resolve_default_config_path()
        if resolved is None:
            return {}
        path = resolved
    else:
        path = Path(config_path)
    with path.open("r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            return yaml.safe_load(fh) or {}
        return json.load(fh)


def read_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}. Use .csv or .parquet")


def build_quality_config(config_path: Optional[str], run_dir: Path) -> Dict[str, Any]:
    cfg = load_config(config_path)
    quality = dict(cfg.get("quality", {}))
    quality.setdefault("dataset_mode", "auto")
    quality.setdefault("include_text_derived_outliers", "auto")
    quality.setdefault("text_column_min_non_null_ratio", 0.1)
    quality.setdefault("text_column_min_avg_length", 2.0)
    quality.setdefault("review_before_apply", True)
    quality.setdefault("generate_preview_outputs_before_review", False)
    quality.setdefault("row_actions_max_total", 50_000)
    quality["project_root"] = str(run_dir.resolve())
    storage = dict(quality.get("storage", {}))
    storage.setdefault("reports_dir", "02_quality/reports")
    storage.setdefault("interim_dir", "02_quality/data")
    storage.setdefault("review_dir", "02_quality/review")
    quality["storage"] = storage
    cfg["quality"] = quality
    return cfg


def resolve_collect_output_path(state: PipelineState) -> Optional[str]:
    collect_stage_id = STAGE_BY_SHORT_NAME["collect"].stage_id
    for stage in state.stages:
        if stage.stage_id != collect_stage_id:
            continue
        for key in ("collect_merged_dataset_parquet", "collect_merged_output", "merged_dataframe", "stage_output"):
            if key in stage.artifacts:
                return stage.artifacts[key]
    return None


def resolve_quality_input_path(state: PipelineState) -> Optional[str]:
    quality_stage_id = STAGE_BY_SHORT_NAME["quality"].stage_id
    for stage in state.stages:
        if stage.stage_id == quality_stage_id and stage.artifacts.get("quality_input_dataset"):
            return stage.artifacts["quality_input_dataset"]
    return resolve_collect_output_path(state)


def resolve_quality_output_path(state: PipelineState) -> Optional[str]:
    quality_stage_id = STAGE_BY_SHORT_NAME["quality"].stage_id
    for stage in state.stages:
        if stage.stage_id != quality_stage_id:
            continue
        for key in ("stage_cleaned_output", "cleaned_final", "cleaned_final_parquet"):
            if key in stage.artifacts:
                return stage.artifacts[key]
    return None


def resolve_annotate_output_path(state: PipelineState) -> Optional[str]:
    annotate_stage_id = STAGE_BY_SHORT_NAME["annotate"].stage_id
    for stage in state.stages:
        if stage.stage_id != annotate_stage_id:
            continue
        for key in ("annotate_final_reviewed", "annotate_auto_labeled", "annotate_skipped_passthrough"):
            if key in stage.artifacts:
                return stage.artifacts[key]
    return None


def resolve_train_input_path(state: PipelineState) -> Optional[str]:
    al_stage_id = STAGE_BY_SHORT_NAME["al"].stage_id
    for stage in state.stages:
        if stage.stage_id == al_stage_id:
            for key in ("al_final_dataset", "al_input_dataset"):
                if key in stage.artifacts:
                    return stage.artifacts[key]
    return resolve_annotate_output_path(state)

