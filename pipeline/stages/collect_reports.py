"""Post-run collect artifacts: stable paths, data card, EDA, source summary, terminal summary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agents.data_collection.schemas import CollectionPlan, CollectionResult, TopicProfile
from agents.data_collection_agent import CORE_SCHEMA_COLUMNS

from pipeline.collect_snapshots import DiscoverySnapshot, UserSelection
from pipeline.stages.collect_eda import generate_collect_eda


@dataclass
class CollectFinalizeResult:
    artifact_paths: Dict[str, str]
    terminal_summary: str
    merged_parquet_path: str
    row_count: int


def ensure_core_schema_on_merged(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure merged output has every core unified column (nullable if missing after concat)."""

    out = df.copy()
    for col in CORE_SCHEMA_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def _scalar_is_non_bool_real(x: Any) -> bool:
    """True for numeric scalars suitable for ``to_numeric``, excluding bool."""
    if isinstance(x, bool):
        return False
    return isinstance(x, Real)


def _is_text_scalar(x: Any) -> bool:
    """``str`` or NumPy string scalar (``isinstance(x, str)`` is False for ``np.str_``)."""
    if isinstance(x, str):
        return True
    return isinstance(x, np.str_)


def _object_series_with_bytes_decode(s: pd.Series) -> pd.Series:
    non_na = s.dropna()
    if non_na.empty or not any(isinstance(x, (bytes, bytearray)) for x in non_na):
        return s
    return s.map(lambda v: v.decode("utf-8", errors="replace") if isinstance(v, (bytes, bytearray)) else v)


def coerce_merged_frame_for_parquet_export(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize object dtypes that pyarrow cannot write (mixed datetimes, mixed number/str, bytes)."""

    out = df.copy()
    for col in out.columns:
        s = out[col]
        if not pd.api.types.is_object_dtype(s):
            continue
        non_na = s.dropna()
        if non_na.empty:
            continue

        s_work = _object_series_with_bytes_decode(s)
        non_na2 = s_work.dropna()
        if non_na2.empty:
            continue

        # Timestamps may appear only after many plain-string rows.
        if any(isinstance(x, (pd.Timestamp, datetime, date)) for x in non_na2):
            out[col] = pd.to_datetime(s_work, errors="coerce", utc=False)
            continue

        # PyArrow cannot write object columns that mix real scalars with str; some cells may be
        # odds sentinels ("NR"), typos ("2.,3"), etc. Coerce parses numbers and maps the rest to NaN.
        has_real_and_text = (
            any(_scalar_is_non_bool_real(x) for x in non_na2)
            and any(_is_text_scalar(x) for x in non_na2)
        )
        if has_real_and_text:
            out[col] = pd.to_numeric(s_work, errors="coerce")

    return out


def _build_source_summary_json(
    result: CollectionResult,
    plan: Optional[CollectionPlan],
    selection: Optional[UserSelection],
) -> Dict[str, Any]:
    per_source: List[Dict[str, Any]] = []
    for sid, stats in result.per_source_stats.items():
        per_source.append(
            {
                "source_id": sid,
                "row_count": stats.get("rows"),
                "columns": stats.get("columns", []),
                "validation_warnings": stats.get("validation_warnings", []),
            }
        )
    return {
        "per_source": per_source,
        "plan_rationale": plan.rationale if plan else None,
        "plan_warnings": list(plan.warnings) if plan else [],
        "user_selection": (
            {
                "candidate_numbers": selection.candidate_numbers,
                "candidate_keys": selection.candidate_keys,
                "entries": selection.entries,
            }
            if selection
            else None
        ),
    }


def _build_data_card_md(
    *,
    topic_profile: TopicProfile,
    df: pd.DataFrame,
    merged_parquet: Path,
    source_summary_path: Path,
    eda_summary_path: str,
    plots_dir: Path,
    plot_paths: List[str],
    selection: Optional[UserSelection],
    plan: Optional[CollectionPlan],
    discovery: Optional[DiscoverySnapshot],
    eda_skipped: List[str],
) -> str:
    lines: List[str] = [
        "# Collect stage — data card\n",
        "## Collection request",
        f"- **Topic:** {topic_profile.topic or '—'}",
        f"- **Modality:** {topic_profile.modality}",
        f"- **Language:** {topic_profile.language}",
        f"- **Task type:** {topic_profile.task_type}",
        f"- **Size target:** {topic_profile.size_target}",
        f"- **Needs labels:** {topic_profile.needs_labels}",
        "",
        "## Selected sources",
    ]
    if selection and selection.entries:
        for e in selection.entries:
            lines.append(
                f"- **#{e.get('candidate_number')}** `{e.get('candidate_key')}` — {e.get('name')} "
                f"({e.get('source_type')})"
            )
    else:
        lines.append("- *(legacy / interactive run — no structured selection snapshot)*")

    if discovery is not None:
        lines.extend(
            [
                "",
                "## Discovery snapshot",
                f"- **Candidates discovered:** {len(discovery.ordered)}",
                f"- **Snapshot time:** {discovery.created_at or '—'}",
            ]
        )

    lines.extend(
        [
            "",
            "## Plan context",
            f"- **Rationale:** {plan.rationale if plan else '—'}",
        ]
    )
    if plan and plan.warnings:
        lines.append("- **Planner warnings:**")
        for w in plan.warnings:
            lines.append(f"  - {w}")
    lines.extend(
        [
            "",
            "## Merged dataset",
            f"- **Rows:** {len(df)}",
            f"- **Columns:** {len(df.columns)}",
            f"- **Parquet:** `{merged_parquet.resolve()}`",
            "",
            "## Core unified schema (required columns)",
        ]
    )
    for c in CORE_SCHEMA_COLUMNS:
        present = c in df.columns
        nn = int(df[c].notna().sum()) if present else 0
        lines.append(f"- `{c}`: {'present' if present else '**MISSING**'}, non-null={nn}")

    core_names = set(CORE_SCHEMA_COLUMNS)
    notable = [c for c in df.columns if str(c) not in core_names][:40]
    lines.extend(["", "## Notable source-specific columns (sample)", ", ".join(f"`{c}`" for c in notable) + (" …" if len(df.columns) > len(notable) + len(core_names) else "")])

    lines.extend(
        [
            "",
            "## Risks & limitations",
            "- Mixed schemas across sources: extra columns are preserved; downstream stages should validate.",
            "- Core `text` / `label` may be sparse for purely tabular sports tables unless aliases matched.",
            "",
            "## EDA & lineage artifacts",
            f"- **Source summary:** `{source_summary_path.resolve()}`",
            f"- **EDA report:** `{eda_summary_path}`",
            f"- **Plots directory:** `{plots_dir.resolve()}`",
        ]
    )
    if plot_paths:
        lines.append("- **Plot files:**")
        for p in plot_paths:
            lines.append(f"  - `{p}`")

    if eda_skipped:
        lines.extend(["", "## EDA items skipped / not applicable"])
        for s in eda_skipped:
            lines.append(f"- {s}")

    return "\n".join(lines) + "\n"


def _terminal_summary_lines(
    *,
    n_sources: int,
    n_rows: int,
    n_cols: int,
    artifacts: Dict[str, str],
) -> str:
    core_preview = ", ".join(CORE_SCHEMA_COLUMNS[:6]) + ", …"
    lines = [
        "",
        "Collection complete.",
        f"Selected sources (with data): {n_sources}",
        f"Merged rows: {n_rows}",
        f"Merged columns: {n_cols}",
        f"Core schema (all present): {core_preview}",
        "Artifacts:",
        f"  - merged dataset: {artifacts.get('collect_merged_output', '')}",
        f"  - data card: {artifacts.get('collect_data_card_md', '')}",
        f"  - EDA summary: {artifacts.get('collect_eda_summary_md', '')}",
        f"  - source summary: {artifacts.get('collect_source_summary_json', '')}",
        f"  - plots dir: {artifacts.get('collect_eda_plots_dir', '')}",
        "Next action:",
        "  python run_pipeline.py quality run",
        "",
    ]
    return "\n".join(lines)


def finalize_collect_stage_artifacts(
    *,
    stage_dir: Path,
    result: CollectionResult,
    topic_profile: TopicProfile,
    plan: Optional[CollectionPlan] = None,
    selection: Optional[UserSelection] = None,
    discovery: Optional[DiscoverySnapshot] = None,
) -> CollectFinalizeResult:
    """Write canonical collect outputs under ``stage_dir`` and return paths + terminal summary text."""

    if result.dataframe is None or result.dataframe.empty:
        raise ValueError("Collect produced an empty merged dataframe.")

    if len(result.per_source_stats) < 2:
        raise ValueError(
            "Collect stage requires at least 2 sources with collected rows for this pipeline. "
            f"Got {len(result.per_source_stats)} source(s). Re-run `collect select` with at least two "
            "executable sources, or use legacy collect only if teaching demo allows a single source."
        )

    df = ensure_core_schema_on_merged(result.dataframe)
    df = coerce_merged_frame_for_parquet_export(df)
    data_dir = stage_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    merged_parquet = data_dir / "merged_dataset.parquet"
    df.to_parquet(merged_parquet, index=False)

    reports_dir = stage_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = reports_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    src_payload = _build_source_summary_json(result, plan, selection)
    src_json = reports_dir / "source_summary.json"
    src_json.write_text(json.dumps(src_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    eda_info = generate_collect_eda(df, reports_dir)
    eda_skipped = list(eda_info.get("skipped_notes", []))

    data_card_body = _build_data_card_md(
        topic_profile=topic_profile,
        df=df,
        merged_parquet=merged_parquet,
        source_summary_path=src_json,
        eda_summary_path=eda_info["eda_summary_path"],
        plots_dir=plots_dir,
        plot_paths=list(eda_info.get("plot_paths", [])),
        selection=selection,
        plan=plan,
        discovery=discovery,
        eda_skipped=eda_skipped,
    )
    data_card_path = reports_dir / "data_card.md"
    data_card_path.write_text(data_card_body, encoding="utf-8")

    artifacts: Dict[str, str] = {
        "collect_merged_output": str(merged_parquet.resolve()),
        "collect_merged_dataset_parquet": str(merged_parquet.resolve()),
        "collect_data_card_md": str(data_card_path.resolve()),
        "collect_eda_summary_md": eda_info["eda_summary_path"],
        "collect_source_summary_json": str(src_json.resolve()),
        "collect_eda_plots_dir": str(plots_dir.resolve()),
    }
    for i, p in enumerate(eda_info.get("plot_paths", [])):
        artifacts[f"collect_eda_plot_{i}"] = p

    term = _terminal_summary_lines(
        n_sources=len(result.per_source_stats),
        n_rows=len(df),
        n_cols=len(df.columns),
        artifacts=artifacts,
    )

    return CollectFinalizeResult(
        artifact_paths=artifacts,
        terminal_summary=term,
        merged_parquet_path=str(merged_parquet.resolve()),
        row_count=len(df),
    )
