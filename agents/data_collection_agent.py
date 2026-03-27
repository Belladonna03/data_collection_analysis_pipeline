from __future__ import annotations

import inspect
import json
import math
import re
import traceback
import zlib
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.data_collection.connectors.api_connector import APIConnector
from agents.data_collection.connectors.base import BaseConnector
from agents.data_collection.connectors.github_connector import GitHubDataConnector
from agents.data_collection.connectors.hf_connector import HFDatasetConnector
from agents.data_collection.connectors.http_file_connector import HTTPFileConnector
from agents.data_collection.connectors.kaggle_connector import KaggleConnector
from agents.data_collection.connectors.scrape_connector import ScrapeConnector
from agents.data_collection.conversation import ConversationManager
from agents.data_collection.discovery import SourceDiscoveryService
from agents.data_collection.llm_factory import build_llm
from agents.data_collection.merge_budget import allocate_row_budgets, apply_budget_to_sample_size
from agents.data_collection.planner import StrategyPlanner
from agents.data_collection.registry import ConnectorRegistry, ConnectorNotRegisteredError
from agents.data_collection.schemas import (
    AutoScrapeResult,
    CollectionPlan,
    CollectionResult,
    SessionStatus,
    SourceSpec,
    SourceType,
    TopicProfile,
    ValidationReport,
)
from agents.data_collection.row_fingerprint import series_row_fingerprints
from agents.data_collection.session import CollectionSessionState, create_empty_session, update_status
from agents.data_collection.canonical_sample import (
    CanonicalNormalizationSummary,
    apply_chat_instruction_extraction,
    finalize_drop_empty_text,
    merge_canonical_summaries,
)
from agents.data_collection.text_unified_schema import (
    TEXT_PIPELINE_CONTRACT_COLUMNS,
    ensure_text_pipeline_contract_columns,
    fill_missing_text_from_title_body,
    reorder_dataframe_contract_first,
    scalar_cell_is_na,
    validate_merged_text_pipeline_contract,
)

# Metadata excluded from record_hash (hash captures content; provenance / synthetic ids excluded).
HASH_METADATA_COLUMNS: frozenset[str] = frozenset(
    {
        "source",
        "source_type",
        "source_url",
        "source_id",
        "collected_at",
        "record_hash",
        "id",
        "metadata",
    }
)

# Text pipeline contract first; then modality extras and internal columns (source-specific fields follow).
CORE_SCHEMA_COLUMNS: tuple[str, ...] = TEXT_PIPELINE_CONTRACT_COLUMNS + (
    "audio",
    "image",
    "source_id",
    "record_hash",
)

_PROVENANCE_COLUMNS = HASH_METADATA_COLUMNS  # backward-compatible name


def _resolve_source_url(source_spec: SourceSpec) -> str | None:
    """Best-effort canonical URL for provenance (no extra fields beyond existing spec)."""

    if source_spec.url:
        return source_spec.url
    if source_spec.repo_url:
        return source_spec.repo_url
    if source_spec.endpoint:
        return source_spec.endpoint
    if source_spec.type is SourceType.HF_DATASET:
        dataset_key = source_spec.dataset_id or source_spec.name
        if dataset_key:
            return f"https://huggingface.co/datasets/{dataset_key}"
    if source_spec.type is SourceType.KAGGLE:
        ref = source_spec.dataset_ref or source_spec.name
        if ref:
            return f"https://www.kaggle.com/datasets/{ref}"
    return None


# Note: "title" / "body" are contract columns, not aliases for standalone `text`;
# combined copy is produced by ``fill_missing_text_from_title_body`` after core mapping.
_TEXT_ALIASES: frozenset[str] = frozenset(
    {"prompt", "text", "content", "description", "sentence", "input", "utterance"}
)
_LABEL_ALIASES: frozenset[str] = frozenset(
    {
        "label",
        "y",
        "target",
        "labels",
        "winner_name",
        "human_label",
        "final_label",
        "auto_label",
        "class",
        "category",
    }
)
_AUDIO_ALIASES: frozenset[str] = frozenset({"audio", "audio_path", "audio_url", "speech", "wav"})
_IMAGE_ALIASES: frozenset[str] = frozenset({"image", "image_url", "img", "photo", "picture"})
_ID_ALIASES: frozenset[str] = frozenset(
    {
        "id",
        "_id",
        "uuid",
        "row_id",
        "pk",
        "record_id",
        "sample_id",
        "annotation_id",
        "thread_id",
    }
)


def _first_matching_column(df: pd.DataFrame, aliases: frozenset[str]) -> str | None:
    for col in df.columns:
        if str(col).strip().lower() in aliases:
            return str(col)
    return None


def _series_all_na(s: pd.Series) -> bool:
    if s.empty:
        return True
    return bool(s.isna().all())


class _DataNormalizer:
    """Small dataframe normalizer for MVP execution.

    After alias mapping, applies :func:`agents.data_collection.canonical_sample.apply_chat_instruction_extraction`
    so ``messages`` / ``instructions`` become ``text`` / ``target_text``, then title/body fill and drops rows
    with empty ``text``. See the README section on the merged dataset schema.
    """

    last_canonical_summary: CanonicalNormalizationSummary | None

    def __init__(self) -> None:
        self.last_canonical_summary = None

    def normalize(self, df: pd.DataFrame, source_spec: SourceSpec) -> pd.DataFrame:
        """Add text pipeline contract columns, canonical text extraction, metadata hash; keep extras."""

        self.last_canonical_summary = None
        normalized_df = df.copy()
        if "source" not in normalized_df.columns:
            normalized_df["source"] = source_spec.name
        if "source_type" not in normalized_df.columns:
            normalized_df["source_type"] = source_spec.type.value
        if "source_url" not in normalized_df.columns:
            normalized_df["source_url"] = _resolve_source_url(source_spec)
        if "source_id" not in normalized_df.columns:
            normalized_df["source_id"] = source_spec.id
        if "collected_at" not in normalized_df.columns:
            normalized_df["collected_at"] = datetime.now(timezone.utc).isoformat()

        self._ensure_explicit_title_body_columns(normalized_df)
        self._ensure_core_content_fields(normalized_df)
        normalized_df, canon_summary = apply_chat_instruction_extraction(
            normalized_df,
            source_key=str(source_spec.id),
        )
        self.last_canonical_summary = canon_summary
        normalized_df = fill_missing_text_from_title_body(normalized_df)
        normalized_df, _ = finalize_drop_empty_text(normalized_df, canon_summary)
        self._ensure_stable_row_ids(normalized_df, source_spec)
        self._ensure_metadata_json_column(normalized_df)
        self._ensure_metadata_nullable_defaults(normalized_df)
        normalized_df = ensure_text_pipeline_contract_columns(normalized_df)

        hash_input_cols = [col for col in normalized_df.columns if col not in HASH_METADATA_COLUMNS]
        if not normalized_df.empty and hash_input_cols:
            normalized_df["record_hash"] = self._row_hashes(normalized_df, hash_input_cols)
        elif "record_hash" not in normalized_df.columns:
            normalized_df["record_hash"] = pd.Series([pd.NA] * len(normalized_df), dtype=object)

        return reorder_dataframe_contract_first(normalized_df)

    @staticmethod
    def _ensure_metadata_nullable_defaults(df: pd.DataFrame) -> None:
        for col in ("source", "source_type", "source_url", "source_id", "collected_at"):
            if col not in df.columns:
                df[col] = pd.NA

    @staticmethod
    def _ensure_explicit_title_body_columns(df: pd.DataFrame) -> None:
        for col in ("title", "body"):
            if col not in df.columns:
                df[col] = pd.Series([pd.NA] * len(df), index=df.index, dtype=object)

    def _ensure_stable_row_ids(self, df: pd.DataFrame, source_spec: SourceSpec) -> None:
        """Guarantee contract `id` column: reuse upstream id or synthesize stable per-source ids."""

        if df.empty:
            if "id" not in df.columns:
                df["id"] = pd.Series(dtype=object)
            return
        if "id" in df.columns and not _series_all_na(df["id"]):
            df["id"] = df["id"].astype(str)
            return
        alias = _first_matching_column(df, _ID_ALIASES)
        if alias is not None and alias != "id":
            df["id"] = df[alias].astype(str)
            return
        prefix = str(source_spec.id).replace(":", "_").replace("/", "_").strip() or "src"
        df["id"] = pd.Series([f"{prefix}_{i}" for i in range(len(df))], index=df.index, dtype=str)

    @staticmethod
    def _ensure_metadata_json_column(df: pd.DataFrame) -> None:
        """UTF-8 JSON string per row (``\"{}\"`` when absent); safe for mixed-source concat and Parquet."""

        def _to_json_object_str(value: Any) -> str:
            if value is None:
                return "{}"
            if scalar_cell_is_na(value):
                return "{}"
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return "{}"
                try:
                    json.loads(text)
                    return text
                except json.JSONDecodeError:
                    return json.dumps({"raw": text}, ensure_ascii=False)
            if isinstance(value, dict):
                return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
            return json.dumps({"value": value}, ensure_ascii=False, default=str)

        if df.empty:
            if "metadata" not in df.columns:
                df["metadata"] = pd.Series(dtype="object")
            return
        if "metadata" not in df.columns:
            df["metadata"] = pd.Series(["{}"] * len(df), index=df.index, dtype=object)
        else:
            df["metadata"] = df["metadata"].map(_to_json_object_str)

    def _ensure_core_content_fields(self, df: pd.DataFrame) -> None:
        """Guarantee text/audio/image/label columns (nullable), with soft alias backfill."""

        def _fill_series(alias: str | None, empty_val: Any) -> pd.Series:
            if alias is None:
                return pd.Series(np.nan, index=df.index, dtype=object)
            return df[alias].copy()

        if "text" not in df.columns:
            alias = _first_matching_column(df, _TEXT_ALIASES)
            df["text"] = _fill_series(alias, np.nan)
        elif _series_all_na(df["text"]):
            alias = _first_matching_column(df, _TEXT_ALIASES)
            if alias is not None and alias != "text":
                df["text"] = df[alias].copy()

        if "label" not in df.columns:
            alias = _first_matching_column(df, _LABEL_ALIASES)
            df["label"] = _fill_series(alias, np.nan)
        elif _series_all_na(df["label"]):
            alias = _first_matching_column(df, _LABEL_ALIASES)
            if alias is not None and alias != "label":
                df["label"] = df[alias].copy()

        if "audio" not in df.columns:
            alias = _first_matching_column(df, _AUDIO_ALIASES)
            df["audio"] = _fill_series(alias, np.nan)
        elif _series_all_na(df["audio"]):
            alias = _first_matching_column(df, _AUDIO_ALIASES)
            if alias is not None and alias != "audio":
                df["audio"] = df[alias].copy()

        if "image" not in df.columns:
            alias = _first_matching_column(df, _IMAGE_ALIASES)
            df["image"] = _fill_series(alias, np.nan)
        elif _series_all_na(df["image"]):
            alias = _first_matching_column(df, _IMAGE_ALIASES)
            if alias is not None and alias != "image":
                df["image"] = df[alias].copy()

    @staticmethod
    def _row_hashes(df: pd.DataFrame, content_columns: list[str]) -> pd.Series:
        """SHA256 hex digest per row over sorted content columns (excludes provenance)."""

        return series_row_fingerprints(df, content_columns, tqdm_desc="record_hash")


class _DataValidator:
    """Small dataframe validator for MVP execution."""

    def validate(
        self,
        df: pd.DataFrame,
        expected_schema: dict[str, str] | None = None,
    ) -> ValidationReport:
        """Return a lightweight validation report."""

        expected_schema = expected_schema or {}
        missing_columns = [
            column_name for column_name in expected_schema if column_name not in df.columns
        ]

        null_stats: dict[str, float] = {}
        if not df.empty:
            for column_name in df.columns:
                null_stats[column_name] = float(df[column_name].isna().mean())

        warnings: list[str] = []
        if df.empty:
            warnings.append("Dataset is empty.")
        if missing_columns:
            warnings.append("Some expected columns are missing.")

        return ValidationReport(
            missing_columns=missing_columns,
            null_stats=null_stats,
            duplicate_count=0,
            warnings=warnings,
        )


def resolve_collection_sampling_cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge sampling keys from ``collection`` and ``collection.defaults`` (collection wins)."""

    if not config:
        return {}
    coll = dict(config.get("collection") or {})
    defaults = dict(coll.get("defaults") or {})
    keys = ("max_merged_rows", "stratify_column", "stratify_random_state")
    out: dict[str, Any] = {}
    for key in keys:
        if coll.get(key) is not None:
            out[key] = coll[key]
        elif defaults.get(key) is not None:
            out[key] = defaults[key]
    return out


def _stratum_sample_seed(random_state: int, label: str) -> int:
    h = zlib.crc32(f"{label}\0".encode("utf-8")) & 0xFFFFFFFF
    return int((random_state + h) & 0x7FFFFFFF) or 1


def apply_merged_row_cap(
    df: pd.DataFrame,
    sampling_cfg: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Cap merged rows using proportional quotas per stratify column (default ``source``)."""

    if df.empty or not sampling_cfg:
        return df, None
    raw_limit = sampling_cfg.get("max_merged_rows")
    if raw_limit is None:
        return df, None
    try:
        max_rows = int(raw_limit)
    except (TypeError, ValueError):
        return df, None
    if max_rows <= 0 or len(df) <= max_rows:
        return df, None

    try:
        rs = int(sampling_cfg.get("stratify_random_state", 42))
    except (TypeError, ValueError):
        rs = 42

    col_raw = sampling_cfg.get("stratify_column", "source")
    col = str(col_raw).strip() if col_raw else "source"

    if col not in df.columns:
        out = df.sample(n=max_rows, random_state=rs, replace=False).reset_index(drop=True)
        meta: dict[str, Any] = {
            "applied": True,
            "method": "simple_random",
            "reason": f"stratify_column {col!r} not in merged columns",
            "before_rows": int(len(df)),
            "after_rows": int(len(out)),
            "max_merged_rows": max_rows,
            "stratify_random_state": rs,
        }
        return reorder_dataframe_contract_first(out), meta

    strat_series = df[col].map(lambda x: "__na__" if scalar_cell_is_na(x) else str(x))
    vc = strat_series.value_counts()
    groups = list(vc.index)
    total = int(len(df))
    k = max_rows
    frac = {g: k * int(vc[g]) / total for g in groups}
    floors: dict[Any, int] = {g: min(int(vc[g]), int(math.floor(frac[g]))) for g in groups}
    deficit = k - sum(floors.values())
    frac_part = sorted(
        ((frac[g] - math.floor(frac[g]), g) for g in groups),
        key=lambda t: t[0],
        reverse=True,
    )
    for _fp, g in frac_part:
        if deficit <= 0:
            break
        if floors[g] < vc[g]:
            floors[g] += 1
            deficit -= 1
    while deficit > 0:
        candidates = [g for g in groups if floors[g] < vc[g]]
        if not candidates:
            break
        g = max(candidates, key=lambda x: int(vc[x]) - floors[x])
        floors[g] += 1
        deficit -= 1

    parts: list[pd.DataFrame] = []
    per_stratum: dict[str, dict[str, int]] = {}
    for g, n_take in floors.items():
        if n_take <= 0:
            continue
        sub = df.loc[strat_series == g]
        avail = int(len(sub))
        actual = min(n_take, avail)
        if actual <= 0:
            continue
        label = str(g)
        if actual >= avail:
            part = sub
        else:
            part = sub.sample(n=actual, random_state=_stratum_sample_seed(rs, label), replace=False)
        parts.append(part)
        per_stratum[label] = {"before": avail, "after": int(len(part))}

    if not parts:
        return df, None

    out = pd.concat(parts, ignore_index=True, sort=False)
    out = ensure_text_pipeline_contract_columns(out)
    out = fill_missing_text_from_title_body(out)
    out = reorder_dataframe_contract_first(out)
    meta = {
        "applied": True,
        "method": "stratified_proportional",
        "stratify_column": col,
        "before_rows": total,
        "after_rows": int(len(out)),
        "max_merged_rows": k,
        "per_stratum": per_stratum,
        "stratify_random_state": rs,
    }
    return out, meta


class _DataMerger:
    """Small dataframe merger for MVP execution."""

    last_merge_summary: dict[str, Any] | None

    def __init__(self) -> None:
        self.last_merge_summary = None

    def merge(self, sources: list[pd.DataFrame]) -> pd.DataFrame:
        """Align on pandas concat; each frame is already normalized to the text contract."""

        self.last_merge_summary = None
        if not sources:
            return pd.DataFrame()
        non_empty_sources = [source for source in sources if not source.empty]
        if not non_empty_sources:
            return pd.DataFrame()
        rows_per_source = [int(len(s)) for s in non_empty_sources]
        merged = pd.concat(non_empty_sources, ignore_index=True, sort=False)
        rows_after_concat = int(len(merged))
        merged = ensure_text_pipeline_contract_columns(merged)
        merged = fill_missing_text_from_title_body(merged)

        before_trim = int(len(merged))
        if not merged.empty and "text" in merged.columns:
            nonempty = merged["text"].notna() & (merged["text"].astype(str).str.strip().ne(""))
            merged = merged.loc[nonempty].reset_index(drop=True)
        merged_empty_text_dropped = before_trim - int(len(merged))

        before_dedup = int(len(merged))
        if not merged.empty:
            if (
                "record_hash" in merged.columns
                and merged["record_hash"].notna().any()
            ):
                merged = merged.drop_duplicates(subset=["record_hash"], keep="first").reset_index(
                    drop=True
                )
            else:
                dedup_cols = [c for c in ("text", "source", "source_id") if c in merged.columns]
                if dedup_cols:
                    merged = merged.drop_duplicates(subset=dedup_cols, keep="first").reset_index(
                        drop=True
                    )
        exact_duplicates_removed = before_dedup - int(len(merged))

        self.last_merge_summary = {
            "rows_per_source": rows_per_source,
            "rows_after_concat": rows_after_concat,
            "merged_empty_text_dropped": merged_empty_text_dropped,
            "exact_duplicates_removed": exact_duplicates_removed,
            "rows_final": int(len(merged)),
        }
        print(
            f"MERGE: sources={rows_per_source} concat={rows_after_concat} "
            f"empty_text_dropped={merged_empty_text_dropped} "
            f"dup_removed={exact_duplicates_removed} final={len(merged)}",
            flush=True,
        )
        return reorder_dataframe_contract_first(merged)


class _DatasetProfiler:
    """Small dataframe profiler for MVP execution."""

    def profile(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return basic shape and schema stats."""

        return {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": list(df.columns),
        }


class _InMemoryArtifactStorage:
    """File-backed artifact storage with an in-memory index."""

    def __init__(self, base_dir: str | Path = "artifacts/data_collection") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts: dict[str, Any] = {}
        self.current_run_dir: Path | None = None

    def start_run(self, run_name: str | None = None) -> Path:
        """Create and return a run directory."""

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_name = self._sanitize_name(run_name or "data_collection_run")
        self.current_run_dir = self.base_dir / f"{timestamp}_{safe_name}"
        self.current_run_dir.mkdir(parents=True, exist_ok=True)
        return self.current_run_dir

    def save_dataframe(self, name: str, df: pd.DataFrame) -> str:
        """Store a dataframe and return its saved path."""

        target_dir = self.current_run_dir or self.start_run()
        path = target_dir / f"{self._sanitize_name(name)}.csv"
        df.to_csv(path, index=False)
        path_str = str(path.resolve())
        self._artifacts[name] = {"type": "dataframe", "path": path_str}
        return path_str

    def save_object(self, name: str, value: Any) -> str:
        """Store an object and return its saved path."""

        target_dir = self.current_run_dir or self.start_run()
        path = target_dir / f"{self._sanitize_name(name)}.json"
        serializable_value = self._to_serializable(value)
        with path.open("w", encoding="utf-8") as file:
            json.dump(serializable_value, file, indent=2, ensure_ascii=False)
        path_str = str(path.resolve())
        self._artifacts[name] = {"type": "object", "path": path_str}
        return path_str

    def save_text(self, name: str, text: str, *, suffix: str = ".txt") -> str:
        """Store plain text (e.g. generated scraper code) and return its saved path."""

        target_dir = self.current_run_dir or self.start_run()
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        path = target_dir / f"{self._sanitize_name(name)}{safe_suffix}"
        path.write_text(text, encoding="utf-8")
        path_str = str(path.resolve())
        self._artifacts[name] = {"type": "text", "path": path_str}
        return path_str

    def get(self, name: str) -> Any:
        """Load stored artifact metadata by key."""

        return self._artifacts[name]

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Convert arbitrary artifact names into safe file names."""

        return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "artifact"

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        """Convert complex values into JSON-serializable structures."""

        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        if isinstance(value, dict):
            return {
                str(key): _InMemoryArtifactStorage._to_serializable(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [_InMemoryArtifactStorage._to_serializable(item) for item in value]
        return value


class DataCollectionAgent:
    """Facade that combines clarification, discovery, planning, and execution."""

    def __init__(
        self,
        config: str | dict[str, Any] | None = None,
        llm: Any | None = None,
    ) -> None:
        self.config = self._load_config(config)
        self.llm = llm if llm is not None else build_llm(self.config.get("llm", {}))
        self.session: CollectionSessionState = create_empty_session()
        self.conversation_manager = ConversationManager(self.session)
        self.registry = ConnectorRegistry()
        self.normalizer = _DataNormalizer()
        self.validator = _DataValidator()
        self.merger = _DataMerger()
        self.profiler = _DatasetProfiler()
        storage_dir = self.config.get("storage", {}).get(
            "artifacts_dir",
            "artifacts/data_collection",
        )
        self.storage = _InMemoryArtifactStorage(base_dir=storage_dir)
        discovery_cfg = dict(self.config.get("discovery", {}) or {})
        discovery_cfg["_connectors_scrape"] = (self.config.get("connectors") or {}).get("scrape") or {}
        if "scrape" in self.config:
            discovery_cfg["_top_level_scrape"] = self.config.get("scrape") or {}
        self.discovery_service = SourceDiscoveryService(
            config=discovery_cfg,
            llm=self.llm,
        )
        self.planner = StrategyPlanner()
        self._register_connectors()

    def chat_step(self, user_message: str) -> dict[str, Any]:
        """Process one user message in the clarification loop."""

        next_question = self.conversation_manager.handle_user_message(user_message)
        if next_question is not None:
            return {
                "status": self.session.status.value,
                "ready_for_discovery": False,
                "next_question": next_question,
                "topic_profile": self.session.topic_profile,
            }

        update_status(self.session, SessionStatus.DISCOVERING)
        return {
            "status": self.session.status.value,
            "ready_for_discovery": True,
            "message": "Topic profile is complete. Agent is ready to discover sources.",
            "topic_profile": self.session.topic_profile,
        }

    def discover_sources(self, topic_profile: TopicProfile) -> list:
        """Discover source candidates for a topic profile."""

        candidates = self.discovery_service.discover(topic_profile)
        self.session.candidates = candidates
        if getattr(self.discovery_service, "last_journal", None) is not None:
            self.session.artifacts["discovery_journal"] = self.storage.save_object(
                "discovery:journal",
                self.discovery_service.last_journal,
            )
        return candidates

    def execute_plan(self, plan: CollectionPlan) -> CollectionResult:
        """Execute a collection plan end to end."""

        try:
            update_status(self.session, SessionStatus.COLLECTING)
            normalized_frames: list[pd.DataFrame] = []
            canonical_summaries: list[CanonicalNormalizationSummary] = []
            per_source_stats: dict[str, dict[str, Any]] = {}
            artifacts: dict[str, str] = {}
            run_dir = self.storage.start_run(plan.topic_profile.topic or "data_collection")
            artifacts["run_dir"] = str(run_dir.resolve())

            ordered_sources = StrategyPlanner.order_source_specs_for_execution(plan.sources)
            sampling_cfg = resolve_collection_sampling_cfg(self.config)
            budget_map: dict[str, int] | None = None
            mr_raw = sampling_cfg.get("max_merged_rows")
            if mr_raw is not None:
                try:
                    mri = int(mr_raw)
                    if mri > 0:
                        strat_col = str(sampling_cfg.get("stratify_column") or "source")
                        budget_map = allocate_row_budgets(ordered_sources, mri, strat_col)
                        artifacts["collect_row_budgets"] = self.storage.save_object(
                            "collect:row_budgets",
                            {
                                "max_merged_rows": mri,
                                "stratify_column": strat_col,
                                "per_source_id": budget_map,
                            },
                        )
                        print(
                            f"COLLECT: max_merged_rows={mri} — per-source row budgets applied before fetch "
                            f"(stratify_column={strat_col!r}, {len(budget_map)} sources). "
                            "See artifact collect:row_budgets.",
                            flush=True,
                        )
                except (TypeError, ValueError):
                    budget_map = None

            for source_spec in ordered_sources:
                spec_for_collect = (
                    apply_budget_to_sample_size(source_spec, budget_map[source_spec.id])
                    if budget_map is not None
                    else source_spec
                )
                if (
                    source_spec.type is SourceType.SCRAPE
                    and source_spec.scraper_spec
                    and not source_spec.generated_code
                ):
                    try:
                        from agents.data_collection.scraper_codegen import generate_scraper_code
                        from agents.data_collection.scraper_spec import ScraperSpec
                        from agents.data_collection.scraper_spec_execution import (
                            source_spec_uses_structured_requests_html_scraper,
                        )

                        if source_spec_uses_structured_requests_html_scraper(source_spec):
                            # Human-readable script is saved as *_generated_scraper.py after collect (debug only).
                            pass
                        else:
                            merged_spec = dict(source_spec.scraper_spec or {})
                            if source_spec.url:
                                merged_spec.setdefault("entry_url", source_spec.url)
                            spec_obj = ScraperSpec.from_dict(merged_spec)
                            source_spec.generated_code = generate_scraper_code(spec_obj)
                    except Exception:
                        pass

                connector = self._create_connector(source_spec.type)
                raw_df = connector.collect(spec_for_collect)
                normalized_df = self.normalizer.normalize(raw_df, source_spec)
                if self.normalizer.last_canonical_summary is not None:
                    canonical_summaries.append(self.normalizer.last_canonical_summary)
                validation_report = self.validator.validate(
                    normalized_df,
                    expected_schema=plan.expected_schema,
                )

                normalized_frames.append(normalized_df)

                raw_key = self.storage.save_dataframe(
                    f"{source_spec.id}:raw",
                    raw_df,
                )
                normalized_key = self.storage.save_dataframe(
                    f"{source_spec.id}:normalized",
                    normalized_df,
                )
                validation_key = self.storage.save_object(
                    f"{source_spec.id}:validation",
                    validation_report,
                )

                artifacts[f"{source_spec.id}_raw"] = raw_key
                artifacts[f"{source_spec.id}_normalized"] = normalized_key
                artifacts[f"{source_spec.id}_validation"] = validation_key

                if source_spec.scraper_spec:
                    artifacts[f"{source_spec.id}_scraper_spec"] = self.storage.save_object(
                        f"{source_spec.id}:scraper_spec",
                        source_spec.scraper_spec,
                    )
                    if source_spec.type is SourceType.SCRAPE:
                        try:
                            import json as _json

                            prev_records = _json.loads(
                                raw_df.head(50).to_json(orient="records", date_format="iso")
                            )
                            artifacts[f"{source_spec.id}_preview_rows"] = self.storage.save_object(
                                f"{source_spec.id}:preview_rows",
                                prev_records,
                            )
                        except Exception:
                            pass
                        try:
                            from agents.data_collection.scraper_spec_execution import (
                                debug_snippet_for_spec,
                                source_spec_uses_structured_requests_html_scraper,
                            )

                            if source_spec_uses_structured_requests_html_scraper(source_spec):
                                sp = source_spec.scraper_spec if isinstance(source_spec.scraper_spec, dict) else {}
                                summary = {
                                    "extraction_mode": sp.get("extraction_mode"),
                                    "item_selector": sp.get("item_selector"),
                                    "fields": sp.get("fields"),
                                }
                                artifacts[f"{source_spec.id}_scraper_debug_snippet"] = (
                                    self.storage.save_text(
                                        f"{source_spec.id}:scraper_debug_snippet",
                                        debug_snippet_for_spec(summary),
                                        suffix=".txt",
                                    )
                                )
                                try:
                                    from agents.data_collection.scraper_codegen import generate_debug_scraper_py
                                    from agents.data_collection.scraper_spec import ScraperSpec

                                    merged_sp = dict(source_spec.scraper_spec or {})
                                    if source_spec.url:
                                        merged_sp.setdefault("entry_url", source_spec.url)
                                    spec_dbg = ScraperSpec.from_dict(merged_sp)
                                    dbg_py = generate_debug_scraper_py(
                                        spec_dbg,
                                        source_url=source_spec.url,
                                    )
                                    artifacts[f"{source_spec.id}_generated_scraper"] = (
                                        self.storage.save_text(
                                            f"{source_spec.id}:generated_scraper",
                                            dbg_py,
                                            suffix=".py",
                                        )
                                    )
                                except Exception:
                                    pass
                        except Exception:
                            pass
                if source_spec.generated_code:
                    code_suffix = ".py" if (source_spec.scraper_runtime or "").casefold() in {
                        "python",
                        "py",
                        "",
                    } else ".txt"
                    artifacts[f"{source_spec.id}_generated_code"] = self.storage.save_text(
                        f"{source_spec.id}:generated_code",
                        source_spec.generated_code,
                        suffix=code_suffix,
                    )

                per_source_stats[source_spec.id] = {
                    "rows": int(len(normalized_df)),
                    "columns": list(normalized_df.columns),
                    "validation_warnings": list(validation_report.warnings),
                }

            merged_df = self.merger.merge(normalized_frames)
            if canonical_summaries:
                summary_payload = merge_canonical_summaries(canonical_summaries)
                summary_payload["merge"] = self.merger.last_merge_summary
                artifacts["collect:canonical_summary"] = self.storage.save_object(
                    "collect:canonical_summary",
                    summary_payload,
                )
                print(
                    "COLLECT canonical drops (totals): "
                    f"{summary_payload['totals'].get('dropped', {})!r}",
                    flush=True,
                )
            merged_df, sampling_meta = apply_merged_row_cap(merged_df, sampling_cfg)
            if sampling_meta:
                if budget_map is not None:
                    sampling_meta = {
                        **sampling_meta,
                        "pre_collect_per_source_budgets": True,
                    }
                artifacts["merged_sampling"] = self.storage.save_object(
                    "merged:sampling",
                    sampling_meta,
                )

            merged_validation = self.validator.validate(
                merged_df,
                expected_schema=plan.expected_schema,
            )
            merged_validation.warnings.extend(validate_merged_text_pipeline_contract(merged_df))
            if budget_map is not None:
                merged_validation.warnings.append(
                    f"Per-source row budgets from max_merged_rows={sampling_cfg.get('max_merged_rows')} "
                    f"(stratify_column={sampling_cfg.get('stratify_column', 'source')}) applied before fetch. "
                    "See collect:row_budgets artifact."
                )
            if sampling_meta:
                merged_validation.warnings.append(
                    f"Merged dataset capped to {sampling_meta['after_rows']} rows "
                    f"({sampling_meta.get('method')}, stratify_column={sampling_meta.get('stratify_column', 'n/a')}). "
                    "See merged:sampling artifact."
                )
            merged_profile = self.profiler.profile(merged_df)

            artifacts["merged_dataframe"] = self.storage.save_dataframe(
                "merged:dataframe",
                merged_df,
            )
            artifacts["merged_validation"] = self.storage.save_object(
                "merged:validation",
                merged_validation,
            )
            artifacts["merged_profile"] = self.storage.save_object(
                "merged:profile",
                merged_profile,
            )

            self.session.artifacts.update(artifacts)
            update_status(self.session, SessionStatus.DONE)
            return CollectionResult(
                dataframe=merged_df,
                per_source_stats=per_source_stats,
                artifacts=artifacts,
                validation_report=merged_validation,
            )
        except Exception:
            update_status(self.session, SessionStatus.ERROR)
            raise

    def run_prepared_plan(self, plan: CollectionPlan) -> CollectionResult:
        """Resolve connector availability and execute *plan* without interactive clarification.

        Non-executable sources are dropped by :meth:`_make_executable_plan`; if none remain,
        raises ``ValueError`` (same as :meth:`interactive_run` execution path).
        """

        executable_plan = self._make_executable_plan(plan)
        self.session.selected_plan = executable_plan
        update_status(self.session, SessionStatus.AWAITING_APPROVAL)
        return self.execute_plan(executable_plan)

    def interactive_run(self, *, reuse_session_plans: bool = False) -> CollectionResult:
        """Run discovery, planning, and execution from current session state.

        When *reuse_session_plans* is True, reuse :attr:`session.proposed_plans` /
        :attr:`session.candidates` / a pre-set :attr:`session.selected_plan` instead of
        always re-running discovery (used by the unified ``collect`` CLI multi-step flow).
        """

        if not self.conversation_manager.is_ready_for_discovery():
            missing_fields = self.conversation_manager.get_missing_fields()
            raise ValueError(
                "Topic profile is incomplete. Missing fields: "
                + ", ".join(missing_fields)
            )

        topic_profile = self.session.topic_profile
        selected_plan = self.session.selected_plan
        if selected_plan is None:
            if reuse_session_plans and self.session.proposed_plans:
                selected_plan = self._select_best_plan(self.session.proposed_plans)
            elif reuse_session_plans and self.session.candidates:
                proposed_plans = self.planner.build_plans(topic_profile, self.session.candidates)
                self.session.proposed_plans = proposed_plans
                selected_plan = self._select_best_plan(proposed_plans)
            else:
                candidates = self.discover_sources(topic_profile)
                proposed_plans = self.planner.build_plans(topic_profile, candidates)
                self.session.proposed_plans = proposed_plans
                selected_plan = self._select_best_plan(proposed_plans)
        else:
            selected_plan = self._resolve_plan_for_execution(selected_plan)
        executable_plan = self._make_executable_plan(selected_plan)
        self.session.selected_plan = executable_plan
        update_status(self.session, SessionStatus.AWAITING_APPROVAL)

        result = self.execute_plan(executable_plan)
        return result

    def load_dataset(self, name: str, source: str = "hf") -> pd.DataFrame:
        """Thin wrapper around dataset connectors."""

        source_type = SourceType.HF_DATASET if source == "hf" else SourceType.KAGGLE
        source_spec = SourceSpec(
            id=f"{source_type.value}:{name}",
            type=source_type,
            name=name,
            dataset_id=name if source_type is SourceType.HF_DATASET else None,
            dataset_ref=name if source_type is SourceType.KAGGLE else None,
            enabled=True,
        )
        connector = self._create_connector(source_type)
        return connector.collect(source_spec)

    def fetch_api(
        self,
        endpoint: str,
        params: dict | None = None,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        response_path: str | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Thin wrapper around the API connector.

        *response_path* follows the same dot syntax as :class:`~agents.data_collection.schemas.SourceSpec`.
        *json_body* is sent as JSON (POST/PATCH/PUT) when set.
        """

        source_spec = SourceSpec(
            id="api:adhoc",
            type=SourceType.API,
            name=endpoint,
            endpoint=endpoint,
            method=method,
            headers=dict(headers or {}),
            params=dict(params or {}),
            response_path=response_path,
            json_body=json_body,
            enabled=True,
        )
        connector = self._create_connector(SourceType.API)
        return connector.collect(source_spec)

    def scrape(
        self,
        url: str,
        selector: str = "",
        *,
        content_mode: str = "html",
        response_path: str | None = None,
    ) -> pd.DataFrame:
        """Thin wrapper around the scrape connector.

        *content_mode* ``html`` (default) uses *selector* on fetched HTML.
        ``json`` / ``json_api`` / ``application_json`` treats *url* as a JSON document URL
        and uses *response_path* like :meth:`fetch_api`.
        """

        source_spec = SourceSpec(
            id="scrape:adhoc",
            type=SourceType.SCRAPE,
            name=url,
            url=url,
            selector=selector or None,
            scrape_content_mode=content_mode,
            response_path=response_path,
            enabled=True,
        )
        connector = self._create_connector(SourceType.SCRAPE)
        return connector.collect(source_spec)

    def auto_scrape(
        self,
        url: str,
        *,
        html: str | None = None,
        timeout: float = 20.0,
        max_pages: int | None = None,
        expected_schema: dict[str, str] | None = None,
    ) -> AutoScrapeResult:
        """Inspect *url* (or inline *html*), plan a :class:`ScraperSpec`, run generated code, normalize, validate.

        This is **opt-in** and separate from :meth:`scrape`, which keeps using the manual CSS selector path
        and :class:`ScrapeConnector` unchanged.

        On failure, returns ``success=False`` with a short ``error`` string and writes ``scraper_spec`` /
        ``generated_code`` / traceback summaries to the current artifact run directory when available.
        """

        from agents.data_collection.scraper_codegen import generate_scraper_code
        from agents.data_collection.scraper_planner import propose_scraper_spec
        from agents.data_collection.scraper_runner import run_generated_scraper

        run_dir = self.storage.start_run("auto_scrape")
        artifacts: dict[str, str] = {"auto_scrape_run_dir": str(run_dir.resolve())}

        scraper_spec_obj = None
        generated_code: str | None = None
        spec_dict: dict[str, Any] = {}

        try:
            mp = int(max_pages) if max_pages is not None else 3
            scraper_spec_obj = propose_scraper_spec(
                url,
                html=html,
                timeout=timeout,
                max_pages=mp,
            )
            spec_dict = scraper_spec_obj.to_dict()
            artifacts["auto_scrape_scraper_spec"] = self.storage.save_object(
                "auto_scrape:scraper_spec",
                spec_dict,
            )
        except Exception as exc:
            summary = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            artifacts["auto_scrape_failure"] = self.storage.save_text(
                "auto_scrape:failure_summary",
                summary,
            )
            self.session.artifacts.update(artifacts)
            return AutoScrapeResult(
                success=False,
                dataframe=pd.DataFrame(),
                error=str(exc),
                scraper_spec=spec_dict,
                artifacts=artifacts,
            )

        if not (scraper_spec_obj.item_selector or "").strip() or not scraper_spec_obj.fields:
            err = (
                "Inspection did not yield a usable item_selector or fields; "
                "cannot generate a scraper. Check layout_hints in the saved scraper_spec artifact."
            )
            artifacts["auto_scrape_failure"] = self.storage.save_text(
                "auto_scrape:failure_summary",
                err,
            )
            self.session.artifacts.update(artifacts)
            return AutoScrapeResult(
                success=False,
                dataframe=pd.DataFrame(),
                error=err,
                scraper_spec=spec_dict,
                artifacts=artifacts,
            )

        try:
            generated_code = generate_scraper_code(scraper_spec_obj)
            artifacts["auto_scrape_generated_code"] = self.storage.save_text(
                "auto_scrape:generated_code",
                generated_code,
                suffix=".py",
            )
            raw_df = run_generated_scraper(generated_code, timeout=timeout)
        except Exception as exc:
            summary = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            artifacts["auto_scrape_failure"] = self.storage.save_text(
                "auto_scrape:failure_summary",
                summary,
            )
            self.session.artifacts.update(artifacts)
            return AutoScrapeResult(
                success=False,
                dataframe=pd.DataFrame(),
                error=str(exc),
                scraper_spec=spec_dict,
                generated_code=generated_code,
                artifacts=artifacts,
            )

        source_spec = SourceSpec(
            id="scrape:auto",
            type=SourceType.SCRAPE,
            name=url,
            url=url,
            selector=scraper_spec_obj.item_selector,
            scraper_spec=spec_dict,
            generated_code=generated_code,
            scraper_runtime="python",
            enabled=True,
        )
        normalized_df = self.normalizer.normalize(raw_df, source_spec)
        validation_report = self.validator.validate(
            normalized_df,
            expected_schema=expected_schema,
        )

        artifacts["auto_scrape_raw"] = self.storage.save_dataframe("auto_scrape:raw", raw_df)
        artifacts["auto_scrape_normalized"] = self.storage.save_dataframe(
            "auto_scrape:normalized",
            normalized_df,
        )
        artifacts["auto_scrape_validation"] = self.storage.save_object(
            "auto_scrape:validation",
            validation_report,
        )
        self.session.artifacts.update(artifacts)

        return AutoScrapeResult(
            success=True,
            dataframe=normalized_df,
            validation_report=validation_report,
            scraper_spec=spec_dict,
            generated_code=generated_code,
            artifacts=artifacts,
        )

    def fetch_http_file(self, url: str) -> pd.DataFrame:
        """Thin wrapper around the HTTP file connector."""

        source_spec = SourceSpec(
            id="http_file:adhoc",
            type=SourceType.HTTP_FILE,
            name=url,
            url=url,
            enabled=True,
        )
        connector = self._create_connector(SourceType.HTTP_FILE)
        return connector.collect(source_spec)

    def fetch_github_dataset(self, repo_url: str, file_patterns: list[str] | None = None) -> pd.DataFrame:
        """Thin wrapper around the GitHub dataset connector."""

        source_spec = SourceSpec(
            id="github_dataset:adhoc",
            type=SourceType.GITHUB_DATASET,
            name=repo_url,
            repo_url=repo_url,
            file_patterns=file_patterns or [],
            enabled=True,
        )
        connector = self._create_connector(SourceType.GITHUB_DATASET)
        return connector.collect(source_spec)

    def merge(self, sources: list[pd.DataFrame]) -> pd.DataFrame:
        """Normalize each raw frame to the text contract, then merge."""

        normalized_frames: list[pd.DataFrame] = []
        for i, frame in enumerate(sources):
            if frame is None or getattr(frame, "empty", False):
                continue
            spec = SourceSpec(
                id=f"merge_input:{i}",
                type=SourceType.HTTP_FILE,
                name=f"merge_input_{i}",
                enabled=True,
            )
            normalized_frames.append(self.normalizer.normalize(frame, spec))
        return self.merger.merge(normalized_frames)

    def reset_session(self) -> CollectionSessionState:
        """Reset the current conversation and execution session."""

        self.session = create_empty_session()
        self.conversation_manager = ConversationManager(self.session)
        return self.session

    def _register_connectors(self) -> None:
        """Register available connectors."""

        self.registry.register(SourceType.HF_DATASET, HFDatasetConnector)
        self.registry.register(SourceType.KAGGLE, KaggleConnector)
        self.registry.register(SourceType.GITHUB_DATASET, GitHubDataConnector)
        self.registry.register(SourceType.HTTP_FILE, HTTPFileConnector)
        self.registry.register(SourceType.API, APIConnector)
        self.registry.register(SourceType.SCRAPE, ScrapeConnector)

    def _create_connector(self, source_type: SourceType | str) -> BaseConnector:
        """Create a connector instance with config-driven kwargs."""

        connector_cls = self.registry.get(source_type)
        connector_kwargs = self._filter_connector_kwargs(
            connector_cls,
            self._get_connector_kwargs(source_type),
        )
        return connector_cls(**connector_kwargs)

    def _get_connector_kwargs(self, source_type: SourceType | str) -> dict[str, Any]:
        """Return connector init kwargs from config."""

        normalized_type = (
            source_type if isinstance(source_type, SourceType) else SourceType(source_type)
        )
        connectors_config = self.config.get("connectors", {})
        return dict(connectors_config.get(normalized_type.value, {}))

    @staticmethod
    def _filter_connector_kwargs(
        connector_cls: type[BaseConnector],
        connector_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep only kwargs accepted by a connector constructor."""

        signature = inspect.signature(connector_cls.__init__)
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return connector_kwargs

        allowed_names = {
            name
            for name, parameter in signature.parameters.items()
            if name != "self"
            and parameter.kind
            in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        }
        return {
            key: value for key, value in connector_kwargs.items() if key in allowed_names
        }

    def _select_best_plan(self, plans: list[CollectionPlan]) -> CollectionPlan:
        """Select the best available plan."""

        if not plans:
            raise ValueError("Planner did not produce any collection plans.")

        ranked_plans = sorted(
            plans,
            key=lambda plan: (
                self._unsupported_source_count(plan),
                -self._executable_source_count(plan),
                len(plan.warnings),
                -len(plan.sources),
            ),
        )
        return ranked_plans[0]

    def _make_executable_plan(self, plan: CollectionPlan) -> CollectionPlan:
        """Drop sources without executable connectors and annotate warnings."""

        executable_sources: list[SourceSpec] = []
        skipped_messages: list[str] = []
        for source in plan.sources:
            is_executable, reason = self._get_source_execution_status(source)
            if is_executable:
                executable_sources.append(source)
            else:
                skipped_messages.append(self._format_non_executable_source(source, reason))

        warnings = list(plan.warnings)
        if skipped_messages:
            warnings.extend(skipped_messages)
        if not executable_sources:
            raise ValueError(
                "Plan contains no executable sources that can be collected with the current configuration."
            )

        return replace(plan, sources=executable_sources, warnings=warnings)

    def _resolve_plan_for_execution(self, selected_plan: CollectionPlan) -> CollectionPlan:
        """Prefer a runnable plan when the selected one is not executable."""

        if self._executable_source_count(selected_plan) > 0:
            return selected_plan

        if not self.session.proposed_plans:
            return selected_plan

        fallback_plan = self._select_best_plan(self.session.proposed_plans)
        if fallback_plan is selected_plan:
            return selected_plan

        fallback_warnings = list(fallback_plan.warnings)
        fallback_warnings.append(
            "Switched execution to a different runnable plan because the selected plan had no executable sources."
        )
        return replace(fallback_plan, warnings=fallback_warnings)

    def _is_source_executable(self, source: SourceSpec) -> bool:
        """Check whether a source type can be executed now."""

        is_executable, _ = self._get_source_execution_status(source)
        return is_executable

    def _get_source_execution_status(self, source: SourceSpec) -> tuple[bool, str | None]:
        """Return runtime executability for one source spec."""

        if not source.is_executable:
            return False, source.non_executable_reason or "Source is marked as discovery-only."
        if source.type is SourceType.REPOSITORY:
            return False, source.non_executable_reason or "Repository candidates require a dataset-capable connector."
        if source.type is SourceType.API and not source.endpoint:
            return False, "API source spec requires a non-empty 'endpoint'."
        try:
            connector = self._create_connector(source.type)
        except ConnectorNotRegisteredError:
            return False, f"No connector is registered for source type '{source.type.value}'."
        return connector.can_execute(source)

    def _unsupported_source_count(self, plan: CollectionPlan) -> int:
        """Count unsupported sources in a plan."""

        return sum(0 if self._is_source_executable(source) else 1 for source in plan.sources)

    @staticmethod
    def _format_non_executable_source(source: SourceSpec, reason: str | None) -> str:
        """Build a readable warning for a skipped discovery-only source."""

        reason = reason or source.non_executable_reason or "No executable connector is available for this source yet."
        return (
            f"Skipped non-executable source `{source.name}` "
            f"({source.type.value}): {reason}"
        )

    def _executable_source_count(self, plan: CollectionPlan) -> int:
        """Count executable sources in a plan."""

        return sum(1 for source in plan.sources if self._is_source_executable(source))

    @staticmethod
    def _load_config(config: str | dict[str, Any] | None) -> dict[str, Any]:
        """Load config from dict, JSON path, or YAML path."""

        if config is None:
            return {}
        if isinstance(config, dict):
            return dict(config)

        config_path = Path(config)
        suffix = config_path.suffix.lower()
        with config_path.open("r", encoding="utf-8") as file:
            if suffix in {".yaml", ".yml"}:
                try:
                    import yaml
                except ImportError as exc:
                    raise ImportError(
                        "YAML config requires the 'pyyaml' package. "
                        "Install it with: pip install pyyaml"
                    ) from exc
                return yaml.safe_load(file) or {}
            return json.load(file)


# Example:
# agent = DataCollectionAgent()
# agent.chat_step("movie reviews")
# agent.chat_step("text")
# agent.chat_step("english")
# agent.chat_step("classification")
# agent.chat_step("1000")
# agent.chat_step("yes")
# result = agent.interactive_run()
