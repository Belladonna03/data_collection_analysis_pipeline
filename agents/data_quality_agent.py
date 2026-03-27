from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from agents.data_collection.llm_factory import build_llm
from agents.data_collection.row_fingerprint import safe_duplicated, series_row_fingerprints
from agents.data_quality.schemas import ComparisonReport, QualityReport, QualityStageResult
from agents.data_quality.scalar_norm import (
    coerce_quality_row_id,
    config_column_name,
    iter_config_column_names,
    normalize_dataframe_object_cells,
    normalize_scalar_like,
)
from agents.data_quality.text_checks import (
    language_mismatch_indices,
    near_duplicate_drop_indices,
    pii_breakdown_counts,
    pii_hit_mask,
    redact_pii,
)


DEFAULT_LABEL_HINTS = ("label", "target", "class", "category", "sentiment")
_INTERNAL_QUALITY_COLUMNS = frozenset({"__quality_row_id", "__quality_original_index"})

# Optional top-level keys in ``fix(..., strategy)`` merged into ``strategy['text_quality']`` then removed.
TEXT_QUALITY_STRATEGY_ALIAS_KEYS: frozenset[str] = frozenset(
    {
        "redact_basic_pii",
        "drop_empty_text",
        "drop_short_text",
        "drop_near_duplicates",
        "language_filter",
    }
)


class HumanReviewRequired(Exception):
    """Raised when the quality stage stops for human approval (HITL gate)."""

    def __init__(self, message: str, stage_result: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.stage_result = stage_result or {}




def _duplicate_masks_for_frame(work: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Two ``DataFrame.duplicated`` masks with at most one fingerprint pass for unhashable rows."""

    if work.empty:
        empty = pd.Series(dtype=bool, index=work.index)
        return empty, empty
    try:
        return work.duplicated(keep="first"), work.duplicated(keep=False)
    except TypeError:
        sigs = series_row_fingerprints(work, list(work.columns), tqdm_desc="quality_dup")
        return sigs.duplicated(keep="first"), sigs.duplicated(keep=False)


def _normalize_language_expected(raw: Any) -> str | None:
    """Coerce config / merged ``language.expected`` to a single locale code.

    Avoids ``if expected`` truth tests on numpy arrays (ambiguous bool error).
    """

    if raw is None or raw is False:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        return s or None
    if isinstance(raw, (list, tuple)):
        if not raw:
            return None
        return _normalize_language_expected(raw[0])
    try:
        import numpy as np

        if isinstance(raw, np.ndarray):
            if raw.size == 0:
                return None
            flat0 = raw.reshape(-1)[0]
            return _normalize_language_expected(flat0)
        if isinstance(raw, np.generic):
            return _normalize_language_expected(raw.item())
    except ImportError:
        pass
    if isinstance(raw, pd.Series):
        if raw.empty:
            return None
        return _normalize_language_expected(raw.iloc[0])
    s = str(raw).strip()
    return s or None


DEFAULT_TEXT_DUPLICATE_HINTS = (
    "prompt",
    "text",
    "instruction",
    "question",
    "input",
    "output",
    "response",
    "content",
)

# Tabular-friendly keys first; text placeholders (e.g. empty ``text``) must not win over these.
_PREFERRED_DUPLICATE_KEY_COLUMNS = (
    "record_hash",
    "source_id",
    "id",
    "row_id",
    "sample_id",
)


class _QualityArtifactStorage:
    """Small file-backed storage for quality-stage artifacts."""

    def __init__(self, project_root: Path, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.project_root = project_root
        self.reports_dir = self.project_root / config.get("reports_dir", "reports/quality")
        self.plots_dir = self.reports_dir / "plots"
        self.interim_dir = self.project_root / config.get("interim_dir", "data/interim")
        self.review_dir = self.project_root / config.get("review_dir", "review")

    def save_json(self, relative_path: str | Path, payload: Any) -> str:
        """Save JSON payload under the project root."""

        path = self.project_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        return str(path.resolve())

    def save_markdown(self, relative_path: str | Path, content: str) -> str:
        """Save markdown payload under the project root."""

        path = self.project_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path.resolve())

    def save_dataframe(self, relative_path: str | Path, df: pd.DataFrame) -> str:
        """Save dataframe as parquet under the project root."""

        path = self.project_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        export_df = df.copy()
        export_df.attrs.clear()
        export_df.to_parquet(path, index=False)
        return str(path.resolve())

    def save_figure(self, relative_path: str | Path, fig: Any) -> str:
        """Persist a matplotlib figure under the project root."""

        path = self.project_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path.resolve())


class DataQualityAgent:
    """
    Reusable data-quality stage for text-heavy or tabular pipeline datasets.

    ``dataset_mode`` / meaningful-text gating avoid empty ``text`` / ``prompt`` columns driving
    duplicate keys or expensive char/word-length outlier features on tabular data.

    Core API: `detect_issues`, `fix`, `compare`, `build_strategy_recommendations`, `prepare_review_bundle`,
    `apply_review_decision`, `explain_issues_and_recommend_strategy`.

    Pipeline / HITL: use `run_stage` to block until `approved=true`, then re-run with the
    same raw dataframe and the decision dict. Internal row ids and numeric coercion metadata
    are recorded in `last_stage_prep_meta` and final review artifacts under ``review/``.
    LLM usage is lazy (only for explanations) and never required for cleaning.
    """

    def __init__(
        self,
        config: str | dict[str, Any] | None = None,
        llm: Any | None = None,
    ) -> None:
        self.config = self._load_config(config)
        self.quality_config = dict(self.config.get("quality", {}))
        self.project_root = Path(
            self.quality_config.get("project_root")
            or Path(__file__).resolve().parents[1]
        )
        self.storage = _QualityArtifactStorage(
            project_root=self.project_root,
            config=self.quality_config.get("storage", {}),
        )
        self._llm_explicit = llm
        self._llm_lazy: Any | None = None
        self.last_fix_audit: list[dict[str, Any]] = []
        self.last_fix_row_actions: list[dict[str, Any]] = []
        self.last_artifacts: dict[str, str] = {}
        self.last_stage_prep_meta: dict[str, Any] = {}

    def _get_llm(self) -> Any | None:
        """Resolve LLM lazily for explain-only paths; never required for detect/fix/compare."""

        if self._llm_explicit is not None:
            return self._llm_explicit
        if self.quality_config.get("enable_llm_explanations") is False:
            return None
        llm_cfg = self.config.get("llm") or {}
        if llm_cfg.get("enabled") is False:
            return None
        if self._llm_lazy is None:
            try:
                self._llm_lazy = build_llm(llm_cfg)
            except Exception:
                # LLM is a bonus capability; never block core data-quality flow.
                self._llm_lazy = None
        return self._llm_lazy

    def _row_action_id_limit(self) -> int:
        return int(self.quality_config.get("row_action_id_list_max", 10_000))

    def _dataset_mode(self) -> str:
        return str(self.quality_config.get("dataset_mode", "auto")).strip().lower() or "auto"

    def _is_meaningful_text_column(self, series: pd.Series) -> bool:
        """True when a string/object column has enough non-blank values and average string length."""

        if not (pd.api.types.is_string_dtype(series) or series.dtype == object):
            return False
        threshold_num = float(self.quality_config.get("numeric_coercion_min_ratio", 0.8))
        if self._column_numeric_parse_ratio(series) >= threshold_num:
            return False
        min_ratio = float(self.quality_config.get("text_column_min_non_null_ratio", 0.1))
        min_avg_len = float(self.quality_config.get("text_column_min_avg_length", 2.0))
        non_null = series.notna()
        blank = self._blank_text_mask(series)
        effective = non_null & ~blank
        n = len(series)
        if n == 0:
            return False
        if float(effective.sum() / n) < min_ratio:
            return False
        vals = series[effective]
        if vals.empty:
            return False
        lengths = vals.astype(str).str.len().astype(float)
        return bool(lengths.mean() >= min_avg_len)

    def _infer_meaningful_text_columns(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self._infer_text_columns(df) if self._is_meaningful_text_column(df[c])]

    def _text_columns_for_duplicate_and_critical_fallback(self, df: pd.DataFrame) -> list[str]:
        """TEXT mode: legacy all inferred text columns; otherwise prefer meaningful-only."""

        if self._dataset_mode() == "text":
            return self._infer_text_columns(df)
        return self._infer_meaningful_text_columns(df)

    def _include_text_derived_outlier_features(self, df: pd.DataFrame) -> bool:
        """Whether to build char_len/word_len outlier features (honours ``include_text_derived_outliers``)."""

        raw = self.quality_config.get("include_text_derived_outliers", "auto")
        if raw is True:
            return True
        if raw is False:
            return False
        if isinstance(raw, str):
            low = raw.strip().lower()
            if low in ("true", "1", "yes", "on"):
                return True
            if low in ("false", "0", "no", "off"):
                return False
        mode = self._dataset_mode()
        if mode == "tabular":
            return False
        if mode == "text":
            return True
        return bool(self._infer_meaningful_text_columns(df))

    def _text_columns_for_derived_outliers(self, df: pd.DataFrame) -> list[str]:
        if not self._include_text_derived_outlier_features(df):
            return []
        raw = self.quality_config.get("include_text_derived_outliers", "auto")
        explicit_on = raw is True or (
            isinstance(raw, str) and raw.strip().lower() in ("true", "1", "yes", "on")
        )
        if self._dataset_mode() == "text" or explicit_on:
            return self._infer_text_columns(df)
        return self._infer_meaningful_text_columns(df)

    @staticmethod
    def _column_numeric_parse_ratio(series: pd.Series) -> float:
        non_missing = series.notna()
        if not non_missing.any():
            return 0.0
        subset = series[non_missing]
        parsed = pd.to_numeric(subset, errors="coerce")
        ok = parsed.notna()
        return float(ok.sum() / len(subset))

    def _coerce_numeric_like_object_columns(
        self,
        df: pd.DataFrame,
        threshold: float,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Coerce object/string columns to numeric when parse success ratio >= threshold."""

        working = df.copy(deep=True)
        coercion: dict[str, Any] = {"columns_coerced": [], "threshold": threshold}
        for column in list(working.columns):
            if column in _INTERNAL_QUALITY_COLUMNS:
                continue
            if pd.api.types.is_numeric_dtype(working[column]):
                continue
            if not (pd.api.types.is_object_dtype(working[column]) or pd.api.types.is_string_dtype(working[column])):
                continue
            ratio = self._column_numeric_parse_ratio(working[column])
            if ratio < threshold:
                continue
            parsed = pd.to_numeric(working[column], errors="coerce")
            working[column] = parsed
            coercion["columns_coerced"].append({"column": column, "parse_success_ratio": round(ratio, 4)})
        return working, coercion

    def _explicit_id_column(self, df: pd.DataFrame) -> str | None:
        configured = config_column_name(self.quality_config.get("id_column"))
        if configured is not None and configured in df.columns:
            return configured
        for name in ("id", "row_id", "sample_id"):
            if name in df.columns:
                return name
        return None

    def _ensure_internal_row_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach stable row ids and serialized original index for provenance."""

        out = df.copy(deep=True)
        if "__quality_row_id" not in out.columns:
            out["__quality_row_id"] = range(len(out))
        if "__quality_original_index" not in out.columns:
            out["__quality_original_index"] = [self._stringify_index(idx) for idx in out.index]
        return out

    def _columns_for_heterogeneous_cell_normalization(self, df: pd.DataFrame) -> list[str]:
        """Columns where list/ndarray/dict cells should be JSON/text scalars before quality logic."""

        cols: set[str] = set(self._infer_text_columns(df))
        cols.update(iter_config_column_names(self.quality_config.get("critical_text_columns")))
        cols.update(iter_config_column_names(self.quality_config.get("duplicate_subset")))
        cols.update(iter_config_column_names(self.quality_config.get("normalized_text_columns")))
        for key in ("text_column", "title_column", "body_column", "label_column"):
            c = config_column_name(self.quality_config.get(key))
            if c is not None and c in df.columns:
                cols.add(c)
        for hint in ("text", "title", "body", "label", "metadata"):
            if hint in df.columns:
                cols.add(hint)
        lbl = self._infer_label_column(df)
        if lbl is not None:
            cols.add(lbl)
        return sorted(col for col in cols if col not in _INTERNAL_QUALITY_COLUMNS and col in df.columns)

    def _prepare_stage_dataframe(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Pipeline-stage normalization: row ids, original index snapshot, numeric-like coercion.

        Internal columns (__quality_*) are stripped from user-facing outputs via
        `_strip_internal_quality_columns` but retained in interim/audit artifacts when needed.
        """

        threshold = float(self.quality_config.get("numeric_coercion_min_ratio", 0.8))
        meta: dict[str, Any] = {
            "numeric_coercion_min_ratio": threshold,
            "explicit_id_column": self._explicit_id_column(df),
        }
        working = df.copy(deep=True)
        meta["original_index_dtype"] = str(working.index.dtype)
        working = self._ensure_internal_row_ids(working)
        working = normalize_dataframe_object_cells(
            working,
            internal_cols=_INTERNAL_QUALITY_COLUMNS,
            columns_subset=self._columns_for_heterogeneous_cell_normalization(working),
        )
        working, coercion_meta = self._coerce_numeric_like_object_columns(working, threshold)
        meta["numeric_coercion"] = coercion_meta
        return working, meta

    def _strip_internal_quality_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        drop = [c for c in _INTERNAL_QUALITY_COLUMNS if c in df.columns]
        if not drop:
            return df.copy(deep=True)
        return df.drop(columns=drop).copy()

    def _append_row_events(
        self,
        df: pd.DataFrame,
        indices: Any,
        row_actions: list[dict[str, Any]],
        *,
        action: str,
        issue_type: str,
        column_or_feature: str | None,
        strategy: str,
        reason: str,
    ) -> None:
        cap = self._row_action_id_limit()
        if cap <= 0:
            return
        max_total = int(self.quality_config.get("row_actions_max_total", 50_000))
        room = max_total - len(row_actions)
        if room <= 0:
            return
        cap = min(cap, room)
        idx_list = list(indices) if not isinstance(indices, list) else indices
        for idx in idx_list[:cap]:
            if idx not in df.index:
                continue
            row = df.loc[idx]
            rid = row["__quality_row_id"] if "__quality_row_id" in df.columns else None
            oid = row.get("__quality_original_index", self._stringify_index(idx))
            row_actions.append(
                {
                    "row_id": coerce_quality_row_id(rid),
                    "original_index": oid,
                    "action": action,
                    "issue_type": issue_type,
                    "column_or_feature": column_or_feature,
                    "strategy": strategy,
                    "reason": reason,
                    "details": reason,
                }
            )

    def detect_issues(self, df: pd.DataFrame) -> QualityReport:
        """Detect quality issues in a dataframe."""

        if bool(self.quality_config.get("stage_normalization", True)):
            working_df, prep_meta = self._prepare_stage_dataframe(df)
            self.last_stage_prep_meta = prep_meta
        else:
            working_df = self._ensure_internal_row_ids(df.copy(deep=True))
        analysis_df = self._strip_internal_quality_columns(working_df)
        missing_report = self._detect_missing(analysis_df)
        duplicates_report = self._detect_duplicates(analysis_df)
        outliers_report = self._detect_outliers(analysis_df)
        imbalance_report = self._detect_class_imbalance(analysis_df)
        text_quality_report = self._detect_text_quality(analysis_df)
        report = QualityReport(
            generated_at=self._now_iso(),
            row_count=int(len(analysis_df)),
            column_count=int(len(analysis_df.columns)),
            text_columns=self._infer_text_columns(analysis_df),
            numeric_columns=self._infer_numeric_columns(analysis_df),
            missing=missing_report,
            duplicates=duplicates_report,
            outliers=outliers_report,
            class_imbalance=imbalance_report,
            recommendations=self._build_recommendations(
                missing_report,
                duplicates_report,
                outliers_report,
                imbalance_report,
                text_quality_report,
            ),
            text_quality=text_quality_report,
        )
        return report

    def fix(self, df: pd.DataFrame, strategy: dict[str, Any]) -> pd.DataFrame:
        """Apply a cleaning strategy without mutating the input dataframe."""

        strategy = self._expand_text_quality_strategy_aliases(dict(strategy))
        row_actions: list[dict[str, Any]] = []
        if bool(self.quality_config.get("stage_normalization", True)):
            working_df, prep_meta = self._prepare_stage_dataframe(df)
            self.last_stage_prep_meta = prep_meta
        else:
            working_df = df.copy(deep=True)
            working_df = self._ensure_internal_row_ids(working_df)
            self.last_stage_prep_meta = {}
        audit: list[dict[str, Any]] = []
        working_df = self._standardize_text_missing_values(working_df, audit, row_actions)
        working_df = self._apply_missing_strategy(working_df, strategy, audit, row_actions)
        working_df = self._apply_duplicate_strategy(working_df, strategy, audit, row_actions)
        working_df = self._apply_outlier_strategy(working_df, strategy, audit, row_actions)
        working_df = self._apply_text_quality_strategy(working_df, strategy, audit, row_actions)
        self.last_fix_audit = audit
        self.last_fix_row_actions = row_actions
        cleaned = self._strip_internal_quality_columns(working_df)
        cleaned.attrs["quality_audit"] = audit
        cleaned.attrs["quality_row_actions"] = list(row_actions)
        return cleaned

    def compare(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
    ) -> ComparisonReport:
        """Compare dataset quality before and after cleaning."""

        before_strip = self._strip_internal_quality_columns(df_before)
        after_strip = self._strip_internal_quality_columns(df_after)
        before_report = self.detect_issues(before_strip)
        after_report = self.detect_issues(after_strip)
        metrics = [
            self._metric_row("rows", before_report.row_count, after_report.row_count),
            self._metric_row(
                "total_missing_cells",
                before_report.missing.get("total_missing_cells", 0),
                after_report.missing.get("total_missing_cells", 0),
            ),
            self._metric_row(
                "rows_with_any_missing",
                before_report.missing.get("rows_with_any_missing", 0),
                after_report.missing.get("rows_with_any_missing", 0),
            ),
            self._metric_row(
                "full_duplicates",
                before_report.duplicates.get("full_row_duplicates", {}).get("count", 0),
                after_report.duplicates.get("full_row_duplicates", {}).get("count", 0),
            ),
            self._metric_row(
                "subset_duplicates",
                before_report.duplicates.get("subset_duplicates", {}).get("count", 0),
                after_report.duplicates.get("subset_duplicates", {}).get("count", 0),
            ),
            self._metric_row(
                "normalized_text_duplicates",
                before_report.duplicates.get("normalized_text_duplicates", {}).get("count", 0),
                after_report.duplicates.get("normalized_text_duplicates", {}).get("count", 0),
            ),
            self._metric_row(
                "outlier_rows_any_iqr",
                before_report.outliers.get("summary", {}).get("row_count_any_iqr", 0),
                after_report.outliers.get("summary", {}).get("row_count_any_iqr", 0),
            ),
            self._metric_row(
                "outlier_rows_any_zscore",
                before_report.outliers.get("summary", {}).get("row_count_any_zscore", 0),
                after_report.outliers.get("summary", {}).get("row_count_any_zscore", 0),
            ),
        ]
        label_before = (before_report.class_imbalance or {}).get("proportions", {})
        label_after = (after_report.class_imbalance or {}).get("proportions", {})
        metrics.append(
            self._metric_row(
                "class_max_min_ratio",
                (before_report.class_imbalance or {}).get("max_min_ratio", "n/a"),
                (after_report.class_imbalance or {}).get("max_min_ratio", "n/a"),
            )
        )
        tb = self._text_quality_flat_metrics(before_report.text_quality)
        ta = self._text_quality_flat_metrics(after_report.text_quality)
        metrics.extend(
            [
                self._metric_row("text_empty_or_short_rows", tb["text_empty_or_short_rows"], ta["text_empty_or_short_rows"]),
                self._metric_row("text_any_blank_rows", tb["text_any_blank_rows"], ta["text_any_blank_rows"]),
                self._metric_row("text_any_short_rows", tb["text_any_short_rows"], ta["text_any_short_rows"]),
                self._metric_row("text_pii_rows_flagged", tb["text_pii_rows"], ta["text_pii_rows"]),
                self._metric_row("text_near_duplicate_rows_flagged", tb["text_near_duplicate_rows"], ta["text_near_duplicate_rows"]),
                self._metric_row("text_language_mismatch_rows", tb["text_language_mismatch_rows"], ta["text_language_mismatch_rows"]),
                self._metric_row(
                    "text_exact_duplicate_rows_droppable",
                    tb["text_exact_duplicate_rows_droppable"],
                    ta["text_exact_duplicate_rows_droppable"],
                ),
            ]
        )
        markdown_table = self._metrics_to_markdown_table(metrics)
        per_column_missing = self._compare_per_column_missing(
            before_report.missing.get("columns", {}),
            after_report.missing.get("columns", {}),
        )
        per_feature_before = self._outlier_feature_counts(before_report.outliers.get("features", {}))
        per_feature_after = self._outlier_feature_counts(after_report.outliers.get("features", {}))
        br = int(before_report.row_count)
        ar = int(after_report.row_count)
        row_removed = max(0, br - ar)
        markdown_sections = self._comparison_markdown_sections(
            metrics,
            per_column_missing,
            before_report.duplicates,
            after_report.duplicates,
            per_feature_before,
            per_feature_after,
            row_removed,
            br,
            ar,
            label_before,
            label_after,
            before_report.text_quality or {},
            after_report.text_quality or {},
        )
        return ComparisonReport(
            generated_at=self._now_iso(),
            before_rows=int(len(df_before)),
            after_rows=int(len(df_after)),
            metrics=metrics,
            class_distribution_before=label_before,
            class_distribution_after=label_after,
            markdown_table=markdown_table,
            rows_removed=row_removed,
            per_column_missing=per_column_missing,
            duplicates_breakdown={
                "before": dict(before_report.duplicates),
                "after": dict(after_report.duplicates),
            },
            outliers_per_feature={
                "before": per_feature_before,
                "after": per_feature_after,
            },
            summary_markdown=markdown_sections,
            duplicates_breakdown_before=dict(before_report.duplicates),
            duplicates_breakdown_after=dict(after_report.duplicates),
            per_feature_outliers_before=per_feature_before,
            per_feature_outliers_after=per_feature_after,
            row_count_removed=row_removed,
            kept_row_count=ar,
            removed_row_count=row_removed,
            markdown_sections=markdown_sections,
        )

    def build_strategy_recommendations(
        self,
        df: pd.DataFrame,
        report: QualityReport | None = None,
    ) -> dict[str, Any]:
        """Build deterministic recommended vs alternative strategies (no LLM)."""

        if report is None:
            report = self.detect_issues(df)
        previews = self.default_preview_strategies(df)
        recommended = previews["conservative"]
        alternative = previews["strict"]
        missing_rows = int(report.missing.get("rows_with_any_missing", 0) or 0)
        dup_subset = int(report.duplicates.get("subset_duplicates", {}).get("count", 0) or 0)
        dup_nt = int(report.duplicates.get("normalized_text_duplicates", {}).get("count", 0) or 0)
        out_iqr = int(report.outliers.get("summary", {}).get("row_count_any_iqr", 0) or 0)
        imb = report.class_imbalance or {}
        imb_note = (
            "Class imbalance is flagged; only reported—no automatic resampling or rebalancing."
            if imb.get("is_imbalanced")
            else "No strong class imbalance signal under the configured threshold."
        )
        tqm = self._text_quality_flat_metrics(report.text_quality)
        tq_note = ""
        if report.text_quality and report.text_quality.get("enabled"):
            tq_note = (
                f" Text integrity: {tqm['text_empty_or_short_rows']} row(s) empty/too-short union "
                f"({tqm['text_any_blank_rows']} any-blank, {tqm['text_any_short_rows']} too-short), "
                f"{tqm['text_pii_rows']} PII-flagged row(s), {tqm['text_near_duplicate_rows']} near-duplicate row(s), "
                f"{tqm['text_language_mismatch_rows']} language mismatch(es) (when detection enabled)."
            )
        why: dict[str, str] = {
            "recommended": (
                "Tabular-friendly default: median/mode imputation for missing values, drop duplicates on inferred keys, "
                "clip numeric outliers at IQR bounds. Text-length derived features (if present) stay report-only. "
                "Conservative preview redacts PII in text columns but does not drop near-duplicates or language outliers."
                + (f" {missing_rows} row(s) have any missing value." if missing_rows else " No missing-value rows flagged.")
                + (
                    f" Subset duplicate-count {dup_subset}; normalized-text duplicate-count {dup_nt}."
                    if (dup_subset or dup_nt)
                    else ""
                )
                + (f" {out_iqr} row(s) hit at least one IQR-based outlier feature." if out_iqr else "")
                + tq_note
            ),
            "alternative": (
                "Stricter profile: more row drops on categorical missing and remove_iqr outlier handling "
                "(including text-derived length features when configured). "
                "Strict preview may drop rows with empty/short critical text, optional near-duplicates and language "
                "mismatches (per config), and drops rows with PII patterns instead of redacting."
            ),
            "class_imbalance": imb_note,
        }
        return {
            "schema_version": 1,
            "recommended": recommended,
            "alternative": alternative,
            "why": why,
        }

    def _compare_per_column_missing(
        self,
        before_cols: dict[str, Any],
        after_cols: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        names = sorted(set(before_cols) | set(after_cols))
        for name in names:
            b = int((before_cols.get(name) or {}).get("count", 0))
            a = int((after_cols.get(name) or {}).get("count", 0))
            out[name] = {
                "before": b,
                "after": a,
                "delta": a - b,
            }
        return out

    @staticmethod
    def _outlier_feature_counts(features: dict[str, Any]) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for fname, payload in features.items():
            result[fname] = {
                "iqr_count": int((payload.get("iqr") or {}).get("count", 0)),
                "zscore_count": int((payload.get("zscore") or {}).get("count", 0)),
            }
        return result

    def _comparison_markdown_sections(
        self,
        metrics: list[dict[str, Any]],
        per_column_missing: dict[str, dict[str, Any]],
        dup_before: dict[str, Any],
        dup_after: dict[str, Any],
        feat_before: dict[str, dict[str, int]],
        feat_after: dict[str, dict[str, int]],
        row_removed: int,
        before_rows: int,
        after_rows: int,
        class_before: dict[str, Any],
        class_after: dict[str, Any],
        text_quality_before: dict[str, Any] | None = None,
        text_quality_after: dict[str, Any] | None = None,
    ) -> str:
        tqb = text_quality_before or {}
        tqa = text_quality_after or {}
        lines = [
            "## Summary metrics",
            self._metrics_to_markdown_table(metrics),
            "",
            "## Row retention",
            f"- before_rows: `{before_rows}`",
            f"- after_rows: `{after_rows}`",
            f"- rows_removed (max of 0, before − after): `{row_removed}`",
            "",
            "## Missing values by column",
            self._markdown_table(
                ["column", "missing_before", "missing_after", "delta"],
                [[c, v["before"], v["after"], v["delta"]] for c, v in sorted(per_column_missing.items())],
            ),
            "",
            "## Duplicates (before)",
            self._json_fenced_block(dup_before),
            "",
            "## Duplicates (after)",
            self._json_fenced_block(dup_after),
            "",
            "## Text integrity (detected)",
            "Before:",
            self._json_fenced_block(tqb),
            "",
            "After:",
            self._json_fenced_block(tqa),
            "",
            "## Text quality metrics (before → after)",
            self._text_quality_compare_table(tqb, tqa),
            "",
            "## Class imbalance",
            self._markdown_table(
                ["state", "label_distribution"],
                [
                    ["before", self._stringify_value(class_before)],
                    ["after", self._stringify_value(class_after)],
                ],
            ),
            "",
            "## Outlier counts by feature (IQR / z-score)",
        ]
        feat_names = sorted(set(feat_before) | set(feat_after))
        feat_rows = []
        for name in feat_names:
            b = feat_before.get(name, {"iqr_count": 0, "zscore_count": 0})
            a = feat_after.get(name, {"iqr_count": 0, "zscore_count": 0})
            feat_rows.append(
                [
                    name,
                    b["iqr_count"],
                    b["zscore_count"],
                    a["iqr_count"],
                    a["zscore_count"],
                ]
            )
        lines.append(
            self._markdown_table(
                ["feature", "iqr_before", "z_before", "iqr_after", "z_after"],
                feat_rows,
            )
        )
        return "\n".join(lines)

    def prepare_review_bundle(
        self,
        raw_df: pd.DataFrame,
        preview_strategies: dict[str, dict[str, Any]] | None = None,
        task_description: str | None = None,
    ) -> dict[str, str]:
        """Run detect, optional preview fixtures, strategy recommendations, and review bundle."""

        prepared, prep_meta = self._prepare_stage_dataframe(raw_df)
        self.last_stage_prep_meta = prep_meta
        preview_strategies = preview_strategies or self.default_preview_strategies(prepared)
        quality_report = self.detect_issues(prepared)
        quality_report_json = self.storage.save_json(
            "reports/quality/quality_report.json",
            quality_report.to_dict(),
        )
        quality_report_md = self.storage.save_markdown(
            "reports/quality/quality_report.md",
            self._quality_report_to_markdown(quality_report),
        )
        recommendations = self.build_strategy_recommendations(prepared, quality_report)
        strategy_recommendations_json = self.storage.save_json(
            "reports/quality/strategy_recommendations.json",
            recommendations,
        )

        before_cmp = self._strip_internal_quality_columns(prepared)
        plot_paths = self._export_plots(before_cmp, quality_report)
        gen_preview = bool(self.quality_config.get("generate_preview_outputs_before_review", False))

        artifacts: dict[str, str] = {
            "quality_report_json": quality_report_json,
            "quality_report_md": quality_report_md,
            "strategy_recommendations_json": strategy_recommendations_json,
            **plot_paths,
        }

        if gen_preview:
            conservative_df = self.fix(prepared, preview_strategies["conservative"])
            conservative_audit = list(self.last_fix_audit)
            conservative_row_actions = list(self.last_fix_row_actions)
            strict_df = self.fix(prepared, preview_strategies["strict"])
            strict_audit = list(self.last_fix_audit)
            strict_row_actions = list(self.last_fix_row_actions)
            conservative_path = self.storage.save_dataframe(
                "data/interim/cleaned_preview_conservative.parquet",
                conservative_df,
            )
            strict_path = self.storage.save_dataframe(
                "data/interim/cleaned_preview_strict.parquet",
                strict_df,
            )
            conservative_comparison = self.compare(before_cmp, conservative_df)
            strict_comparison = self.compare(before_cmp, strict_df)
            comparison_payload = {
                "generated_at": self._now_iso(),
                "previews": {
                    "conservative": conservative_comparison.to_dict(),
                    "strict": strict_comparison.to_dict(),
                },
            }
            comparison_report_json = self.storage.save_json(
                "reports/quality/comparison_report.json",
                comparison_payload,
            )
            comparison_report_md = self.storage.save_markdown(
                "reports/quality/comparison_report.md",
                self._comparison_bundle_to_markdown(
                    conservative_comparison,
                    strict_comparison,
                    preview_strategies,
                    conservative_audit,
                    strict_audit,
                ),
            )
            bundle_body = self._build_review_bundle_markdown(
                raw_df=before_cmp,
                raw_df_with_ids=prepared,
                report=quality_report,
                conservative_df=conservative_df,
                strict_df=strict_df,
                conservative_comparison=conservative_comparison,
                strict_comparison=strict_comparison,
                conservative_audit=conservative_audit,
                strict_audit=strict_audit,
                conservative_row_actions=conservative_row_actions,
                strict_row_actions=strict_row_actions,
                preview_strategies=preview_strategies,
                plot_paths=plot_paths,
            )
            artifacts["comparison_report_json"] = comparison_report_json
            artifacts["comparison_report_md"] = comparison_report_md
            artifacts["cleaned_preview_conservative"] = conservative_path
            artifacts["cleaned_preview_strict"] = strict_path
        else:
            bundle_body = self._build_analysis_review_bundle_markdown(
                raw_df=before_cmp,
                raw_df_with_ids=prepared,
                report=quality_report,
                recommendations=recommendations,
                preview_strategies=preview_strategies,
                plot_paths=plot_paths,
            )

        review_bundle_md = self.storage.save_markdown("review/quality_review_bundle.md", bundle_body)
        decision_template_path = self.storage.save_json(
            "review/quality_review_decision_template.json",
            self._build_review_decision_template(before_cmp, recommendations),
        )
        artifacts["review_bundle_md"] = review_bundle_md
        artifacts["review_decision_template_json"] = decision_template_path

        if task_description:
            explanation = self.explain_issues_and_recommend_strategy(quality_report, task_description)
            explanation += (
                "\n\n---\n"
                "Final cleaning strategy must be explicitly approved by a human reviewer "
                "before applying the decision."
            )
            artifacts["strategy_recommendation_md"] = self.storage.save_markdown(
                "reports/quality/strategy_recommendation.md",
                explanation,
            )
        self.last_artifacts = artifacts
        return artifacts

    def _resolve_final_strategy(self, decision: dict[str, Any], prepared_df: pd.DataFrame) -> dict[str, Any]:
        """Pick strategy dict: non-null ``final_strategy`` wins, else ``available_strategies[name]``."""

        fs = decision.get("final_strategy")
        if fs is not None and isinstance(fs, dict) and len(fs) > 0:
            return dict(fs)

        live = self.default_preview_strategies(prepared_df)
        merged: dict[str, Any] = {
            "recommended": live["conservative"],
            "alternative": live["strict"],
        }
        file_avail = decision.get("available_strategies")
        if isinstance(file_avail, dict):
            for k, v in file_avail.items():
                if isinstance(v, dict) and v:
                    merged[k] = v
        legacy = decision.get("available_preview_strategies")
        if isinstance(legacy, dict):
            if "conservative" in legacy and isinstance(legacy["conservative"], dict):
                merged.setdefault("recommended", legacy["conservative"])
            if "strict" in legacy and isinstance(legacy["strict"], dict):
                merged.setdefault("alternative", legacy["strict"])

        name = decision.get("selected_strategy_name")
        if name is None:
            sp = decision.get("selected_preview")
            if sp == "conservative":
                name = "recommended"
            elif sp == "strict":
                name = "alternative"
            else:
                name = sp
        if name is None or (isinstance(name, str) and not name.strip()):
            name = "recommended"
        if not isinstance(name, str):
            raise ValueError(
                f"selected_strategy_name must be a string; got {type(name).__name__}."
            )
        name = name.strip()
        if name not in merged:
            valid = sorted(merged.keys())
            raise ValueError(
                f"Unknown selected_strategy_name {name!r}. Use one of {valid}, "
                "or set final_strategy to a full custom strategy object."
            )
        chosen = merged[name]
        if not isinstance(chosen, dict) or not chosen:
            raise ValueError(
                f"Strategy {name!r} must be a non-empty JSON object; got {type(chosen).__name__}."
            )
        return dict(chosen)

    def apply_review_decision(
        self,
        raw_df: pd.DataFrame,
        decision: str | Path | dict[str, Any],
    ) -> pd.DataFrame:
        """Apply a reviewed decision. Raises until explicit approval is present."""

        decision_payload = self._load_decision(decision)
        if not decision_payload.get("approved", False):
            raise ValueError(
                "Quality review decision is not approved. "
                "Set approved=true before final cleaning can continue."
            )


        prepared, prep_meta = self._prepare_stage_dataframe(raw_df)
        self.last_stage_prep_meta = prep_meta
        try:
            final_strategy = self._resolve_final_strategy(decision_payload, prepared)
        except KeyError as exc:
            raise ValueError(
                "Quality decision is missing required keys to resolve a cleaning strategy. "
                f"Detail: {exc}"
            ) from exc


        cleaned_df = self.fix(prepared, final_strategy)
        before_cmp = self._strip_internal_quality_columns(prepared)
        comparison = self.compare(before_cmp, cleaned_df)
        if len(cleaned_df) >= 10_000:
            print(
                "QUALITY: running final detect_issues on cleaned frame (then save reports and parquet) ...",
                flush=True,
            )
        final_report = self.detect_issues(cleaned_df)
        final_paths = self._persist_final_quality_artifacts(
            decision_payload,
            cleaned_df,
            final_report,
            comparison,
            prepared,
            before_cmp,
        )
        self.last_artifacts = {**self.last_artifacts, **final_paths}
        self._write_quality_stage_status(
            "approved_and_applied",
            {"artifacts": final_paths, "decision": decision_payload},
        )
        return cleaned_df

    def _persist_final_quality_artifacts(
        self,
        decision_payload: dict[str, Any],
        cleaned_df: pd.DataFrame,
        final_report: QualityReport,
        comparison: ComparisonReport,
        prepared_with_ids: pd.DataFrame,
        before_cmp: pd.DataFrame,
    ) -> dict[str, str]:
        """Write final parquet, reports, row-level audit, and decision record."""

        paths: dict[str, str] = {}
        paths["cleaned_final"] = self.storage.save_dataframe(
            "data/interim/cleaned_final.parquet",
            cleaned_df,
        )
        paths["final_quality_report_json"] = self.storage.save_json(
            "reports/quality/final_quality_report.json",
            final_report.to_dict(),
        )
        paths["final_quality_report_md"] = self.storage.save_markdown(
            "reports/quality/final_quality_report.md",
            self._quality_report_to_markdown(final_report),
        )
        paths["final_comparison_report_json"] = self.storage.save_json(
            "reports/quality/final_comparison_report.json",
            comparison.to_dict(),
        )
        cmp_md = "\n".join(
            [
                "# Final comparison (before → after cleaning)",
                "",
                comparison.markdown_table,
                "",
                comparison.markdown_sections,
            ]
        )
        paths["final_comparison_report_md"] = self.storage.save_markdown(
            "reports/quality/final_comparison_report.md",
            cmp_md,
        )
        row_actions = list(self.last_fix_row_actions)
        ra_df = pd.DataFrame(row_actions) if row_actions else pd.DataFrame()
        paths["final_row_actions"] = self.storage.save_dataframe(
            "review/final_row_actions.parquet",
            ra_df,
        )
        removed_ids = {
            r["row_id"]
            for r in row_actions
            if r.get("action") == "removed" and r.get("row_id") is not None
        }
        rem_out = pd.DataFrame()
        if removed_ids and "__quality_row_id" in prepared_with_ids.columns:
            rem = prepared_with_ids[prepared_with_ids["__quality_row_id"].isin(removed_ids)].copy()
            rem_out = self._strip_internal_quality_columns(rem)
        if rem_out.empty and not before_cmp.empty:
            rem_out = before_cmp.loc[~before_cmp.index.isin(cleaned_df.index)].copy()
        paths["final_removed_rows"] = self.storage.save_dataframe(
            "review/final_removed_rows.parquet",
            rem_out,
        )
        applied_record = {
            "applied_at": self._now_iso(),
            "approved": bool(decision_payload.get("approved")),
            "selected_preview": decision_payload.get("selected_preview"),
            "final_strategy": decision_payload.get("final_strategy"),
            "artifacts": paths,
            "stage_prep_meta": self.last_stage_prep_meta,
        }
        paths["final_review_decision_applied"] = self.storage.save_json(
            "review/final_review_decision_applied.json",
            applied_record,
        )
        return paths

    def _write_quality_stage_status(self, status: str, extra: dict[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {
            "status": status,
            "updated_at": self._now_iso(),
        }
        if extra:
            payload.update(extra)
        return self.storage.save_json("review/quality_stage_status.json", payload)

    def explain_issues_and_recommend_strategy(
        self,
        report: QualityReport | dict[str, Any],
        task_description: str,
    ) -> str:
        """Explain quality findings and recommend a strategy without mutating data."""

        report_payload = report.to_dict() if isinstance(report, QualityReport) else dict(report)
        llm = self._get_llm()
        if llm is None:
            return self._fallback_strategy_explanation(report_payload, task_description)

        prompt = (
            "You are a data-quality advisor for an NLP dataset pipeline.\n"
            "Explain the detected issues in plain language, recommend a conservative and a strict "
            "cleaning strategy, and explicitly say that the decision must be reviewed by a human.\n"
            "Do not modify data and do not claim you executed cleaning.\n\n"
            f"Task description:\n{task_description}\n\n"
            f"Quality report:\n{json.dumps(report_payload, ensure_ascii=False, indent=2)}"
        )
        try:
            response = llm.invoke(prompt)
            content = getattr(response, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
        except Exception:
            pass
        return self._fallback_strategy_explanation(report_payload, task_description)

    def run_stage(
        self,
        raw_df: pd.DataFrame,
        decision: str | Path | dict[str, Any] | None = None,
        preview_strategies: dict[str, dict[str, Any]] | None = None,
        task_description: str | None = None,
        *,
        save_llm_explanation: bool = True,
        raise_on_awaiting_review: bool = True,
    ) -> pd.DataFrame | QualityStageResult:
        """
        Pipeline-friendly entrypoint: apply an approved decision or stop for HITL.

        When no approved decision is provided, writes `review/quality_stage_status.json`
        with status ``awaiting_review`` and raises `HumanReviewRequired`.

        On success after approval, returns the cleaned dataframe (same contract as
        `apply_review_decision`).
        """

        decision_dict: dict[str, Any] | None = None
        if decision is not None:
            decision_dict = decision if isinstance(decision, dict) else self._load_decision(decision)
        if decision_dict and decision_dict.get("approved") is True:
            return self.apply_review_decision(raw_df, decision_dict)

        artifacts = self.prepare_review_bundle(
            raw_df,
            preview_strategies=preview_strategies,
            task_description=task_description if save_llm_explanation else None,
        )
        self.last_artifacts = artifacts
        status_path = self._write_quality_stage_status(
            "awaiting_review",
            {
                "artifacts": artifacts,
                "hint": "Fill review/quality_review_decision_template.json and re-run with decision.",
            },
        )
        stage_result = QualityStageResult(
            status="awaiting_review",
            artifacts=artifacts,
            quality_stage_status_json=status_path,
            decision_template_json=artifacts.get("review_decision_template_json", ""),
            message=(
                "Quality stage is awaiting human review. Approve the decision template, then rerun "
                "with approved=true decision payload."
            ),
        )
        if not raise_on_awaiting_review:
            return stage_result
        raise HumanReviewRequired(
            "Quality stage is awaiting human review. Approve the decision template, then call run_stage "
            "again with the same raw dataframe and the approved decision payload.",
            stage_result=stage_result.to_dict(),
        )

    def default_preview_strategies(self, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Return conservative and strict preview strategies."""

        critical_text_columns = self._infer_critical_text_columns(df)
        normalized_subset = self._default_normalized_duplicate_subset(df)
        text_outlier_features = self._default_text_outlier_features(df)
        # Conservative: clip numeric outliers; text-derived lengths are inspect-only (no fake clip).
        conservative_outliers: dict[str, Any]
        if text_outlier_features:
            conservative_outliers = {
                "default": {"strategy": "clip_iqr"},
                "per_column": {feat: {"strategy": "report_only_iqr"} for feat in text_outlier_features},
            }
        else:
            conservative_outliers = {"default": {"strategy": "clip_iqr"}}

        # Strict: remove rows that are length outliers on text-derived features; remove_iqr on numerics.
        strict_outliers: dict[str, Any]
        if text_outlier_features:
            strict_outliers = {
                "default": {"strategy": "remove_iqr"},
                "per_column": {feat: {"strategy": "remove_iqr"} for feat in text_outlier_features},
            }
        else:
            strict_outliers = {"default": {"strategy": "remove_iqr"}}

        previews: dict[str, dict[str, Any]] = {
            "conservative": {
                "missing": {
                    "default_numeric": {"strategy": "median"},
                    "default_categorical": {"strategy": "mode"},
                    "per_column": {
                        column: {"strategy": "drop_rows"}
                        for column in critical_text_columns
                    },
                },
                "duplicates": {
                    "action": "drop",
                    "subset": normalized_subset or None,
                },
                "outliers": conservative_outliers,
            },
            "strict": {
                "missing": {
                    "default_numeric": {"strategy": "median"},
                    "default_categorical": {"strategy": "drop_rows"},
                    "per_column": {
                        column: {"strategy": "drop_rows"}
                        for column in critical_text_columns
                    },
                },
                "duplicates": {
                    "action": "drop",
                    "subset": normalized_subset or self._infer_duplicate_subset(df) or None,
                },
                "outliers": strict_outliers,
            },
        }
        tc = self._text_checks_config()
        base_tq_conservative = {
            "pii": {
                "action": "redact",
                "columns": "auto",
                "redact_emails": tc["pii"]["redact_emails"],
                "redact_phones": tc["pii"]["redact_phones"],
                "redact_handles": tc["pii"]["redact_handles"],
            },
            "empty_or_short": {"action": "none"},
            "near_duplicates": {"action": "none"},
            "language": {"action": "none"},
        }
        base_tq_strict = {
            "pii": {"action": "drop_rows", "columns": "auto"},
            "empty_or_short": {"action": "drop_rows", "min_chars": tc["min_text_chars"], "columns": "auto"},
            "near_duplicates": (
                {"action": "drop_rows", "similarity_threshold": tc["near_duplicates"]["similarity_threshold"]}
                if tc["near_duplicates"]["enabled"]
                else {"action": "none"}
            ),
            "language": (
                {
                    "action": "drop_rows",
                    "expected": tc["language"]["expected"],
                    "max_rows_detect": tc["language"]["max_rows_detect"],
                }
                if (tc["language"]["enabled"] and tc["language"]["expected"])
                else {"action": "none"}
            ),
        }
        previews["conservative"]["text_quality"] = base_tq_conservative
        previews["strict"]["text_quality"] = base_tq_strict
        return previews

    def _expand_text_quality_strategy_aliases(self, strategy: dict[str, Any]) -> dict[str, Any]:
        """Merge convenience keys (``drop_empty_text``, ``redact_basic_pii``, …) into ``text_quality``.

        Nested ``strategy['text_quality']`` always wins via :meth:`dict.setdefault` for inner fields.
        """

        out = {k: v for k, v in strategy.items() if k not in TEXT_QUALITY_STRATEGY_ALIAS_KEYS}
        tq_in = strategy.get("text_quality")
        tq: dict[str, Any] = dict(tq_in) if isinstance(tq_in, dict) else {}
        tc = self._text_checks_config()

        if strategy.get("redact_basic_pii"):
            p = tq.setdefault("pii", {})
            if isinstance(p, dict):
                p.setdefault("action", "redact")
                p.setdefault("columns", "auto")
                p.setdefault("redact_emails", tc["pii"]["redact_emails"])
                p.setdefault("redact_phones", tc["pii"]["redact_phones"])
                p.setdefault("redact_handles", tc["pii"]["redact_handles"])

        de = bool(strategy.get("drop_empty_text"))
        ds = bool(strategy.get("drop_short_text"))
        if de or ds:
            es = tq.setdefault("empty_or_short", {})
            if isinstance(es, dict):
                es.setdefault("action", "drop_rows")
                es.setdefault("min_chars", tc["min_text_chars"])
                es.setdefault("columns", "auto")
                if de and not ds:
                    es["drop_blank"] = True
                    es["drop_short"] = False
                elif ds and not de:
                    es["drop_blank"] = False
                    es["drop_short"] = True
                else:
                    es["drop_blank"] = True
                    es["drop_short"] = True

        if strategy.get("drop_near_duplicates"):
            nd = tq.setdefault("near_duplicates", {})
            if isinstance(nd, dict):
                nd.setdefault("action", "drop_rows")
                nd.setdefault("similarity_threshold", tc["near_duplicates"]["similarity_threshold"])

        lf = strategy.get("language_filter")
        if lf:
            lang = tq.setdefault("language", {})
            if isinstance(lang, dict):
                if isinstance(lf, dict):
                    lang.setdefault("action", str(lf.get("action", "drop_rows")).lower())
                    exp = _normalize_language_expected(lf.get("expected"))
                    if exp is not None:
                        lang.setdefault("expected", exp)
                    else:
                        lang.setdefault("expected", tc["language"]["expected"])
                    lang.setdefault(
                        "max_rows_detect",
                        int(lf.get("max_rows_detect", tc["language"]["max_rows_detect"])),
                    )
                else:
                    lang.setdefault("action", "drop_rows")
                    lang.setdefault("expected", tc["language"]["expected"])
                    lang.setdefault("max_rows_detect", tc["language"]["max_rows_detect"])

        if tq:
            out["text_quality"] = tq
        return out

    def _text_checks_config(self) -> dict[str, Any]:
        """Merge `quality.text_checks` with legacy keys (`min_text_length`, `language_filter`, etc.)."""

        qc = self.quality_config
        user = qc.get("text_checks")
        user = user if isinstance(user, dict) else {}
        lf = qc.get("language_filter") if isinstance(qc.get("language_filter"), dict) else {}
        pii_legacy = qc.get("pii_redaction") if isinstance(qc.get("pii_redaction"), dict) else {}
        nd_user = user.get("near_duplicates") if isinstance(user.get("near_duplicates"), dict) else {}
        lang_user = user.get("language") if isinstance(user.get("language"), dict) else {}
        pii_user = user.get("pii") if isinstance(user.get("pii"), dict) else {}

        allow_raw = lf.get("allow_languages")
        expected_legacy: str | None = None
        if isinstance(allow_raw, (list, tuple)) and allow_raw:
            expected_legacy = _normalize_language_expected(allow_raw[0])
        else:
            try:
                import numpy as np

                if isinstance(allow_raw, np.ndarray) and allow_raw.size:
                    expected_legacy = _normalize_language_expected(allow_raw.reshape(-1)[0])
            except ImportError:
                pass

        near_enabled = nd_user.get("enabled")
        if near_enabled is None:
            near_enabled = bool(qc.get("drop_near_duplicates", False))

        lang_enabled = lang_user.get("enabled")
        if lang_enabled is None:
            lang_enabled = bool(lf.get("enabled", False))

        pii_enabled = pii_user.get("enabled")
        if pii_enabled is None:
            pii_enabled = bool(pii_legacy.get("enabled", True))

        min_chars = user.get("min_text_chars")
        if min_chars is None:
            min_chars = qc.get("min_text_length", 15)

        return {
            "enabled": bool(user.get("enabled", True)),
            "min_text_chars": int(min_chars),
            "near_duplicates": {
                "enabled": bool(near_enabled),
                "similarity_threshold": float(
                    nd_user.get("similarity_threshold", qc.get("near_duplicate_similarity", 0.92))
                ),
                "max_rows_scan": int(nd_user.get("max_rows_scan", 6000)),
            },
            "language": {
                "enabled": bool(lang_enabled),
                "expected": _normalize_language_expected(
                    lang_user.get("expected", expected_legacy),
                ),
                "max_rows_detect": int(lang_user.get("max_rows_detect", 2500)),
            },
            "pii": {
                "enabled": bool(pii_enabled),
                "redact_emails": bool(pii_user.get("redact_emails", pii_legacy.get("redact_emails", True))),
                "redact_phones": bool(
                    pii_user.get("redact_phones", pii_legacy.get("redact_phone_numbers", True))
                ),
                "redact_handles": bool(
                    pii_user.get("redact_handles", pii_legacy.get("redact_user_handles", True))
                ),
            },
        }

    def _primary_content_column(self, df: pd.DataFrame) -> str | None:
        for key in ("text_column", "body_column"):
            c = config_column_name(self.quality_config.get(key))
            if c is not None and c in df.columns and c in self._infer_text_columns(df):
                return c
        for fallback in ("text", "body", "content"):
            if fallback in df.columns and fallback in self._infer_text_columns(df):
                return fallback
        mt = self._infer_meaningful_text_columns(df)
        return mt[0] if mt else None

    def _text_columns_for_integrity_checks(self, df: pd.DataFrame) -> list[str]:
        configured = iter_config_column_names(self.quality_config.get("critical_text_columns"))
        cols = [c for c in configured if c in df.columns and c in self._infer_text_columns(df)]
        if cols:
            return cols
        mt = self._infer_meaningful_text_columns(df)
        if mt:
            return mt[:5]
        return [c for c in self._infer_text_columns(df)[:3]]

    def _detect_text_quality(self, df: pd.DataFrame) -> dict[str, Any]:
        cfg = self._text_checks_config()
        if not cfg["enabled"]:
            return {"enabled": False, "skipped": True, "reason": "text_checks.disabled"}

        text_cols = self._text_columns_for_integrity_checks(df)
        min_chars = max(1, cfg["min_text_chars"])

        empty_short_by_col: dict[str, dict[str, Any]] = {}
        union_mask = pd.Series(False, index=df.index, dtype=bool)
        any_blank = pd.Series(False, index=df.index, dtype=bool)
        any_short_non_blank = pd.Series(False, index=df.index, dtype=bool)
        for col in text_cols:
            series = df[col]
            blank = self._blank_text_mask(series)
            empty_like = self._empty_text_mask(series)
            lengths = series.fillna("").astype(str).str.strip().str.len()
            too_short = (~empty_like) & (lengths < min_chars)
            flagged = empty_like | too_short
            empty_short_by_col[col] = {
                "empty_or_blank_rows": int(empty_like.sum()),
                "too_short_rows": int((too_short & ~empty_like).sum()),
                "min_chars": min_chars,
            }
            union_mask = union_mask | flagged
            any_blank = any_blank | empty_like
            any_short_non_blank = any_short_non_blank | too_short

        primary = self._primary_content_column(df)
        exact_txt: dict[str, Any] = {"column": primary, "duplicate_rows_droppable": 0, "rows_in_duplicate_groups": 0}
        if primary and primary in df.columns:
            sub_drop, sub_all = _duplicate_masks_for_frame(df[[primary]])
            exact_txt = {
                "column": primary,
                "duplicate_rows_droppable": int(sub_drop.sum()),
                "rows_in_duplicate_groups": int(sub_all.sum()),
            }

        pii_section: dict[str, Any] = {
            "enabled": cfg["pii"]["enabled"],
            "rows_flagged": 0,
            "breakdown": {"email_rows": 0, "phone_rows": 0, "handle_rows": 0, "any_rows": 0},
        }
        if cfg["pii"]["enabled"] and text_cols:
            any_pii = pii_hit_mask(df[text_cols[0]])
            for col in text_cols[1:]:
                any_pii = any_pii | pii_hit_mask(df[col])
            agg = {"email_rows": 0, "phone_rows": 0, "handle_rows": 0}
            for col in text_cols:
                b = pii_breakdown_counts(df[col])
                for k in agg:
                    agg[k] += b[k]
            agg["any_rows"] = int(any_pii.sum())
            pii_section["rows_flagged"] = agg["any_rows"]
            pii_section["breakdown"] = agg

        nd_section: dict[str, Any] = {
            "enabled": cfg["near_duplicates"]["enabled"],
            "rows_with_near_duplicate_partner": 0,
            "skipped_reason": None,
            "sample_indices": [],
        }
        if cfg["near_duplicates"]["enabled"] and primary and primary in df.columns:
            series = df[primary].fillna("").astype(str)
            max_scan = int(cfg["near_duplicates"]["max_rows_scan"])
            labels = list(df.index)
            total_rows = len(df)
            if total_rows > max_scan:
                rnd_idx = (
                    pd.Series(range(total_rows), index=labels)
                    .sample(n=max_scan, random_state=0)
                    .index.tolist()
                )
                texts = [str(series.loc[i]) for i in rnd_idx]
                labels = rnd_idx
                nd_section["skipped_reason"] = f"sampled_{max_scan}_of_{total_rows}"
            else:
                texts = series.tolist()
            thr = float(cfg["near_duplicates"]["similarity_threshold"])
            to_drop = near_duplicate_drop_indices(texts, labels, similarity_threshold=thr)
            nd_section["rows_with_near_duplicate_partner"] = len(to_drop)
            nd_section["sample_indices"] = self._sample_indices(list(to_drop))

        lang_section: dict[str, Any] = {
            "enabled": cfg["language"]["enabled"],
            "expected": cfg["language"]["expected"],
            "mismatch_rows": 0,
            "skipped_reason": None,
            "sample_indices": [],
        }
        exp = cfg["language"]["expected"]
        if cfg["language"]["enabled"] and exp and primary and primary in df.columns:
            mismatched, skip = language_mismatch_indices(
                df[primary],
                exp,
                max_rows=int(cfg["language"]["max_rows_detect"]),
            )
            if skip:
                lang_section["skipped_reason"] = skip
            elif mismatched is not None:
                lang_section["mismatch_rows"] = len(mismatched)
                lang_section["sample_indices"] = self._sample_indices(list(mismatched))

        return {
            "enabled": True,
            "config": {
                "min_text_chars": min_chars,
                "near_duplicate_threshold": cfg["near_duplicates"]["similarity_threshold"],
            },
            "columns_scanned": text_cols,
            "empty_or_short": {
                "by_column": empty_short_by_col,
                "rows_union": int(union_mask.sum()),
                "rows_any_blank": int(any_blank.sum()),
                "rows_any_short_non_blank": int(any_short_non_blank.sum()),
                "sample_indices": self._sample_indices(list(df.index[union_mask])),
            },
            "exact_text_duplicates": exact_txt,
            "pii": pii_section,
            "near_duplicates": nd_section,
            "language": lang_section,
        }

    def _normalize_text_quality_config(self, raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError("text_quality strategy must be a dict or omitted.")
        return raw

    def _apply_text_quality_strategy(
        self,
        df: pd.DataFrame,
        strategy: dict[str, Any],
        audit: list[dict[str, Any]],
        row_actions: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Apply optional ``strategy['text_quality']`` (PII redact/drop, short text, near-dup, language)."""

        tq = self._normalize_text_quality_config(strategy.get("text_quality"))
        if not tq:
            return df

        working_df = df.copy(deep=True)
        tc = self._text_checks_config()
        text_cols = self._text_columns_for_integrity_checks(working_df)
        primary = self._primary_content_column(working_df)

        pii_cfg = tq.get("pii")
        if isinstance(pii_cfg, dict):
            action = str(pii_cfg.get("action", "none")).lower()
            if action not in {"none", "skip", ""}:
                pcols = pii_cfg.get("columns")
                if pcols == "auto" or pcols is None:
                    target_cols = text_cols
                else:
                    target_cols = [c for c in pii_cfg["columns"] if c in working_df.columns]
                redact_emails = bool(pii_cfg.get("redact_emails", tc["pii"]["redact_emails"]))
                redact_phones = bool(pii_cfg.get("redact_phones", tc["pii"]["redact_phones"]))
                redact_handles = bool(pii_cfg.get("redact_handles", tc["pii"]["redact_handles"]))
                if action == "redact" and target_cols:
                    total_hits = 0
                    for col in target_cols:
                        series = working_df[col]
                        mask = pii_hit_mask(series)
                        total_hits += int(mask.sum())
                        working_df.loc[mask, col] = [
                            redact_pii(
                                str(v),
                                emails=redact_emails,
                                phones=redact_phones,
                                handles=redact_handles,
                            )
                            for v in working_df.loc[mask, col]
                        ]
                    audit.append(
                        {
                            "type": "text_quality",
                            "subtype": "pii",
                            "strategy": "redact",
                            "affected_cells": total_hits,
                            "mutation_applied": total_hits > 0,
                        }
                    )
                elif action == "drop_rows" and target_cols:
                    masks = [pii_hit_mask(working_df[c]) for c in target_cols]
                    drop_m = masks[0]
                    for m in masks[1:]:
                        drop_m = drop_m | m
                    if drop_m.any():
                        n = int(drop_m.sum())
                        self._append_row_events(
                            working_df,
                            list(working_df.index[drop_m]),
                            row_actions,
                            action="removed",
                            issue_type="text_quality_pii",
                            column_or_feature=",".join(target_cols),
                            strategy=action,
                            reason="row dropped: PII pattern in text column",
                        )
                        working_df = working_df.loc[~drop_m].copy()
                        audit.append(
                            {
                                "type": "text_quality",
                                "subtype": "pii",
                                "strategy": "drop_rows",
                                "affected_rows": n,
                                "mutation_applied": True,
                            }
                        )

        es_cfg = tq.get("empty_or_short")
        if isinstance(es_cfg, dict) and str(es_cfg.get("action", "none")).lower() == "drop_rows":
            min_c = int(es_cfg.get("min_chars", tc["min_text_chars"]))
            drop_blank = bool(es_cfg.get("drop_blank", True))
            drop_short = bool(es_cfg.get("drop_short", True))
            cols = es_cfg.get("columns")
            if cols == "auto" or cols is None:
                es_cols = text_cols
            else:
                es_cols = [c for c in cols if c in working_df.columns]
            drop_m = pd.Series(False, index=working_df.index, dtype=bool)
            for col in es_cols:
                s = working_df[col]
                empty_like = self._empty_text_mask(s)
                short = (~empty_like) & (s.fillna("").astype(str).str.strip().str.len() < min_c)
                part = pd.Series(False, index=working_df.index, dtype=bool)
                if drop_blank:
                    part = part | empty_like
                if drop_short:
                    part = part | short
                drop_m = drop_m | part
            if drop_m.any():
                n = int(drop_m.sum())
                self._append_row_events(
                    working_df,
                    list(working_df.index[drop_m]),
                    row_actions,
                    action="removed",
                    issue_type="text_quality_empty_short",
                    column_or_feature=",".join(es_cols),
                    strategy="drop_rows",
                    reason=f"empty or text shorter than {min_c} chars",
                )
                working_df = working_df.loc[~drop_m].copy()
                audit.append(
                    {
                        "type": "text_quality",
                        "subtype": "empty_or_short",
                        "strategy": "drop_rows",
                        "min_chars": min_c,
                        "affected_rows": n,
                        "mutation_applied": True,
                    }
                )

        nd_cfg = tq.get("near_duplicates")
        if (
            isinstance(nd_cfg, dict)
            and str(nd_cfg.get("action", "none")).lower() == "drop_rows"
            and primary
            and primary in working_df.columns
        ):
            thr = float(nd_cfg.get("similarity_threshold", tc["near_duplicates"]["similarity_threshold"]))
            series = working_df[primary].fillna("").astype(str)
            texts = [series.iloc[i] for i in range(len(series))]
            labels = list(working_df.index)
            to_drop = near_duplicate_drop_indices(texts, labels, similarity_threshold=thr)
            drop_m = working_df.index.isin(to_drop)
            if drop_m.any():
                n = int(drop_m.sum())
                self._append_row_events(
                    working_df,
                    list(working_df.index[drop_m]),
                    row_actions,
                    action="removed",
                    issue_type="text_quality_near_duplicate",
                    column_or_feature=primary,
                    strategy="drop_rows",
                    reason="near-duplicate of another row (text similarity)",
                )
                working_df = working_df.loc[~drop_m].copy()
                audit.append(
                    {
                        "type": "text_quality",
                        "subtype": "near_duplicates",
                        "strategy": "drop_rows",
                        "similarity_threshold": thr,
                        "affected_rows": n,
                        "mutation_applied": True,
                    }
                )

        lang_cfg = tq.get("language")
        _lang_exp = _normalize_language_expected(
            lang_cfg.get("expected") if isinstance(lang_cfg, dict) else None,
        )
        if (
            isinstance(lang_cfg, dict)
            and str(lang_cfg.get("action", "none")).lower() == "drop_rows"
            and _lang_exp
            and primary
            and primary in working_df.columns
        ):
            exp = _lang_exp
            max_r = int(lang_cfg.get("max_rows_detect", tc["language"]["max_rows_detect"]))
            mismatched, skip = language_mismatch_indices(working_df[primary], exp, max_rows=max_r)
            if mismatched and skip is None and mismatched:
                drop_m = working_df.index.isin(mismatched)
                n = int(drop_m.sum())
                self._append_row_events(
                    working_df,
                    list(working_df.index[drop_m]),
                    row_actions,
                    action="removed",
                    issue_type="text_quality_language",
                    column_or_feature=primary,
                    strategy="drop_rows",
                    reason=f"language mismatch (expected {exp!r})",
                )
                working_df = working_df.loc[~drop_m].copy()
                audit.append(
                    {
                        "type": "text_quality",
                        "subtype": "language",
                        "strategy": "drop_rows",
                        "expected": exp,
                        "affected_rows": n,
                        "mutation_applied": True,
                    }
                )

        return working_df

    def _text_quality_compare_table(self, before: dict[str, Any], after: dict[str, Any]) -> str:
        """Compact before/after table for text-quality signals used in :meth:`compare`."""

        b_raw = before if before.get("enabled") else None
        a_raw = after if after.get("enabled") else None
        b = self._text_quality_flat_metrics(b_raw)
        a = self._text_quality_flat_metrics(a_raw)
        keys: list[tuple[str, str]] = [
            ("text_any_blank_rows", "any_blank_text_rows"),
            ("text_any_short_rows", "too_short_text_rows"),
            ("text_pii_rows", "pii_rows_flagged"),
            ("text_near_duplicate_rows", "near_duplicate_partner_rows"),
            ("text_language_mismatch_rows", "language_mismatch_rows"),
            ("text_empty_or_short_rows", "empty_or_short_union"),
            ("text_exact_duplicate_rows_droppable", "exact_text_dup_droppable"),
        ]
        rows = [
            [label, int(b[k]), int(a[k]), int(a[k]) - int(b[k])] for k, label in keys
        ]
        return self._markdown_table(["metric", "before", "after", "delta"], rows)

    @staticmethod
    def _text_quality_flat_metrics(tq: dict[str, Any] | None) -> dict[str, int]:
        """Numeric summary for compare() metrics."""

        if not tq or not tq.get("enabled"):
            return {
                "text_empty_or_short_rows": 0,
                "text_any_blank_rows": 0,
                "text_any_short_rows": 0,
                "text_pii_rows": 0,
                "text_near_duplicate_rows": 0,
                "text_language_mismatch_rows": 0,
                "text_exact_duplicate_rows_droppable": 0,
            }
        es = tq.get("empty_or_short") or {}
        pii = tq.get("pii") or {}
        nd = tq.get("near_duplicates") or {}
        lang = tq.get("language") or {}
        ext = tq.get("exact_text_duplicates") or {}
        pii_rows = int(pii.get("rows_flagged") or (pii.get("breakdown") or {}).get("any_rows") or 0)
        rows_blank = int(es.get("rows_any_blank", 0))
        rows_short = int(es.get("rows_any_short_non_blank", 0))
        return {
            "text_empty_or_short_rows": int(es.get("rows_union", 0)),
            "text_any_blank_rows": rows_blank,
            "text_any_short_rows": rows_short,
            "text_pii_rows": pii_rows,
            "text_near_duplicate_rows": int(nd.get("rows_with_near_duplicate_partner", 0)),
            "text_language_mismatch_rows": int(lang.get("mismatch_rows", 0)),
            "text_exact_duplicate_rows_droppable": int(ext.get("duplicate_rows_droppable", 0)),
        }

    def _detect_missing(self, df: pd.DataFrame) -> dict[str, Any]:
        """Detect missing values, including blank text cells."""

        total_rows = len(df)
        column_stats: dict[str, dict[str, Any]] = {}
        total_missing_cells = 0
        rows_with_any_missing = pd.Series(False, index=df.index, dtype=bool)
        for column in df.columns:
            mask = self._missing_mask(df[column])
            missing_count = int(mask.sum())
            total_missing_cells += missing_count
            if total_rows:
                rows_with_any_missing = rows_with_any_missing | mask
            column_stats[column] = {
                "count": missing_count,
                "percentage": self._percentage(missing_count, total_rows),
            }
        return {
            "total_missing_cells": int(total_missing_cells),
            "rows_with_any_missing": int(rows_with_any_missing.sum()) if total_rows else 0,
            "columns": column_stats,
        }

    def _detect_duplicates(self, df: pd.DataFrame) -> dict[str, Any]:
        """Detect row, subset, and normalized text duplicates."""

        total_rows = len(df)
        if total_rows:
            full_mask, full_all_mask = _duplicate_masks_for_frame(df)
        else:
            full_mask = pd.Series(dtype=bool)
            full_all_mask = pd.Series(dtype=bool)
        subset_columns = self._infer_duplicate_subset(df)
        subset_report: dict[str, Any] = {
            "subset": subset_columns,
            "count": 0,
            "percentage": 0.0,
            "sample_indices": [],
        }
        if subset_columns:
            subset_work = df.loc[:, list(subset_columns)]
            subset_mask, subset_all_mask = _duplicate_masks_for_frame(subset_work)
            subset_report = {
                "subset": subset_columns,
                "count": int(subset_mask.sum()),
                "percentage": self._percentage(int(subset_mask.sum()), total_rows),
                "sample_indices": self._sample_indices(df.index[subset_all_mask]),
            }

        normalized_subset = self._default_normalized_duplicate_subset(df)
        normalized_report: dict[str, Any] = {
            "subset": normalized_subset,
            "count": 0,
            "percentage": 0.0,
            "sample_indices": [],
        }
        if normalized_subset:
            key_frame = self._build_duplicate_key_frame(df, normalized_subset)
            if not key_frame.empty:
                normalized_mask, normalized_all_mask = _duplicate_masks_for_frame(key_frame)
                normalized_report = {
                    "subset": normalized_subset,
                    "count": int(normalized_mask.sum()),
                    "percentage": self._percentage(int(normalized_mask.sum()), total_rows),
                    "sample_indices": self._sample_indices(df.index[normalized_all_mask]),
                }

        return {
            "full_row_duplicates": {
                "count": int(full_mask.sum()) if total_rows else 0,
                "percentage": self._percentage(int(full_mask.sum()) if total_rows else 0, total_rows),
                "sample_indices": self._sample_indices(df.index[full_all_mask]) if total_rows else [],
            },
            "subset_duplicates": subset_report,
            "normalized_text_duplicates": normalized_report,
        }

    def _detect_outliers(self, df: pd.DataFrame) -> dict[str, Any]:
        """Detect outliers via IQR and z-score."""

        feature_df, feature_meta = self._build_outlier_feature_frame(df)
        features: dict[str, dict[str, Any]] = {}
        iqr_indices: set[Any] = set()
        zscore_indices: set[Any] = set()
        for column in feature_df.columns:
            series = pd.to_numeric(feature_df[column], errors="coerce")
            iqr_stats = self._compute_iqr_outliers(series)
            zscore_stats = self._compute_zscore_outliers(series)
            iqr_indices.update(iqr_stats.pop("_indices"))
            zscore_indices.update(zscore_stats.pop("_indices"))
            feature_payload = dict(feature_meta[column])
            feature_payload["iqr"] = iqr_stats
            feature_payload["zscore"] = zscore_stats
            features[column] = feature_payload

        return {
            "features": features,
            "summary": {
                "feature_count": int(len(feature_df.columns)),
                "row_count_any_iqr": int(len(iqr_indices)),
                "row_count_any_zscore": int(len(zscore_indices)),
                "sample_indices_any_iqr": self._sample_indices(list(iqr_indices)),
                "sample_indices_any_zscore": self._sample_indices(list(zscore_indices)),
            },
        }

    def _detect_class_imbalance(self, df: pd.DataFrame) -> dict[str, Any] | None:
        """Detect class imbalance when a label column exists."""

        label_column = self._infer_label_column(df)
        if label_column is None:
            return None
        series = df[label_column]
        series = series[~self._missing_mask(series)]
        if series.empty:
            return {
                "label_column": label_column,
                "counts": {},
                "proportions": {},
                "max_min_ratio": None,
                "is_imbalanced": False,
            }

        counts = series.astype(str).value_counts(dropna=False)
        proportions = {
            label: round(float(count / counts.sum()), 4)
            for label, count in counts.items()
        }
        if len(counts) <= 1:
            ratio: float | None = None
            is_imbalanced = False
        else:
            ratio = round(float(counts.max() / counts.min()), 4)
            threshold = float(self.quality_config.get("imbalance_ratio_threshold", 3.0))
            is_imbalanced = bool(ratio >= threshold)
        return {
            "label_column": label_column,
            "counts": {label: int(count) for label, count in counts.items()},
            "proportions": proportions,
            "max_min_ratio": ratio,
            "is_imbalanced": is_imbalanced,
        }

    def _apply_missing_strategy(
        self,
        df: pd.DataFrame,
        strategy: dict[str, Any],
        audit: list[dict[str, Any]],
        row_actions: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Apply missing value strategy."""

        working_df = df.copy(deep=True)
        config = self._normalize_missing_config(strategy.get("missing"))
        if not config:
            return working_df

        drop_mask = pd.Series(False, index=working_df.index, dtype=bool)
        for column in working_df.columns:
            if column in _INTERNAL_QUALITY_COLUMNS:
                continue
            column_strategy = self._select_missing_strategy(config, working_df, column)
            if not column_strategy:
                continue
            missing_mask = self._missing_mask(working_df[column])
            missing_count = int(missing_mask.sum())
            if missing_count == 0:
                continue

            strategy_name = column_strategy.get("strategy")
            if strategy_name == "drop_rows":
                drop_mask = drop_mask | missing_mask
                self._append_row_events(
                    working_df,
                    list(working_df.index[missing_mask]),
                    row_actions,
                    action="removed",
                    issue_type="missing",
                    column_or_feature=column,
                    strategy=strategy_name,
                    reason="row dropped due to missing value in critical column",
                )
                audit.append(
                    {
                        "type": "missing",
                        "column": column,
                        "strategy": strategy_name,
                        "affected_rows": missing_count,
                        "mutation_applied": True,
                    }
                )
                continue

            fill_value = self._resolve_fill_value(working_df[column], column_strategy)
            if fill_value is _MISSING:
                audit.append(
                    {
                        "type": "missing",
                        "column": column,
                        "strategy": strategy_name,
                        "affected_rows": missing_count,
                        "status": "skipped",
                        "mutation_applied": False,
                    }
                )
                continue

            # Skip per-row events for imputation: many columns × row cap → huge lists and slow persist/compare.
            working_df.loc[missing_mask, column] = fill_value
            audit.append(
                {
                    "type": "missing",
                    "column": column,
                    "strategy": strategy_name,
                    "affected_rows": missing_count,
                    "fill_value": self._stringify_value(fill_value),
                    "mutation_applied": True,
                }
            )

        if not working_df.empty and drop_mask.any():
            removed_rows = int(drop_mask.sum())
            working_df = working_df.loc[~drop_mask].copy()
            audit.append(
                {
                    "type": "missing",
                    "strategy": "drop_rows",
                    "affected_rows": removed_rows,
                    "status": "applied",
                    "mutation_applied": True,
                }
            )
        return working_df

    def _apply_duplicate_strategy(
        self,
        df: pd.DataFrame,
        strategy: dict[str, Any],
        audit: list[dict[str, Any]],
        row_actions: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Apply duplicate handling strategy."""

        working_df = df.copy(deep=True)
        config = self._normalize_duplicate_config(strategy.get("duplicates"))
        if not config:
            return working_df

        action = config.get("action", "drop")
        keep = "last" if action == "keep_last" else "first"
        subset = config.get("subset")
        dup_cols = [c for c in working_df.columns if c not in _INTERNAL_QUALITY_COLUMNS]
        slim = working_df[dup_cols] if dup_cols else working_df
        key_frame = self._build_duplicate_key_frame(working_df, subset) if subset else slim
        if key_frame.empty:
            return working_df

        duplicate_mask = safe_duplicated(key_frame, keep=keep)
        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count == 0:
            return working_df

        if action not in {"drop", "keep_first", "keep_last"}:
            raise ValueError(f"Unsupported duplicates strategy: {action!r}.")

        self._append_row_events(
            working_df,
            list(working_df.index[duplicate_mask]),
            row_actions,
            action="removed",
            issue_type="duplicates",
            column_or_feature=", ".join(subset) if subset else "full_row",
            strategy=action,
            reason="duplicate row dropped",
        )
        working_df = working_df.loc[~duplicate_mask].copy()
        audit.append(
            {
                "type": "duplicates",
                "strategy": action,
                "subset": subset or "full_row",
                "affected_rows": duplicate_count,
                "mutation_applied": True,
            }
        )
        return working_df

    def _apply_outlier_strategy(
        self,
        df: pd.DataFrame,
        strategy: dict[str, Any],
        audit: list[dict[str, Any]],
        row_actions: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Apply outlier strategy for numeric and text-derived features."""

        working_df = df.copy(deep=True)
        config = self._normalize_outlier_config(strategy.get("outliers"))
        if not config:
            return working_df

        feature_df, feature_meta = self._build_outlier_feature_frame(working_df)
        if feature_df.empty:
            return working_df

        target_features = self._resolve_outlier_targets(
            feature_meta,
            config.get("columns"),
        )
        if not target_features:
            target_features = list(feature_df.columns)

        allowed = {
            "clip_iqr",
            "remove_iqr",
            "clip_zscore",
            "remove_zscore",
            "report_only_iqr",
            "report_only_zscore",
        }
        rows_to_remove = pd.Series(False, index=working_df.index, dtype=bool)
        for feature_name in target_features:
            feature_strategy = self._select_outlier_strategy(config, feature_name)
            if not feature_strategy:
                continue

            strategy_name = feature_strategy.get("strategy")
            if strategy_name not in allowed:
                raise ValueError(f"Unsupported outliers strategy: {strategy_name!r}.")

            meta = feature_meta[feature_name]
            source_column = meta["source_column"]
            is_derived = bool(meta["derived"])
            feature_type = str(meta.get("feature_type", "unknown"))

            series = pd.to_numeric(feature_df[feature_name], errors="coerce")
            uses_iqr = "iqr" in strategy_name
            bounds = self._compute_iqr_outliers(series) if uses_iqr else self._compute_zscore_outliers(series)
            outlier_indices = list(bounds.pop("_indices"))
            lower = bounds.get("lower_bound")
            upper = bounds.get("upper_bound")
            mask = pd.Series(False, index=working_df.index, dtype=bool)
            if outlier_indices:
                hit = pd.Index(outlier_indices).intersection(working_df.index, sort=False)
                if not hit.empty:
                    mask.loc[hit] = True

            effective_strategy = strategy_name
            report_only = strategy_name in {"report_only_iqr", "report_only_zscore"}
            if not report_only and strategy_name.startswith("clip") and is_derived:
                effective_strategy = "report_only_iqr" if uses_iqr else "report_only_zscore"
                report_only = True

            affected_ids = self._quality_row_ids_for_indices(working_df, list(working_df.index[mask]))

            if strategy_name.startswith("remove"):
                rows_to_remove = rows_to_remove | mask
                self._append_row_events(
                    working_df,
                    list(working_df.index[mask]),
                    row_actions,
                    action="removed",
                    issue_type="outliers",
                    column_or_feature=feature_name,
                    strategy=strategy_name,
                    reason=f"row removed as outlier on feature ({feature_type})",
                )
                audit.append(
                    {
                        "type": "outliers",
                        "feature": feature_name,
                        "feature_type": feature_type,
                        "source_column": source_column,
                        "derived": is_derived,
                        "strategy": strategy_name,
                        "effective_strategy": strategy_name,
                        "affected_rows": int(mask.sum()),
                        "affected_row_ids": affected_ids,
                        "mutation_applied": True,
                        "lower_bound": lower,
                        "upper_bound": upper,
                    }
                )
                continue

            if report_only:
                # report_only_* flags outliers for audit only; skip per-row row_actions here.
                # Otherwise many text-derived features × row_action cap → millions of df.loc calls and huge
                # final_row_actions (e.g. conservative strategy on wide tabular merges).
                audit.append(
                    {
                        "type": "outliers",
                        "feature": feature_name,
                        "feature_type": feature_type,
                        "source_column": source_column,
                        "derived": is_derived,
                        "strategy": effective_strategy,
                        "requested_strategy": strategy_name,
                        "affected_rows": int(mask.sum()),
                        "affected_row_ids": affected_ids,
                        "mutation_applied": False,
                        "lower_bound": lower,
                        "upper_bound": upper,
                    }
                )
                continue

            if source_column not in working_df.columns:
                audit.append(
                    {
                        "type": "outliers",
                        "feature": feature_name,
                        "feature_type": feature_type,
                        "source_column": source_column,
                        "derived": is_derived,
                        "strategy": effective_strategy,
                        "affected_rows": int(mask.sum()),
                        "affected_row_ids": affected_ids,
                        "mutation_applied": False,
                    }
                )
                continue

            col = working_df[source_column]
            num = pd.to_numeric(col, errors="coerce")
            clipped = num.clip(lower=lower, upper=upper)
            # Skip per-row row_actions for numeric clips: wide merged tables hit cap per column
            # (~row_action_id_list_max each) × many numeric cols → 500k+ events and stalled compare/persist.
            # Audit below still records bounds, strategy, and capped affected_row_ids.
            working_df[source_column] = clipped
            audit.append(
                {
                    "type": "outliers",
                    "feature": feature_name,
                    "feature_type": feature_type,
                    "source_column": source_column,
                    "derived": False,
                    "strategy": strategy_name,
                    "effective_strategy": strategy_name,
                    "affected_rows": int(mask.sum()),
                    "affected_row_ids": affected_ids,
                    "mutation_applied": True,
                    "lower_bound": lower,
                    "upper_bound": upper,
                }
            )

        if rows_to_remove.any():
            removed_rows = int(rows_to_remove.sum())
            working_df = working_df.loc[~rows_to_remove].copy()
            audit.append(
                {
                    "type": "outliers",
                    "strategy": "remove_rows",
                    "affected_rows": removed_rows,
                    "mutation_applied": True,
                }
            )
        return working_df

    def _quality_row_ids_for_indices(self, df: pd.DataFrame, indices: list[Any]) -> list[Any]:
        cap = self._row_action_id_limit()
        if "__quality_row_id" not in df.columns or cap <= 0:
            return []
        out: list[Any] = []
        for idx in indices[:cap]:
            if idx not in df.index:
                continue
            val = df.at[idx, "__quality_row_id"]
            qid = coerce_quality_row_id(val)
            if qid is not None:
                out.append(qid)
        return out

    def _standardize_text_missing_values(
        self,
        df: pd.DataFrame,
        audit: list[dict[str, Any]],
        row_actions: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Convert blank text cells into pandas missing values."""

        working_df = df.copy(deep=True)
        for column in self._infer_text_columns(working_df):
            blank_mask = self._blank_text_mask(working_df[column])
            blank_count = int(blank_mask.sum())
            if blank_count == 0:
                continue
            idxs = list(working_df.index[blank_mask])
            self._append_row_events(
                working_df,
                idxs,
                row_actions,
                action="modified",
                issue_type="missing",
                column_or_feature=column,
                strategy="blank_to_na",
                reason="blank or whitespace-only text normalized to NA",
            )
            working_df.loc[blank_mask, column] = pd.NA
            audit.append(
                {
                    "type": "missing",
                    "column": column,
                    "strategy": "blank_to_na",
                    "affected_rows": blank_count,
                    "mutation_applied": True,
                }
            )
        return working_df

    def _build_outlier_feature_frame(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
        """Build feature frame used for outlier detection."""

        feature_df = pd.DataFrame(index=df.index)
        meta: dict[str, dict[str, Any]] = {}
        numeric_columns = self._infer_numeric_columns(df)
        text_derived_cols = self._text_columns_for_derived_outliers(df)

        for column in numeric_columns:
            feature_df[column] = pd.to_numeric(df[column], errors="coerce")
            meta[column] = {
                "source_column": column,
                "derived": False,
                "feature_type": "numeric",
            }

        for column in text_derived_cols:
            series = df[column].fillna("").astype(str)
            char_name = f"char_len::{column}"
            word_name = f"word_len::{column}"
            feature_df[char_name] = series.str.len()
            # Vectorized token count; .map(len) on split lists is very slow on wide/long frames.
            feature_df[word_name] = series.str.split().str.len()
            meta[char_name] = {
                "source_column": column,
                "derived": True,
                "feature_type": "char_len",
            }
            meta[word_name] = {
                "source_column": column,
                "derived": True,
                "feature_type": "word_len",
            }

        return feature_df, meta

    def _compute_iqr_outliers(self, series: pd.Series) -> dict[str, Any]:
        """Compute IQR outliers for one numeric series."""

        clean_series = series.dropna()
        if len(clean_series) < 2:
            return {
                "lower_bound": None,
                "upper_bound": None,
                "count": 0,
                "percentage": 0.0,
                "sample_indices": [],
                "_indices": [],
            }
        q1 = float(clean_series.quantile(0.25))
        q3 = float(clean_series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        indices = list(series.index[mask.fillna(False)])
        return {
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "count": int(mask.fillna(False).sum()),
            "percentage": self._percentage(int(mask.fillna(False).sum()), len(series)),
            "sample_indices": self._sample_indices(indices),
            "_indices": indices,
        }

    def _compute_zscore_outliers(self, series: pd.Series) -> dict[str, Any]:
        """Compute z-score outliers for one numeric series."""

        clean_series = series.dropna()
        if len(clean_series) < 2:
            return {
                "lower_bound": None,
                "upper_bound": None,
                "count": 0,
                "percentage": 0.0,
                "sample_indices": [],
                "_indices": [],
            }
        mean_value = float(clean_series.mean())
        std_value = float(clean_series.std(ddof=0))
        if std_value == 0.0:
            return {
                "lower_bound": mean_value,
                "upper_bound": mean_value,
                "count": 0,
                "percentage": 0.0,
                "sample_indices": [],
                "_indices": [],
            }
        threshold = float(self.quality_config.get("zscore_threshold", 3.0))
        zscores = (series - mean_value).abs() / std_value
        mask = zscores > threshold
        indices = list(series.index[mask.fillna(False)])
        return {
            "lower_bound": round(mean_value - threshold * std_value, 4),
            "upper_bound": round(mean_value + threshold * std_value, 4),
            "count": int(mask.fillna(False).sum()),
            "percentage": self._percentage(int(mask.fillna(False).sum()), len(series)),
            "sample_indices": self._sample_indices(indices),
            "_indices": indices,
        }

    def _build_recommendations(
        self,
        missing_report: dict[str, Any],
        duplicates_report: dict[str, Any],
        outliers_report: dict[str, Any],
        imbalance_report: dict[str, Any] | None,
        text_quality_report: dict[str, Any] | None = None,
    ) -> list[str]:
        """Build lightweight deterministic recommendations."""

        recommendations: list[str] = []
        if missing_report.get("rows_with_any_missing", 0):
            recommendations.append(
                "Review columns with missing values and confirm which text fields are critical enough to drop."
            )
        if duplicates_report.get("normalized_text_duplicates", {}).get("count", 0):
            recommendations.append(
                "Use normalized text duplicate checks for prompt/text-like fields before annotation."
            )
        tq = text_quality_report or {}
        if tq.get("enabled"):
            if (tq.get("empty_or_short") or {}).get("rows_union", 0):
                recommendations.append(
                    "Some rows have empty or very short text in scanned columns; consider drop_rows under "
                    "strategy['text_quality']['empty_or_short'] or raise min_text_chars in config."
                )
            if (tq.get("exact_text_duplicates") or {}).get("duplicate_rows_droppable", 0):
                recommendations.append(
                    "Exact duplicate texts on the primary body column are present; dedupe on that column if ids differ."
                )
            pii = tq.get("pii") or {}
            if pii.get("rows_flagged", 0):
                recommendations.append(
                    "PII-like patterns (email/phone/@handle) were found; use text_quality.pii action redact (default "
                    "conservative preview) or drop_rows for strict pipelines."
                )
            nd = tq.get("near_duplicates") or {}
            if nd.get("rows_with_near_duplicate_partner", 0):
                recommendations.append(
                    "Near-duplicate posts detected; enable strict preview or strategy text_quality.near_duplicates "
                    "drop_rows (config-driven, can be expensive on large frames)."
                )
            lang = tq.get("language") or {}
            if lang.get("mismatch_rows", 0):
                recommendations.append(
                    "Language mismatches vs expected locale; tune quality.language_filter or text_checks.language."
                )
            elif lang.get("skipped_reason") == "langdetect_not_installed":
                recommendations.append(
                    "Install optional `langdetect` to enable language mismatch detection (quality.language_filter)."
                )
        if outliers_report.get("summary", {}).get("row_count_any_iqr", 0):
            recommendations.append(
                "Inspect length-based outliers to separate noise from legitimately long examples."
            )
        if (imbalance_report or {}).get("is_imbalanced"):
            recommendations.append(
                "Class imbalance is material; plan re-sampling or annotation quotas before model training."
            )
        if not recommendations:
            recommendations.append("No critical issues detected; conservative cleaning is likely sufficient.")
        return recommendations

    def _quality_report_to_markdown(self, report: QualityReport) -> str:
        """Render a markdown version of the quality report."""

        missing_rows = [
            [
                column,
                stats["count"],
                stats["percentage"],
            ]
            for column, stats in report.missing.get("columns", {}).items()
        ]
        sections = [
            "# Quality Report",
            "",
            f"- generated_at: `{report.generated_at}`",
            f"- rows: `{report.row_count}`",
            f"- columns: `{report.column_count}`",
            f"- text_columns: `{', '.join(report.text_columns) if report.text_columns else '-'}`",
            f"- numeric_columns: `{', '.join(report.numeric_columns) if report.numeric_columns else '-'}`",
            "",
            "## Missing Values",
            self._markdown_table(
                ["column", "missing_count", "missing_percentage"],
                missing_rows,
            ),
            "",
            "## Duplicates",
            self._markdown_table(
                ["type", "count", "percentage", "subset"],
                [
                    [
                        "full_row",
                        report.duplicates.get("full_row_duplicates", {}).get("count", 0),
                        report.duplicates.get("full_row_duplicates", {}).get("percentage", 0.0),
                        "-",
                    ],
                    [
                        "subset",
                        report.duplicates.get("subset_duplicates", {}).get("count", 0),
                        report.duplicates.get("subset_duplicates", {}).get("percentage", 0.0),
                        ", ".join(report.duplicates.get("subset_duplicates", {}).get("subset", []) or []) or "-",
                    ],
                    [
                        "normalized_text",
                        report.duplicates.get("normalized_text_duplicates", {}).get("count", 0),
                        report.duplicates.get("normalized_text_duplicates", {}).get("percentage", 0.0),
                        ", ".join(
                            report.duplicates.get("normalized_text_duplicates", {}).get("subset", []) or []
                        )
                        or "-",
                    ],
                ],
            ),
            "",
            "## Outliers",
            self._markdown_table(
                ["feature", "type", "iqr_count", "zscore_count", "source_column"],
                [
                    [
                        feature,
                        payload.get("feature_type"),
                        payload.get("iqr", {}).get("count", 0),
                        payload.get("zscore", {}).get("count", 0),
                        payload.get("source_column"),
                    ]
                    for feature, payload in report.outliers.get("features", {}).items()
                ],
            ),
            "",
            "## Class Imbalance",
            self._json_fenced_block(report.class_imbalance or {}),
            "",
            "## Text integrity",
            self._json_fenced_block(report.text_quality or {}),
            "",
            "## Recommendations",
            *[f"- {item}" for item in report.recommendations],
            "",
        ]
        return "\n".join(sections)

    def _comparison_bundle_to_markdown(
        self,
        conservative_comparison: ComparisonReport,
        strict_comparison: ComparisonReport,
        preview_strategies: dict[str, dict[str, Any]],
        conservative_audit: list[dict[str, Any]],
        strict_audit: list[dict[str, Any]],
    ) -> str:
        """Render comparison report for both previews."""

        return "\n".join(
            [
                "# Comparison Report",
                "",
                "## Conservative Preview",
                conservative_comparison.markdown_table,
                "",
                conservative_comparison.markdown_sections,
                "",
                "Strategy:",
                self._json_fenced_block(preview_strategies["conservative"]),
                "",
                "Audit:",
                self._json_fenced_block(conservative_audit),
                "",
                "## Strict Preview",
                strict_comparison.markdown_table,
                "",
                strict_comparison.markdown_sections,
                "",
                "Strategy:",
                self._json_fenced_block(preview_strategies["strict"]),
                "",
                "Audit:",
                self._json_fenced_block(strict_audit),
                "",
            ]
        )

    def _build_analysis_review_bundle_markdown(
        self,
        raw_df: pd.DataFrame,
        raw_df_with_ids: pd.DataFrame,
        report: QualityReport,
        recommendations: dict[str, Any],
        preview_strategies: dict[str, dict[str, Any]],
        plot_paths: dict[str, str],
    ) -> str:
        """Analysis-phase bundle without preview parquets (no fix/compare yet)."""

        why = recommendations.get("why") or {}
        outlier_indices = sorted(
            set(report.outliers.get("summary", {}).get("sample_indices_any_iqr", []))
            | set(report.outliers.get("summary", {}).get("sample_indices_any_zscore", []))
        )
        outlier_examples = raw_df.loc[raw_df.index.isin(outlier_indices)]
        return "\n".join(
            [
                "# Quality review (analysis phase)",
                "",
                "Edit `review/quality_review_decision_template.json` when you are ready to apply cleaning:",
                "",
                "- Set `approved` to `true`.",
                "- Leave `final_strategy` as `null` and set `selected_strategy_name` to `recommended` or `alternative`, **or**",
                "  set `final_strategy` to a full custom strategy object (overrides the name).",
                "",
                "## Detected issues",
                self._quality_report_to_markdown(report),
                "",
                "## Recommended strategy (tabular-friendly / conservative engine)",
                self._json_fenced_block(preview_strategies["conservative"]),
                "",
                "## Alternative strategy (stricter / strict engine)",
                self._json_fenced_block(preview_strategies["strict"]),
                "",
                "## Rationale (deterministic)",
                self._json_fenced_block(why),
                "",
                "## Example outlier rows (sample indices)",
                self._dataframe_to_markdown(outlier_examples),
                "",
                "## Plots",
                *[f"- `{name}`: `{path}`" for name, path in sorted(plot_paths.items())],
                "",
            ]
        )

    def _build_review_bundle_markdown(
        self,
        raw_df: pd.DataFrame,
        raw_df_with_ids: pd.DataFrame,
        report: QualityReport,
        conservative_df: pd.DataFrame,
        strict_df: pd.DataFrame,
        conservative_comparison: ComparisonReport,
        strict_comparison: ComparisonReport,
        conservative_audit: list[dict[str, Any]],
        strict_audit: list[dict[str, Any]],
        conservative_row_actions: list[dict[str, Any]],
        strict_row_actions: list[dict[str, Any]],
        preview_strategies: dict[str, dict[str, Any]],
        plot_paths: dict[str, str],
    ) -> str:
        """Build markdown bundle for manual review."""

        conservative_removed = self._removed_rows_by_row_ids(raw_df_with_ids, conservative_row_actions)
        strict_removed = self._removed_rows_by_row_ids(raw_df_with_ids, strict_row_actions)
        outlier_indices = sorted(
            set(report.outliers.get("summary", {}).get("sample_indices_any_iqr", []))
            | set(report.outliers.get("summary", {}).get("sample_indices_any_zscore", []))
        )
        outlier_examples = raw_df.loc[raw_df.index.isin(outlier_indices)]
        return "\n".join(
            [
                "# Quality Review Bundle",
                "",
                "Human review is required before the pipeline can continue to final cleaning.",
                "",
                "## Found Issues",
                self._quality_report_to_markdown(report),
                "",
                "## Preview Comparisons",
                "### Conservative",
                conservative_comparison.markdown_table,
                "",
                conservative_comparison.markdown_sections,
                "",
                "### Strict",
                strict_comparison.markdown_table,
                "",
                strict_comparison.markdown_sections,
                "",
                "## Preview Strategies",
                "### Conservative Strategy",
                self._json_fenced_block(preview_strategies["conservative"]),
                "",
                "### Strict Strategy",
                self._json_fenced_block(preview_strategies["strict"]),
                "",
                "## Applied Audits",
                "### Conservative Audit",
                self._json_fenced_block(conservative_audit),
                "",
                "### Strict Audit",
                self._json_fenced_block(strict_audit),
                "",
                "## Example Removed Rows",
                "### Conservative Removed Rows",
                self._dataframe_to_markdown(conservative_removed),
                "",
                f"_Removed via stable row ids: {self._stringify_value(self._extract_removed_row_ids(conservative_row_actions))}_",
                "",
                "### Strict Removed Rows",
                self._dataframe_to_markdown(strict_removed),
                "",
                f"_Removed via stable row ids: {self._stringify_value(self._extract_removed_row_ids(strict_row_actions))}_",
                "",
                "## Example Outlier Rows",
                self._dataframe_to_markdown(outlier_examples),
                "",
                "## Plot Artifacts",
                *[f"- `{name}`: `{path}`" for name, path in sorted(plot_paths.items())],
                "",
                "## Decision Reminder",
                "Edit `review/quality_review_decision_template.json`: set `approved=true`, `selected_strategy_name` to `recommended` or `alternative`, or set a custom `final_strategy`.",
                "",
            ]
        )

    @staticmethod
    def _extract_removed_row_ids(row_actions: list[dict[str, Any]]) -> list[int]:
        ids = []
        for action in row_actions:
            if action.get("action") != "removed":
                continue
            row_id = action.get("row_id")
            if row_id is None:
                continue
            try:
                ids.append(int(row_id))
            except Exception:
                continue
        return sorted(set(ids))

    def _removed_rows_by_row_ids(
        self,
        raw_df_with_ids: pd.DataFrame,
        row_actions: list[dict[str, Any]],
    ) -> pd.DataFrame:
        ids = self._extract_removed_row_ids(row_actions)
        if not ids or "__quality_row_id" not in raw_df_with_ids.columns:
            return pd.DataFrame()
        removed = raw_df_with_ids.loc[raw_df_with_ids["__quality_row_id"].isin(ids)].copy()
        return self._strip_internal_quality_columns(removed)

    def _build_review_decision_template(
        self,
        raw_df: pd.DataFrame,
        recommendations: dict[str, Any],
    ) -> dict[str, Any]:
        """Build JSON template for HITL: choose recommended/alternative or paste custom final_strategy."""

        rec = recommendations.get("recommended") or {}
        alt = recommendations.get("alternative") or {}
        why = recommendations.get("why") or {}
        return {
            "schema_version": 2,
            "approved": False,
            "selected_strategy_name": "recommended",
            "available_strategies": {
                "recommended": rec,
                "alternative": alt,
            },
            "final_strategy": None,
            "review_notes": "",
            "reviewed_by": "",
            "reviewed_at": None,
            "generated_at": self._now_iso(),
            "row_count": int(len(raw_df)),
            "why": why,
            "available_preview_strategies": {
                "conservative": rec,
                "strict": alt,
            },
        }

    def _export_plots(
        self,
        df: pd.DataFrame,
        report: QualityReport,
    ) -> dict[str, str]:
        """Export plot artifacts for all supported issue types."""

        plots: dict[str, str] = {}
        plots["plot_missing_values"] = self.storage.save_figure(
            "reports/quality/plots/missing_values.png",
            self._build_missing_plot(report),
        )
        plots["plot_duplicates"] = self.storage.save_figure(
            "reports/quality/plots/duplicates.png",
            self._build_duplicates_plot(report),
        )
        plots["plot_outliers"] = self.storage.save_figure(
            "reports/quality/plots/outliers.png",
            self._build_outliers_plot(report),
        )
        plots["plot_class_imbalance"] = self.storage.save_figure(
            "reports/quality/plots/class_imbalance.png",
            self._build_class_imbalance_plot(df, report),
        )
        return plots

    def _build_missing_plot(self, report: QualityReport) -> Any:
        """Build missing-values plot."""

        fig, ax = plt.subplots(figsize=(8, 4))
        items = report.missing.get("columns", {})
        columns = list(items.keys())
        counts = [stats["count"] for stats in items.values()]
        if counts:
            positions = list(range(len(columns)))
            ax.bar(positions, counts, color="#4C78A8")
            ax.set_xticks(positions)
            ax.set_xticklabels(columns, rotation=45, ha="right")
        else:
            ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.set_title("Missing values by column")
        ax.set_ylabel("Missing count")
        fig.tight_layout()
        return fig

    def _build_duplicates_plot(self, report: QualityReport) -> Any:
        """Build duplicates summary plot."""

        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["full_row", "subset", "normalized_text"]
        counts = [
            report.duplicates.get("full_row_duplicates", {}).get("count", 0),
            report.duplicates.get("subset_duplicates", {}).get("count", 0),
            report.duplicates.get("normalized_text_duplicates", {}).get("count", 0),
        ]
        ax.bar(labels, counts, color=["#72B7B2", "#F58518", "#E45756"])
        ax.set_title("Duplicate rows detected")
        ax.set_ylabel("Duplicate count")
        fig.tight_layout()
        return fig

    def _build_outliers_plot(self, report: QualityReport) -> Any:
        """Build outliers plot."""

        fig, ax = plt.subplots(figsize=(10, 4))
        features = list(report.outliers.get("features", {}).items())
        if features:
            names = [name for name, _ in features]
            iqr_counts = [payload.get("iqr", {}).get("count", 0) for _, payload in features]
            zscore_counts = [payload.get("zscore", {}).get("count", 0) for _, payload in features]
            positions = list(range(len(names)))
            ax.bar([pos - 0.2 for pos in positions], iqr_counts, width=0.4, label="IQR")
            ax.bar([pos + 0.2 for pos in positions], zscore_counts, width=0.4, label="z-score")
            ax.set_xticks(positions)
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No outlier features", ha="center", va="center")
        ax.set_title("Outliers by feature")
        ax.set_ylabel("Flagged rows")
        fig.tight_layout()
        return fig

    def _build_class_imbalance_plot(self, df: pd.DataFrame, report: QualityReport) -> Any:
        """Build class distribution plot."""

        fig, ax = plt.subplots(figsize=(6, 4))
        imbalance = report.class_imbalance
        if imbalance and imbalance.get("counts"):
            labels = list(imbalance["counts"].keys())
            counts = list(imbalance["counts"].values())
            positions = list(range(len(labels)))
            ax.bar(positions, counts, color="#54A24B")
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha="right")
        else:
            label_column = self._infer_label_column(df)
            message = "No label column detected" if label_column is None else "No class counts available"
            ax.text(0.5, 0.5, message, ha="center", va="center")
        ax.set_title("Class distribution")
        ax.set_ylabel("Rows")
        fig.tight_layout()
        return fig

    def _metric_row(self, name: str, before: Any, after: Any) -> dict[str, Any]:
        """Build one comparison metric row."""

        delta = self._metric_delta(before, after)
        return {"metric": name, "before": before, "after": after, "delta": delta}

    def _metric_delta(self, before: Any, after: Any) -> Any:
        """Return a delta when values are numeric."""

        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            return round(float(after - before), 4)
        return "n/a"

    def _metrics_to_markdown_table(self, metrics: list[dict[str, Any]]) -> str:
        """Render metric rows as markdown table."""

        rows = [[item["metric"], item["before"], item["after"], item["delta"]] for item in metrics]
        return self._markdown_table(["metric", "before", "after", "delta"], rows)

    def _markdown_table(self, headers: list[str], rows: list[list[Any]]) -> str:
        """Render a small markdown table."""

        header_row = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join("---" for _ in headers) + " |"
        body = [
            "| " + " | ".join(self._stringify_value(cell) for cell in row) + " |"
            for row in rows
        ]
        return "\n".join([header_row, separator, *body]) if body else "\n".join([header_row, separator])

    def _json_fenced_block(self, payload: Any) -> str:
        """Render JSON fenced code block."""

        return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"

    def _dataframe_to_markdown(self, df: pd.DataFrame, limit: int = 5) -> str:
        """Render a small dataframe sample as markdown."""

        if df.empty:
            return "_No rows._"
        limited_df = df.head(limit).copy()
        limited_df.insert(0, "_index", [self._stringify_value(value) for value in limited_df.index])
        headers = list(limited_df.columns)
        rows = [[limited_df.iloc[row_index][column] for column in headers] for row_index in range(len(limited_df))]
        return self._markdown_table(headers, rows)

    def _fallback_strategy_explanation(
        self,
        report: dict[str, Any],
        task_description: str,
    ) -> str:
        """Return deterministic explanation if LLM is unavailable."""

        missing = report.get("missing", {}).get("rows_with_any_missing", 0)
        duplicates = report.get("duplicates", {}).get("normalized_text_duplicates", {}).get("count", 0)
        outliers = report.get("outliers", {}).get("summary", {}).get("row_count_any_iqr", 0)
        imbalance = report.get("class_imbalance") or {}
        tq = report.get("text_quality") or {}
        tq_line = ""
        if tq.get("enabled"):
            flat = DataQualityAgent._text_quality_flat_metrics(tq)
            tq_line = (
                f"- Text integrity: empty/short union ~{flat['text_empty_or_short_rows']} "
                f"(blank ~{flat['text_any_blank_rows']}, short ~{flat['text_any_short_rows']}), "
                f"PII ~{flat['text_pii_rows']}, near-dup ~{flat['text_near_duplicate_rows']}, "
                f"language ~{flat['text_language_mismatch_rows']}\n"
            )
        return (
            f"Task: {task_description}\n"
            f"- Missing-value rows: {missing}\n"
            f"- Normalized text duplicates: {duplicates}\n"
            f"- IQR outlier rows: {outliers}\n"
            f"- Class imbalance: {imbalance.get('is_imbalanced', False)}\n"
            f"{tq_line}"
            "Recommended approach: review the conservative preview first, then switch to the strict preview only "
            "for columns confirmed as critical by a human reviewer."
        )

    def _infer_text_columns(self, df: pd.DataFrame) -> list[str]:
        """Infer text columns from dataframe dtypes (excludes numeric-like object columns)."""

        threshold = float(self.quality_config.get("numeric_coercion_min_ratio", 0.8))
        text_columns: list[str] = []
        for column in df.columns:
            if column in _INTERNAL_QUALITY_COLUMNS:
                continue
            series = df[column]
            if not (pd.api.types.is_string_dtype(series) or series.dtype == object):
                continue
            if self._column_numeric_parse_ratio(series) >= threshold:
                continue
            text_columns.append(column)
        return text_columns

    def _infer_numeric_columns(self, df: pd.DataFrame) -> list[str]:
        """Infer numeric columns: native numeric dtypes plus numeric-like object/string columns."""

        threshold = float(self.quality_config.get("numeric_coercion_min_ratio", 0.8))
        numeric_columns: list[str] = []
        for column in df.columns:
            if column in _INTERNAL_QUALITY_COLUMNS:
                continue
            series = df[column]
            if pd.api.types.is_numeric_dtype(series):
                numeric_columns.append(column)
                continue
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                if self._column_numeric_parse_ratio(series) >= threshold:
                    numeric_columns.append(column)
        return numeric_columns

    def _infer_label_column(self, df: pd.DataFrame) -> str | None:
        """Infer label column from config or common defaults."""

        configured_label = config_column_name(self.quality_config.get("label_column"))
        if configured_label is not None and configured_label in df.columns:
            return configured_label
        for candidate in DEFAULT_LABEL_HINTS:
            if candidate in df.columns:
                return candidate
        return None

    def _infer_duplicate_subset(self, df: pd.DataFrame) -> list[str]:
        """Infer subset columns for duplicate detection.

        Order: explicit config (existing columns) → stable id-like keys → text hints.
        In ``auto``/``tabular`` modes, text-like fallback uses *meaningful* text columns only
        so empty ``text``/``prompt`` placeholders are not primary duplicate keys.
        """

        subset = [column for column in iter_config_column_names(self.quality_config.get("duplicate_subset")) if column in df.columns]
        if subset:
            return subset
        for name in _PREFERRED_DUPLICATE_KEY_COLUMNS:
            if name in df.columns:
                return [name]
        mode = self._dataset_mode()
        if mode == "text":
            for candidate in DEFAULT_TEXT_DUPLICATE_HINTS:
                if candidate in df.columns:
                    return [candidate]
            return []
        meaningful = self._infer_meaningful_text_columns(df)
        for candidate in DEFAULT_TEXT_DUPLICATE_HINTS:
            if candidate in meaningful:
                return [candidate]
        return []

    def _default_normalized_duplicate_subset(self, df: pd.DataFrame) -> list[str]:
        """Infer normalized duplicate subset."""

        configured_columns = iter_config_column_names(self.quality_config.get("normalized_text_columns"))
        fallback_text = self._text_columns_for_duplicate_and_critical_fallback(df)
        normalized_columns = [
            f"normalized_{column}"
            for column in configured_columns
            if column in df.columns and column in fallback_text
        ]
        if normalized_columns:
            return normalized_columns

        subset = self._infer_duplicate_subset(df)
        if subset and all(column in fallback_text for column in subset):
            return [f"normalized_{column}" for column in subset]

        for candidate in DEFAULT_TEXT_DUPLICATE_HINTS:
            if candidate in df.columns and candidate in fallback_text:
                return [f"normalized_{candidate}"]
        return []

    def _infer_critical_text_columns(self, df: pd.DataFrame) -> list[str]:
        """Infer critical text columns for preview strategies."""

        configured = iter_config_column_names(self.quality_config.get("critical_text_columns"))
        columns = [column for column in configured if column in df.columns]
        if columns:
            return columns
        fallback_text = self._text_columns_for_duplicate_and_critical_fallback(df)
        inferred = [
            column
            for column in DEFAULT_TEXT_DUPLICATE_HINTS
            if column in df.columns and column in fallback_text
        ]
        if inferred:
            return inferred[:2]
        return (fallback_text[:1] if fallback_text else [])

    def _default_text_outlier_features(self, df: pd.DataFrame) -> list[str]:
        """Infer default text-derived features used in preview strategies."""

        text_columns = self._text_columns_for_derived_outliers(df)
        if not text_columns:
            return []
        return [f"char_len::{column}" for column in text_columns] + [
            f"word_len::{column}" for column in text_columns
        ]

    def _build_duplicate_key_frame(
        self,
        df: pd.DataFrame,
        subset: list[str] | None,
    ) -> pd.DataFrame:
        """Build the key frame used to detect duplicate rows."""

        if not subset:
            return df.copy()
        key_frame = pd.DataFrame(index=df.index)
        for column in subset:
            if column in df.columns:
                key_frame[column] = df[column]
                continue
            if column.startswith("normalized_"):
                source_column = column[len("normalized_") :]
                if source_column in df.columns:
                    key_frame[column] = self._normalize_text_series(df[source_column])
        return key_frame

    def _normalize_text_series(self, series: pd.Series) -> pd.Series:
        """Normalize text for duplicate detection."""

        normalized = series.fillna("").astype(str).str.strip().str.lower()
        normalized = normalized.str.replace(r"\s+", " ", regex=True)
        normalized = normalized.mask(normalized.eq(""), pd.NA)
        return normalized

    def _missing_mask(self, series: pd.Series) -> pd.Series:
        """Return boolean mask for missing values."""

        return series.isna() | self._blank_text_mask(series)

    def _blank_text_mask(self, series: pd.Series) -> pd.Series:
        """Return boolean mask for blank strings."""

        if not (pd.api.types.is_string_dtype(series) or series.dtype == object):
            return pd.Series(False, index=series.index, dtype=bool)
        non_null = series.notna()
        normalized = series.where(non_null, "").astype(str)
        return non_null & normalized.str.strip().eq("")

    def _empty_text_mask(self, series: pd.Series) -> pd.Series:
        """Blank / whitespace-only **or** NA (matches post-``blank_to_na`` stage for text_quality drops)."""

        if not (pd.api.types.is_string_dtype(series) or series.dtype == object):
            return series.isna()
        return self._blank_text_mask(series) | series.isna()

    def _normalize_missing_config(self, raw_config: Any) -> dict[str, Any]:
        """Normalize missing strategy config."""

        if raw_config is None:
            return {}
        if isinstance(raw_config, str):
            return {"default": {"strategy": raw_config}, "per_column": {}}
        if not isinstance(raw_config, dict):
            raise ValueError("Missing strategy config must be a string or dict.")

        config: dict[str, Any] = {"per_column": dict(raw_config.get("per_column", {}))}
        if "strategy" in raw_config:
            config["default"] = {"strategy": raw_config["strategy"], **{
                key: value for key, value in raw_config.items() if key not in {"strategy", "per_column"}
            }}
        elif "default" in raw_config:
            config["default"] = raw_config["default"]
        if "default_numeric" in raw_config:
            config["default_numeric"] = raw_config["default_numeric"]
        if "default_categorical" in raw_config:
            config["default_categorical"] = raw_config["default_categorical"]

        special_keys = {
            "default",
            "default_numeric",
            "default_categorical",
            "per_column",
            "strategy",
        }
        for key, value in raw_config.items():
            if key not in special_keys:
                config["per_column"][key] = value
        return config

    def _normalize_duplicate_config(self, raw_config: Any) -> dict[str, Any]:
        """Normalize duplicates strategy config."""

        if raw_config is None:
            return {}
        if isinstance(raw_config, str):
            return {"action": raw_config}
        if not isinstance(raw_config, dict):
            raise ValueError("Duplicates strategy config must be a string or dict.")
        action = raw_config.get("action") or raw_config.get("strategy") or "drop"
        subset = raw_config.get("subset")
        return {"action": action, "subset": subset}

    def _normalize_outlier_config(self, raw_config: Any) -> dict[str, Any]:
        """Normalize outlier strategy config."""

        if raw_config is None:
            return {}
        if isinstance(raw_config, str):
            return {"default": {"strategy": raw_config}, "per_column": {}}
        if not isinstance(raw_config, dict):
            raise ValueError("Outliers strategy config must be a string or dict.")
        config: dict[str, Any] = {
            "per_column": dict(raw_config.get("per_column", {})),
            "columns": raw_config.get("columns"),
        }
        if "strategy" in raw_config:
            config["default"] = {"strategy": raw_config["strategy"]}
        elif "default" in raw_config:
            config["default"] = raw_config["default"]
        return config

    def _select_missing_strategy(
        self,
        config: dict[str, Any],
        df: pd.DataFrame,
        column: str,
    ) -> dict[str, Any] | None:
        """Resolve missing strategy for a column."""

        per_column = config.get("per_column", {})
        if column in per_column:
            return self._ensure_strategy_dict(per_column[column])
        if column in self._infer_numeric_columns(df) and config.get("default_numeric"):
            return self._ensure_strategy_dict(config["default_numeric"])
        if column not in self._infer_numeric_columns(df) and config.get("default_categorical"):
            return self._ensure_strategy_dict(config["default_categorical"])
        if config.get("default"):
            return self._ensure_strategy_dict(config["default"])
        return None

    def _select_outlier_strategy(
        self,
        config: dict[str, Any],
        feature_name: str,
    ) -> dict[str, Any] | None:
        """Resolve outlier strategy for a feature."""

        per_column = config.get("per_column", {})
        if feature_name in per_column:
            return self._ensure_strategy_dict(per_column[feature_name])
        return self._ensure_strategy_dict(config["default"]) if config.get("default") else None

    def _resolve_fill_value(
        self,
        series: pd.Series,
        strategy: dict[str, Any],
    ) -> Any:
        """Resolve imputation value for a series."""

        strategy_name = strategy.get("strategy")
        clean_series = series[~self._missing_mask(series)]
        th = float(self.quality_config.get("numeric_coercion_min_ratio", 0.8))
        if strategy_name in {"mean", "median"}:
            if clean_series.empty:
                return _MISSING
            if not pd.api.types.is_numeric_dtype(series) and self._column_numeric_parse_ratio(series) < th:
                return _MISSING
            nums = pd.to_numeric(clean_series, errors="coerce")
            if nums.notna().sum() == 0:
                return _MISSING
            if strategy_name == "mean":
                return float(nums.mean())
            return float(nums.median())
        if strategy_name == "mode":
            if clean_series.empty:
                return _MISSING
            modes = clean_series.mode(dropna=True)
            return modes.iloc[0] if not modes.empty else _MISSING
        if strategy_name == "constant":
            return strategy.get("value", strategy.get("constant", _MISSING))
        raise ValueError(f"Unsupported missing strategy: {strategy_name!r}.")

    def _resolve_outlier_targets(
        self,
        feature_meta: dict[str, dict[str, Any]],
        configured_columns: list[str] | None,
    ) -> list[str]:
        """Resolve configured columns to feature names."""

        if not configured_columns:
            return []
        targets: list[str] = []
        for column in configured_columns:
            if column in feature_meta:
                targets.append(column)
                continue
            targets.extend(
                feature_name
                for feature_name, meta in feature_meta.items()
                if meta["source_column"] == column
            )
        return sorted(set(targets))

    def _ensure_strategy_dict(self, value: Any) -> dict[str, Any]:
        """Wrap strategy strings into dicts."""

        if isinstance(value, str):
            return {"strategy": value}
        if isinstance(value, dict):
            return value
        raise ValueError("Strategy value must be a string or dict.")

    def _load_decision(self, decision: str | Path | dict[str, Any]) -> dict[str, Any]:
        """Load decision payload from path or dict."""

        if isinstance(decision, dict):
            return dict(decision)
        path = Path(decision)
        try:
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Quality decision file is not valid JSON ({path}): {exc.msg} at line {exc.lineno} col {exc.colno}"
            ) from exc
        except OSError as exc:
            raise ValueError(f"Could not read quality decision file ({path}): {exc}") from exc

    @staticmethod
    def _load_config(config: str | dict[str, Any] | None) -> dict[str, Any]:
        """Load config from dict, JSON, or YAML file."""

        if config is None:
            return {}
        if isinstance(config, dict):
            return dict(config)
        config_path = Path(config)
        with config_path.open("r", encoding="utf-8") as file:
            if config_path.suffix.lower() in {".yaml", ".yml"}:
                import yaml

                return yaml.safe_load(file) or {}
            return json.load(file)

    @staticmethod
    def _sample_indices(index_values: Any, limit: int = 10) -> list[Any]:
        """Return a stable list of sample indices."""

        if isinstance(index_values, pd.Index):
            values = list(index_values)
        else:
            values = list(index_values)
        return [DataQualityAgent._stringify_index(value) for value in values[:limit]]

    @staticmethod
    def _stringify_index(value: Any) -> Any:
        """Convert index values into JSON-safe values."""

        return value.item() if hasattr(value, "item") else value

    @staticmethod
    def _stringify_value(value: Any) -> str:
        """Convert arbitrary values into markdown-safe strings."""

        if isinstance(value, float):
            return f"{value:.4f}"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        if value is None:
            return "null"
        try:
            if pd.api.types.is_scalar(value):
                try:
                    if pd.isna(value):
                        return "NA"
                except (ValueError, TypeError):
                    return str(value)
                return str(value)
        except Exception:
            pass
        nv = normalize_scalar_like(value)
        return "NA" if nv is None else str(nv)

    @staticmethod
    def _percentage(count: int, total: int) -> float:
        """Return percentage in 0-100 scale."""

        if total == 0:
            return 0.0
        return round(float(count / total * 100.0), 4)

    @staticmethod
    def _now_iso() -> str:
        """Return current UTC timestamp."""

        return datetime.now(timezone.utc).isoformat()


_MISSING = object()
