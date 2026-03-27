from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from agents.annotation.labelstudio import (
    build_labelstudio_config,
    build_labelstudio_tasks,
    read_labelstudio_export_path,
)
from agents.annotation.quality import compute_quality_metrics, render_annotation_report
from agents.annotation.review import (
    merge_human_annotations,
    prepare_audit_sample,
    prepare_review_queue,
)
from agents.annotation.spec_generator import build_annotation_spec
from agents.annotation.task_configs import get_task_config
from agents.annotation.text_labeler import DEFAULT_ZERO_SHOT_MODEL, build_text_labeler


class _AnnotationArtifactStorage:
    """Small file-backed storage for annotation artifacts."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def save_dataframe(self, relative_path: str | Path, df: pd.DataFrame) -> str:
        """Save dataframe under the project root."""

        path = self.project_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".csv":
            df.to_csv(path, index=False)
        else:
            df.to_parquet(path, index=False)
        return str(path.resolve())

    def save_json(self, relative_path: str | Path, payload: Any) -> str:
        """Save JSON payload under the project root."""

        path = self.project_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        return str(path.resolve())

    def save_text(self, relative_path: str | Path, content: str) -> str:
        """Save text payload under the project root."""

        path = self.project_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path.resolve())


class AnnotationAgent:
    """Minimal text-first annotation stage with HITL support."""

    def __init__(
        self,
        modality: str = "text",
        task: str = "safety_classification",
        threshold: float = 0.70,
        margin_threshold: float | None = None,
        text_column: str | None = None,
        id_column: str | None = None,
        config: str | dict[str, Any] | None = None,
        backend: str = "auto",
        model_name: str = DEFAULT_ZERO_SHOT_MODEL,
    ) -> None:
        self.config = self._load_config(config)
        annotation_config = dict(self.config.get("annotation", {}))
        configured_text_column = text_column or annotation_config.get("text_column")
        configured_id_column = id_column or annotation_config.get("id_column")
        modality_eff = str(annotation_config.get("modality", modality)).strip()
        task_name_cfg = annotation_config.get("task_name")
        if task_name_cfg is not None and str(task_name_cfg).strip():
            task_resolved = str(task_name_cfg).strip()
        else:
            task_resolved = str(task).strip() if task is not None else "safety_classification"

        self.task_config = get_task_config(
            task=task_resolved,
            modality=modality_eff,
            text_column=configured_text_column,
            id_column=configured_id_column,
            annotation_section=annotation_config,
        )
        self.modality = modality_eff
        self.task = task_resolved
        if annotation_config.get("confidence_threshold") is not None:
            self.threshold = float(annotation_config["confidence_threshold"])
        else:
            self.threshold = float(threshold if threshold is not None else self.task_config.threshold)
        self.task_config.threshold = self.threshold
        self.margin_threshold = float(
            margin_threshold
            if margin_threshold is not None
            else annotation_config.get("margin_threshold", self.task_config.margin_threshold)
        )
        _rq_cap = annotation_config.get("review_queue_max_rows")
        if _rq_cap is None or _rq_cap == "":
            self.review_queue_max_rows: int | None = None
        else:
            _v = int(_rq_cap)
            self.review_queue_max_rows = _v if _v > 0 else None
        self.last_review_queue_meta: dict[str, Any] = {}
        self.project_root = Path(
            annotation_config.get("project_root")
            or Path(__file__).resolve().parents[1]
        )
        self.storage = _AnnotationArtifactStorage(self.project_root)
        self.labeler = build_text_labeler(
            self.task_config,
            backend=annotation_config.get("backend", backend),
            model_name=annotation_config.get("model_name", model_name),
        )
        self.last_artifacts: dict[str, str] = {}

    def auto_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-label a dataframe and persist the default parquet artifact."""

        text_column = self._resolve_text_column(df)
        id_column = self._resolve_id_column(df)
        labeled_df = df.copy(deep=True)
        if id_column not in labeled_df.columns:
            labeled_df[id_column] = [f"row_{index}" for index in range(len(labeled_df))]

        results = [self.labeler.label_text(value) for value in labeled_df[text_column].tolist()]
        labeled_df["annotation_id"] = labeled_df[id_column].astype(str)
        labeled_df["auto_label"] = [result.label for result in results]
        labeled_df["confidence"] = [result.confidence for result in results]
        labeled_df["class_scores"] = [result.class_scores for result in results]
        labeled_df["margin"] = [result.margin for result in results]
        labeled_df["entropy"] = [result.entropy for result in results]
        labeled_df["needs_review"] = [
            self._needs_review(result, text_value)
            for result, text_value in zip(results, labeled_df[text_column].tolist())
        ]
        labeled_df["review_reason"] = [
            self._review_reason(result, text_value)
            for result, text_value in zip(results, labeled_df[text_column].tolist())
        ]
        labeled_df["task"] = self.task_config.name
        labeled_df["label_source"] = [f"auto:{result.backend_name}" for result in results]
        labeled_df["annotator_version"] = self.task_config.model_version

        auto_labeled_path = self.storage.save_dataframe(
            "data/labeled/auto_labeled.parquet",
            labeled_df,
        )
        self.last_artifacts["auto_labeled_parquet"] = auto_labeled_path
        return labeled_df

    def generate_spec(self, df: pd.DataFrame, task: str | None = None) -> str:
        """Generate and save the annotation markdown spec."""

        # When switching task for ad-hoc spec generation, do not inject current YAML labels
        # (they would force config-driven mode and override the requested task).
        task_config = (
            self.task_config
            if task in {None, self.task_config.name}
            else get_task_config(task, modality=self.modality, annotation_section=None)
        )
        text_column = self._resolve_text_column(df, task_config=task_config)
        content = build_annotation_spec(
            df=df,
            task_config=task_config,
            text_column=text_column,
            threshold=self.threshold,
            margin_threshold=self.margin_threshold,
        )
        path = self.storage.save_text("reports/annotation_spec.md", content)
        self.last_artifacts["annotation_spec_md"] = path
        return path

    def check_quality(
        self,
        df: pd.DataFrame,
        human_label_column: str = "human_label",
    ) -> dict[str, object]:
        """Compute annotation metrics and save the markdown report."""

        metrics = compute_quality_metrics(
            df=df,
            task_config=self.task_config,
            human_label_column=human_label_column,
            threshold=self.threshold,
        )
        report_content = render_annotation_report(metrics, self.task_config)
        path = self.storage.save_text("reports/annotation_report.md", report_content)
        self.last_artifacts["annotation_report_md"] = path
        return metrics

    def export_to_labelstudio(
        self,
        df: pd.DataFrame,
        include_predictions: bool = True,
        audit_include_predictions: bool = False,
        audit_n_per_class: int = 25,
    ) -> dict[str, str]:
        """Export full, review, and audit datasets for Label Studio."""

        text_column = self._resolve_text_column(df)
        id_column = self._resolve_id_column(df)
        export_df = df.copy(deep=True)
        if id_column not in export_df.columns:
            export_df[id_column] = [f"row_{index}" for index in range(len(export_df))]
        export_df["annotation_id"] = export_df[id_column].astype(str)

        review_queue = prepare_review_queue(export_df)
        audit_sample = prepare_audit_sample(
            export_df,
            n_per_class=audit_n_per_class,
        )

        review_tasks = build_labelstudio_tasks(
            review_queue,
            self.task_config,
            text_column=text_column,
            id_column="annotation_id",
            include_predictions=include_predictions,
        )
        xml_body = build_labelstudio_config(self.task_config)
        artifacts = {
            "labelstudio_config_xml": self.storage.save_text(
                "reports/labelstudio_config.xml",
                xml_body,
            ),
            "review_queue_csv": self.storage.save_dataframe(
                "data/review/review_queue.csv",
                review_queue,
            ),
            "audit_sample_csv": self.storage.save_dataframe(
                "data/review/audit_sample.csv",
                audit_sample,
            ),
            "review_queue_labelstudio_json": self.storage.save_json(
                "data/review/review_queue_labelstudio.json",
                review_tasks,
            ),
            "audit_sample_labelstudio_json": self.storage.save_json(
                "data/review/audit_sample_labelstudio.json",
                build_labelstudio_tasks(
                    audit_sample,
                    self.task_config,
                    text_column=text_column,
                    id_column="annotation_id",
                    include_predictions=audit_include_predictions,
                ),
            ),
            "full_dataset_labelstudio_json": self.storage.save_json(
                "data/review/full_dataset_labelstudio.json",
                build_labelstudio_tasks(
                    export_df,
                    self.task_config,
                    text_column=text_column,
                    id_column="annotation_id",
                    include_predictions=include_predictions,
                ),
            ),
        }
        ls_block = dict(self.config.get("label_studio") or {})
        task_fn = Path(str(ls_block.get("task_file") or "labelstudio_import.json")).name
        cfg_fn = Path(str(ls_block.get("config_file") or "label_config.xml")).name
        if task_fn != Path("review_queue_labelstudio.json").name:
            artifacts["labelstudio_import_json"] = self.storage.save_json(
                f"data/review/{task_fn}",
                review_tasks,
            )
        else:
            artifacts["labelstudio_import_json"] = artifacts["review_queue_labelstudio_json"]
        if cfg_fn != Path("labelstudio_config.xml").name:
            artifacts["label_config_xml"] = self.storage.save_text(f"reports/{cfg_fn}", xml_body)
        else:
            artifacts["label_config_xml"] = artifacts["labelstudio_config_xml"]
        self.last_artifacts.update(artifacts)
        return artifacts

    def prepare_review_queue(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the low-confidence review queue (worst-first), optionally capped."""

        review_df = prepare_review_queue(df)
        full_n = len(review_df)
        cap = self.review_queue_max_rows
        if cap is not None and full_n > cap:
            review_df = review_df.head(int(cap)).copy()
        self.last_review_queue_meta = {
            "eligible_for_review": full_n,
            "exported_to_queue": len(review_df),
            "cap": cap,
            "truncated": bool(cap is not None and full_n > int(cap)),
        }
        return review_df

    def prepare_audit_sample(
        self,
        df: pd.DataFrame,
        n_per_class: int = 25,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Return a stratified audit sample."""

        return prepare_audit_sample(
            df,
            n_per_class=n_per_class,
            random_state=random_state,
        )

    def merge_human_annotations(
        self,
        auto_df: pd.DataFrame,
        human_df: pd.DataFrame,
        human_label_column: str = "human_label",
    ) -> pd.DataFrame:
        """Merge human labels back into the auto-labeled dataset."""

        return merge_human_annotations(
            auto_df,
            human_df,
            id_column="annotation_id",
            human_label_column=human_label_column,
        )

    def import_from_labelstudio(
        self,
        path: str | Path,
        *,
        strict_labels: bool = True,
        on_duplicate_id: str = "error",
    ) -> pd.DataFrame:
        """Parse a Label Studio JSON export into a dataframe for ``merge_human_annotations``."""

        allowed = frozenset(self.task_config.labels) if strict_labels else None
        return read_labelstudio_export_path(
            Path(path),
            allowed_labels=allowed,
            on_duplicate_id=on_duplicate_id,
        )

    def _resolve_text_column(
        self,
        df: pd.DataFrame,
        task_config: Any | None = None,
    ) -> str:
        """Resolve the text column using task config and common defaults."""

        active_task_config = task_config or self.task_config
        candidates: list[str] = []
        if active_task_config.text_column:
            candidates.append(active_task_config.text_column)
        candidates.extend(active_task_config.text_column_candidates)
        for column in candidates:
            if column in df.columns:
                return column
        raise ValueError(
            "Could not infer text column. "
            f"Tried: {candidates or ['prompt', 'text', 'query', 'content']}"
        )

    def _resolve_id_column(self, df: pd.DataFrame) -> str:
        """Resolve an existing id column or use the default synthetic id."""

        if self.task_config.id_column and self.task_config.id_column in df.columns:
            return self.task_config.id_column
        for candidate in ["id", "record_id", "sample_id", "annotation_id"]:
            if candidate in df.columns:
                return candidate
        return "annotation_id"

    def _needs_review(self, result: Any, text_value: object) -> bool:
        """Return whether the example needs human review."""

        return bool(
            result.confidence < self.threshold
            or result.margin < self.margin_threshold
            or bool(result.warnings)
            or not str(text_value or "").strip()
        )

    def _review_reason(self, result: Any, text_value: object) -> str:
        """Return a semicolon-separated review reason string."""

        reasons: list[str] = []
        if result.confidence < self.threshold:
            reasons.append("low_confidence")
        if result.margin < self.margin_threshold:
            reasons.append("small_margin")
        if not str(text_value or "").strip():
            reasons.append("empty_text")
        reasons.extend(result.warnings)
        return "; ".join(sorted(set(reasons)))

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
