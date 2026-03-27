from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from pipeline.train_baseline import render_metrics_md, resolve_text_column, train_eval_persist


@dataclass
class TrainRunResult:
    model_info_path: str
    metrics_path: str
    artifact_paths: Dict[str, str]


def _resolve_label_column(df: pd.DataFrame) -> str:
    for candidate in ("final_label", "human_label", "label", "auto_label"):
        if candidate in df.columns:
            return candidate
    raise ValueError("TRAIN stage needs a label column (final_label/human_label/label/auto_label).")


def _prefer_human_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Use reviewed labels when present; fill gaps from auto_label for the training target column."""

    if label_col != "final_label":
        return df
    if "final_label" not in df.columns or "auto_label" not in df.columns:
        return df
    out = df.copy()
    fl = out["final_label"]
    al = out["auto_label"]
    empty = fl.isna() | (fl.astype(str).str.strip() == "")
    out.loc[empty, "final_label"] = al.loc[empty]
    return out


def run_train_stage(
    *,
    stage_dir: Path,
    dataset_df: pd.DataFrame,
    config_payload: Dict[str, Any] | None = None,
) -> TrainRunResult:
    """Train TF-IDF + LogisticRegression baseline; persist model, metrics, and reports."""

    cfg = dict(config_payload or {})
    training_cfg = dict(cfg.get("training") or {})
    task = str(training_cfg.get("task_type", "text_classification")).strip().lower()
    if task != "text_classification":
        raise ValueError(
            f"TRAIN stage currently supports only task_type=text_classification; got {task!r}. "
            "Set training.task_type: text_classification in config."
        )

    label_col = _resolve_label_column(dataset_df)
    dataset_df = _prefer_human_labels(dataset_df, label_col)
    text_col = resolve_text_column(dataset_df, training_cfg)
    random_state = int(training_cfg.get("random_state", 42))

    model_dir = Path(training_cfg.get("model_output_dir") or "models")
    if not model_dir.is_absolute():
        model_dir = stage_dir / model_dir
    reports_dir = stage_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_bin = model_dir / "tfidf_logreg.joblib"
    _, metrics, model_info = train_eval_persist(
        dataset_df,
        label_col=label_col,
        text_col=text_col,
        training_cfg=training_cfg,
        model_path=model_bin,
        random_state=random_state,
    )

    primary = str(training_cfg.get("metric_primary", "f1_macro"))
    report_txt = str(metrics.pop("classification_report", ""))
    metrics_out = {
        **{k: round(metrics[k], 6) for k in ("accuracy", "f1_macro", "precision_macro", "recall_macro")},
        "primary_metric": primary,
        "primary_value": round(float(metrics.get(primary, metrics["f1_macro"])), 6),
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
        "n_rows": int(len(dataset_df)),
        "n_rows_labeled": int(metrics.get("n_rows_labeled", 0)),
        "n_classes": metrics["n_classes"],
        "trainer": model_info["trainer"],
        "label_column": label_col,
        "text_column": text_col,
    }

    model_info_path = model_dir / "model_info.json"
    metrics_path = reports_dir / "metrics.json"
    metrics_md = reports_dir / "metrics.md"
    report_path = reports_dir / "classification_report.txt"

    model_info_path.write_text(json.dumps(model_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics_md.write_text(
        render_metrics_md({**metrics_out, **metrics, "trainer": model_info["trainer"]}, primary),
        encoding="utf-8",
    )
    report_path.write_text(report_txt + ("\n" if report_txt and not report_txt.endswith("\n") else ""), encoding="utf-8")

    model_pkl = model_dir / "model.pkl"
    artifacts = {
        "train_model_info_json": str(model_info_path.resolve()),
        "train_metrics_json": str(metrics_path.resolve()),
        "train_metrics_md": str(metrics_md.resolve()),
        "train_classification_report_txt": str(report_path.resolve()),
        "train_model_joblib": str(model_bin.resolve()),
        "train_model_pkl": str(model_pkl.resolve()) if model_pkl.exists() else str(model_bin.resolve()),
    }
    return TrainRunResult(
        model_info_path=str(model_info_path.resolve()),
        metrics_path=str(metrics_path.resolve()),
        artifact_paths=artifacts,
    )
