"""Lightweight, reproducible text-classification baseline for the TRAIN stage (TF-IDF + LogisticRegression)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def resolve_text_column(df: pd.DataFrame, training_cfg: dict[str, Any]) -> str:
    explicit = training_cfg.get("text_column")
    if explicit and explicit in df.columns:
        return str(explicit)
    for c in ("text", "prompt", "content", "body", "input"):
        if c in df.columns:
            return c
    raise ValueError(
        "TRAIN (text): no text column found. Set training.text_column or include one of: "
        + ", ".join(["text", "prompt", "content", "body", "input"])
        + f". Columns: {list(df.columns)}"
    )


def _build_tfidf_params(training_cfg: dict[str, Any]) -> dict[str, Any]:
    nested = dict(training_cfg.get("tfidf") or {})
    # Default token_pattern allows single-character tokens (sklearn's default requires len>=2,
    # which breaks very short texts like smoke-test rows "a"/"b").
    default_pattern = r"(?u)\b\w+\b"
    return {
        "max_features": int(nested.get("max_features", training_cfg.get("tfidf_max_features", 20_000))),
        "ngram_range": (
            int(nested.get("ngram_min", training_cfg.get("tfidf_ngram_min", 1))),
            int(nested.get("ngram_max", training_cfg.get("tfidf_ngram_max", 2))),
        ),
        "min_df": int(nested.get("min_df", training_cfg.get("tfidf_min_df", 1))),
        "max_df": float(nested.get("max_df", training_cfg.get("tfidf_max_df", 1.0))),
        "sublinear_tf": bool(nested.get("sublinear_tf", training_cfg.get("tfidf_sublinear_tf", True))),
        "token_pattern": str(nested.get("token_pattern", training_cfg.get("tfidf_token_pattern", default_pattern))),
    }


def _build_lr_params(training_cfg: dict[str, Any]) -> dict[str, Any]:
    nested = dict(training_cfg.get("logistic_regression") or {})
    return {
        "C": float(nested.get("C", training_cfg.get("lr_C", 1.0))),
        "max_iter": int(nested.get("max_iter", training_cfg.get("lr_max_iter", 2000))),
        "class_weight": nested.get("class_weight", training_cfg.get("lr_class_weight", "balanced")),
        "solver": str(nested.get("solver", training_cfg.get("lr_solver", "lbfgs"))),
        "random_state": int(training_cfg.get("random_state", 42)),
    }


def _build_pipeline(
    *,
    tfidf_kw: dict[str, Any],
    lr_kw: dict[str, Any],
    rs_lr: int,
    use_dummy: bool,
) -> Pipeline:
    if use_dummy:
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer(**tfidf_kw)),
                ("clf", DummyClassifier(strategy="most_frequent", random_state=rs_lr)),
            ]
        )
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**tfidf_kw)),
            ("clf", LogisticRegression(random_state=rs_lr, **lr_kw)),
        ]
    )


def train_eval_persist(
    df: pd.DataFrame,
    *,
    label_col: str,
    text_col: str,
    training_cfg: dict[str, Any],
    model_path: Any,
    random_state: int,
) -> tuple[Pipeline, dict[str, Any], dict[str, Any]]:
    """Fit TF-IDF + LogisticRegression, evaluate on held-out split, return pipeline and metric dicts."""

    if label_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"TRAIN needs columns {label_col!r} and {text_col!r}; got {list(df.columns)}.")

    work = df[[text_col, label_col]].copy()
    work = work.dropna(subset=[label_col])
    work[label_col] = work[label_col].astype(str).str.strip()
    work = work[work[label_col] != ""]
    if len(work) < 2:
        raise ValueError(
            f"TRAIN requires at least 2 rows with non-empty labels in {label_col!r} "
            f"(got {len(work)} after dropping empty/NA)."
        )

    X = work[text_col].fillna("").astype(str)
    y = work[label_col].astype(str)
    n = len(X)

    test_size = float(training_cfg.get("test_size", 0.2))
    test_size = min(max(test_size, 0.05), 0.5)
    if n <= 3:
        test_size = max(1.0 / n, 0.05)

    stratify = y if y.nunique() > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    if len(X_train) < 1 or len(X_test) < 1:
        raise ValueError(
            f"Train/test split produced empty fold (n={n}, test_size={test_size}). "
            "Add more rows or adjust training.test_size."
        )

    tfidf_kw = _build_tfidf_params(training_cfg)
    lr_kw_full = _build_lr_params(training_cfg)
    rs_lr = int(lr_kw_full.pop("random_state", random_state))
    lr_kw = dict(lr_kw_full)

    use_dummy = y.nunique() < 2 or len(X_train) < 2 or y_train.nunique() < 2
    if use_dummy:
        warnings.warn(
            "TRAIN: using DummyClassifier(strategy='most_frequent') inside the TF-IDF pipeline "
            "(single class and/or too few training rows for LogisticRegression).",
            UserWarning,
            stacklevel=2,
        )

    pipeline = _build_pipeline(
        tfidf_kw=tfidf_kw, lr_kw=lr_kw, rs_lr=rs_lr, use_dummy=use_dummy
    )

    if not use_dummy:
        try:
            pipeline.fit(X_train, y_train)
        except ValueError as exc:
            warnings.warn(
                f"TRAIN: LogisticRegression fit failed ({exc!r}); falling back to DummyClassifier.",
                UserWarning,
                stacklevel=2,
            )
            pipeline = _build_pipeline(
                tfidf_kw=tfidf_kw, lr_kw=lr_kw, rs_lr=rs_lr, use_dummy=True
            )
            pipeline.fit(X_train, y_train)
    else:
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    labels_for_report = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    report_txt = classification_report(
        y_test, y_pred, labels=labels_for_report, zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_classes": int(y.nunique()),
        "n_rows_labeled": int(n),
        "classification_report": report_txt,
    }

    mp = Path(model_path)
    joblib.dump(pipeline, mp)
    pkl_path = mp.parent / "model.pkl"
    if pkl_path.resolve() != mp.resolve():
        joblib.dump(pipeline, pkl_path)

    trainer_name = (
        "tfidf_dummy_most_frequent_fallback"
        if isinstance(pipeline.named_steps["clf"], DummyClassifier)
        else "tfidf_logistic_regression"
    )

    model_info = {
        "trainer": trainer_name,
        "sklearn_version": sklearn.__version__,
        "label_column": label_col,
        "text_column": text_col,
        "model_path": str(mp.resolve()),
        "model_pkl_path": str(pkl_path.resolve()),
        "hyperparams": {
            "test_size": test_size,
            "random_state": random_state,
            "tfidf": tfidf_kw,
            "logistic_regression": {**lr_kw, "random_state": rs_lr},
        },
        "n_rows": int(len(df)),
        "n_rows_labeled": int(n),
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
        "classes_": [str(c) for c in pipeline.named_steps["clf"].classes_],
    }
    return pipeline, metrics, model_info


def render_metrics_md(metrics: dict[str, Any], primary: str) -> str:
    lines = [
        "# Train Stage Metrics",
        "",
        f"- primary_metric ({primary}): `{metrics.get(primary, 'n/a')}`",
        f"- accuracy: `{metrics.get('accuracy')}`",
        f"- f1_macro: `{metrics.get('f1_macro')}`",
        f"- precision_macro: `{metrics.get('precision_macro')}`",
        f"- recall_macro: `{metrics.get('recall_macro')}`",
        f"- train_rows: {metrics.get('n_train')}",
        f"- test_rows: {metrics.get('n_test')}",
        f"- classes: {metrics.get('n_classes')}",
        f"- trainer: `{metrics.get('trainer', 'n/a')}`",
        "",
        "Model: TF-IDF + LogisticRegression (DummyClassifier only as an explicit fallback).",
        "",
        "See `classification_report.txt` in the train stage reports for per-class summary.",
        "",
    ]
    return "\n".join(lines)
