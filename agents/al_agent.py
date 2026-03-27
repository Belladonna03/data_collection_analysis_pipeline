from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


EPS = 1e-12
SUPPORTED_STRATEGIES = ("random", "entropy", "margin")


@dataclass(slots=True)
class ALConfig:
    """Configuration for the active-learning stage."""

    text_col: str = "text"
    label_col: str = "label"
    id_col: str = "id"
    model_name: str = "logreg"
    random_state: int = 42
    output_dir: str = "artifacts/al"
    reports_dir: str = "reports"
    human_label_col: str = "human_label"
    review_queue_filename: str = "review_queue_al.csv"
    corrected_review_filename: str = "review_queue_al_corrected.csv"


class ActiveLearningAgent:
    """Simple pool-based active-learning agent for text classification.

    The intended pipeline scenario is:
    1. Receive a reviewed dataset after `AnnotationAgent` and human QA.
    2. Split it into an initial labeled seed, unlabeled pool, and fixed test set.
    3. Run `run_cycle(..., simulation_mode=False)` to export a human review queue.
    4. A human fills `human_label` in the corrected CSV.
    5. Re-run the cycle or orchestrate the next iteration in `run_pipeline.py`.

    In notebook experiments, use `simulation_mode=True` to treat a hidden oracle
    label column in the pool as the human annotation source.
    """

    def __init__(
        self,
        text_col: str = "text",
        label_col: str = "label",
        id_col: str = "id",
        model_name: str = "logreg",
        random_state: int = 42,
        output_dir: str = "artifacts/al",
    ) -> None:
        self.config = ALConfig(
            text_col=text_col,
            label_col=label_col,
            id_col=id_col,
            model_name=model_name,
            random_state=random_state,
            output_dir=output_dir,
        )
        self.project_root = Path(__file__).resolve().parents[1]
        self._refresh_storage_dirs()

        self._rng = np.random.default_rng(self.config.random_state)
        self.model: Pipeline | None = None
        self._last_train_size: int | None = None
        self.last_artifacts: dict[str, str] = {}

    def fit(self, labeled_df: pd.DataFrame) -> ActiveLearningAgent:
        """Fit the sklearn text classifier on labeled data."""

        prepared = self._prepare_labeled_df(labeled_df)
        self.model = self._build_model()
        try:
            self.model.fit(prepared[self.config.text_col], prepared[self.config.label_col])
        except ValueError as exc:
            raise ValueError(
                "Could not fit the active-learning model. "
                "Check that the seed set contains at least two classes and enough repeated text "
                "features for TF-IDF with min_df=2."
            ) from exc
        self._last_train_size = int(len(prepared))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return class probabilities for the provided dataframe."""

        self._ensure_fitted()
        prepared = self._prepare_inference_df(df, df_name="inference dataframe")
        probabilities = self.model.predict_proba(prepared[self.config.text_col])
        return np.asarray(probabilities, dtype=float)

    def query(
        self,
        pool_df: pd.DataFrame,
        strategy: str = "entropy",
        batch_size: int = 20,
    ) -> pd.DataFrame:
        """Select the most informative examples from the unlabeled pool."""

        strategy = self._validate_strategy(strategy)
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self._ensure_fitted()
        prepared = self._prepare_pool_df(pool_df)
        if prepared.empty:
            return self._empty_selection_frame()

        actual_batch_size = min(batch_size, len(prepared))
        probabilities = self.predict_proba(prepared)
        predicted_indices = np.argmax(probabilities, axis=1)
        predicted_labels = self.model.classes_[predicted_indices]
        confidences = probabilities.max(axis=1)

        if strategy == "random":
            scores = self._rng.random(len(prepared))
            selected_positions = np.argsort(scores)[::-1][:actual_batch_size]
        elif strategy == "entropy":
            scores = -np.sum(probabilities * np.log(probabilities + EPS), axis=1)
            selected_positions = np.argsort(scores)[::-1][:actual_batch_size]
        else:
            sorted_probabilities = np.sort(probabilities, axis=1)
            top1 = sorted_probabilities[:, -1]
            top2 = sorted_probabilities[:, -2]
            scores = top1 - top2
            selected_positions = np.argsort(scores)[:actual_batch_size]

        selected = prepared.iloc[selected_positions].copy(deep=True)
        selected["selection_score"] = scores[selected_positions]
        selected["strategy"] = strategy
        selected["model_pred"] = predicted_labels[selected_positions]
        selected["model_confidence"] = confidences[selected_positions]
        return selected.reset_index(drop=True)

    def evaluate(self, test_df: pd.DataFrame) -> dict[str, Any]:
        """Evaluate the current model on a fixed labeled test set."""

        self._ensure_fitted()
        prepared = self._prepare_eval_df(test_df)
        if prepared.empty:
            raise ValueError("test_df must not be empty.")

        predictions = self.model.predict(prepared[self.config.text_col])
        y_true = prepared[self.config.label_col]
        accuracy = accuracy_score(y_true, predictions)
        average, pos_label = self._resolve_f1_config(y_true)
        if average == "binary":
            f1 = f1_score(y_true, predictions, average=average, pos_label=pos_label, zero_division=0)
        else:
            f1 = f1_score(y_true, predictions, average=average, zero_division=0)
        return {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "f1_average": average,
            "n_train": self._last_train_size,
            "n_test": int(len(prepared)),
        }

    def export_review_queue(
        self,
        selected_df: pd.DataFrame,
        iteration: int,
        path: str | Path,
    ) -> Path:
        """Export a batch selected by AL for manual human annotation."""

        export_path = Path(path)
        if not export_path.is_absolute():
            export_path = self.project_root / export_path
        if export_path.suffix.lower() != ".csv":
            export_path = export_path / self.config.review_queue_filename
        export_path.parent.mkdir(parents=True, exist_ok=True)

        export_df = selected_df.copy(deep=True)
        export_df["iteration"] = iteration
        if self.config.human_label_col not in export_df.columns:
            export_df[self.config.human_label_col] = ""
        export_df.to_csv(export_path, index=False)
        self.last_artifacts["review_queue_csv"] = str(export_path.resolve())
        return export_path.resolve()

    def ingest_human_labels(self, reviewed_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the reviewed dataframe into id-text-label triples."""

        self._validate_columns(
            reviewed_df,
            required=[self.config.id_col, self.config.text_col, self.config.human_label_col],
            df_name="reviewed_df",
        )
        if reviewed_df[self.config.human_label_col].isna().any():
            raise ValueError("reviewed_df contains NaN values in human_label.")

        normalized = reviewed_df.copy(deep=True)
        normalized[self.config.human_label_col] = normalized[self.config.human_label_col].astype(str).str.strip()
        empty_mask = normalized[self.config.human_label_col] == ""
        if empty_mask.any():
            raise ValueError(
                "reviewed_df contains empty human_label values. "
                "All selected examples must be labeled by a human before ingestion."
            )
        self._validate_unique_ids(normalized, df_name="reviewed_df")
        normalized = normalized[[self.config.id_col, self.config.text_col, self.config.human_label_col]].rename(
            columns={self.config.human_label_col: self.config.label_col}
        )
        return normalized.reset_index(drop=True)

    def run_cycle(
        self,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame,
        strategy: str = "entropy",
        n_iterations: int = 5,
        batch_size: int = 20,
        simulation_mode: bool = True,
        oracle_label_col: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run a pool-based active-learning loop and return per-iteration metrics."""

        strategy = self._validate_strategy(strategy)
        if n_iterations < 0:
            raise ValueError("n_iterations must be >= 0.")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        current_labeled = self._prepare_labeled_df(labeled_df, df_name="labeled_df")
        current_pool = self._prepare_pool_df(pool_df)
        prepared_test = self._prepare_eval_df(test_df)
        history: list[dict[str, Any]] = []

        for iteration in range(n_iterations + 1):
            self.fit(current_labeled)
            metrics = self.evaluate(prepared_test)
            metrics.update(
                {
                    "iteration": iteration,
                    "strategy": strategy,
                    "n_labeled": int(len(current_labeled)),
                    "n_pool": int(len(current_pool)),
                    "batch_size": int(batch_size),
                    "simulation_mode": bool(simulation_mode),
                }
            )
            history.append(metrics)

            if iteration == n_iterations or current_pool.empty:
                break

            selected = self.query(current_pool, strategy=strategy, batch_size=batch_size)
            actual_selected = len(selected)
            history[-1]["batch_selected"] = int(actual_selected)
            if actual_selected == 0:
                break

            selected_iteration = iteration + 1
            if simulation_mode:
                reviewed = self._simulate_human_review(selected, current_pool, oracle_label_col)
                review_queue_path = None
            else:
                queue_path = self.export_review_queue(
                    selected_df=selected,
                    iteration=selected_iteration,
                    path=self.review_dir / f"iteration_{selected_iteration:02d}",
                )
                review_queue_path = str(queue_path)
                corrected_path = queue_path.with_name(self.config.corrected_review_filename)
                if not corrected_path.exists():
                    raise FileNotFoundError(
                        "Human review is required before the next AL iteration. "
                        f"Fill `{self.config.human_label_col}` in `{queue_path}` and save the corrected file as "
                        f"`{corrected_path}`."
                    )
                reviewed = pd.read_csv(corrected_path)

            history[-1]["review_queue_path"] = review_queue_path
            newly_labeled = self.ingest_human_labels(reviewed)
            current_labeled = pd.concat([current_labeled, newly_labeled], ignore_index=True)
            current_pool = current_pool[~current_pool[self.config.id_col].isin(newly_labeled[self.config.id_col])].reset_index(drop=True)

        history_path = self._save_history_csv(history, strategy=strategy)
        model_path = self._save_model(strategy=strategy)
        self.last_artifacts["history_csv"] = str(history_path)
        self.last_artifacts["model_pickle"] = str(model_path)
        return history

    def compare_strategies(
        self,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame,
        strategies: list[str] | None = None,
        n_iterations: int = 5,
        batch_size: int = 20,
        simulation_mode: bool = True,
        oracle_label_col: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Run the same AL loop for several strategies."""

        strategy_list = strategies or ["random", "entropy"]
        histories: dict[str, list[dict[str, Any]]] = {}
        for strategy in strategy_list:
            child = ActiveLearningAgent(
                text_col=self.config.text_col,
                label_col=self.config.label_col,
                id_col=self.config.id_col,
                model_name=self.config.model_name,
                random_state=self.config.random_state,
                output_dir=str(Path(self.config.output_dir) / strategy),
            )
            child.project_root = self.project_root
            child._refresh_storage_dirs()
            histories[strategy] = child.run_cycle(
                labeled_df=labeled_df,
                pool_df=pool_df,
                test_df=test_df,
                strategy=strategy,
                n_iterations=n_iterations,
                batch_size=batch_size,
                simulation_mode=simulation_mode,
                oracle_label_col=oracle_label_col,
            )
        return histories

    def compute_sample_efficiency(
        self,
        histories: dict[str, list[dict[str, Any]]],
        improved_strategy: str = "entropy",
        baseline_strategy: str = "random",
        metric: str = "f1",
    ) -> dict[str, Any]:
        """Estimate how many labeled examples the improved strategy saves."""

        if improved_strategy not in histories or baseline_strategy not in histories:
            return {
                "baseline_strategy": baseline_strategy,
                "improved_strategy": improved_strategy,
                "metric": metric,
                "target_reached": False,
                "target_quality": None,
                "n_labeled_baseline_final": None,
                "n_labeled_improved_at_target": None,
                "saved_examples": 0,
            }

        baseline_df = pd.DataFrame(histories[baseline_strategy])
        improved_df = pd.DataFrame(histories[improved_strategy])
        if baseline_df.empty or improved_df.empty:
            return {
                "baseline_strategy": baseline_strategy,
                "improved_strategy": improved_strategy,
                "metric": metric,
                "target_reached": False,
                "target_quality": None,
                "n_labeled_baseline_final": None,
                "n_labeled_improved_at_target": None,
                "saved_examples": 0,
            }

        target_quality = float(baseline_df[metric].iloc[-1])
        reaching_rows = improved_df[improved_df[metric] >= target_quality]
        if reaching_rows.empty:
            return {
                "baseline_strategy": baseline_strategy,
                "improved_strategy": improved_strategy,
                "metric": metric,
                "target_reached": False,
                "target_quality": target_quality,
                "n_labeled_baseline_final": int(baseline_df["n_labeled"].iloc[-1]),
                "n_labeled_improved_at_target": None,
                "saved_examples": 0,
            }

        n_labeled_baseline_final = int(baseline_df["n_labeled"].iloc[-1])
        n_labeled_improved_at_target = int(reaching_rows["n_labeled"].iloc[0])
        return {
            "baseline_strategy": baseline_strategy,
            "improved_strategy": improved_strategy,
            "metric": metric,
            "target_reached": True,
            "target_quality": target_quality,
            "n_labeled_baseline_final": n_labeled_baseline_final,
            "n_labeled_improved_at_target": n_labeled_improved_at_target,
            "saved_examples": max(0, n_labeled_baseline_final - n_labeled_improved_at_target),
        }

    def report(
        self,
        histories: dict[str, list[dict[str, Any]]] | list[dict[str, Any]],
        output_path: str = "reports/learning_curve.png",
    ) -> str:
        """Save the learning-curve plot, CSV histories, and markdown summary."""

        normalized_histories = self._normalize_histories(histories)
        output_plot_path = self._resolve_path(output_path)
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        final_rows: list[dict[str, Any]] = []
        for strategy, history in normalized_histories.items():
            history_df = pd.DataFrame(history)
            if history_df.empty:
                continue
            history_csv_path = self._save_history_csv(history, strategy=strategy)
            self.last_artifacts[f"history_csv_{strategy}"] = str(history_csv_path)
            ax.plot(
                history_df["n_labeled"],
                history_df["f1"],
                marker="o",
                linewidth=2,
                label=strategy,
            )
            final_rows.append(history_df.iloc[-1].to_dict())

        ax.set_title("Active Learning Curve")
        ax.set_xlabel("Number of labeled examples")
        ax.set_ylabel("F1 score")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(output_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        sample_efficiency = self.compute_sample_efficiency(normalized_histories)
        report_path = self._write_markdown_report(
            histories=normalized_histories,
            final_rows=final_rows,
            sample_efficiency=sample_efficiency,
            plot_path=output_plot_path,
        )
        self.last_artifacts["learning_curve_png"] = str(output_plot_path.resolve())
        self.last_artifacts["report_md"] = str(report_path)
        return str(output_plot_path.resolve())

    def _build_model(self) -> Pipeline:
        """Build the baseline sklearn text classifier."""

        if self.config.model_name != "logreg":
            raise ValueError(f"Unsupported model_name: {self.config.model_name}. Only 'logreg' is supported.")
        return Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=20_000,
                        ngram_range=(1, 2),
                        min_df=2,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )

    def _prepare_labeled_df(self, df: pd.DataFrame, df_name: str = "labeled_df") -> pd.DataFrame:
        """Validate and normalize a labeled dataframe."""

        self._validate_dataframe(df, df_name=df_name)
        self._validate_columns(df, required=[self.config.id_col, self.config.text_col, self.config.label_col], df_name=df_name)
        self._validate_unique_ids(df, df_name=df_name)
        self._validate_text_values(df, df_name=df_name)
        self._validate_label_values(df, df_name=df_name, label_col=self.config.label_col)

        prepared = df.copy(deep=True)
        prepared[self.config.text_col] = prepared[self.config.text_col].astype(str)
        if prepared.empty:
            raise ValueError(f"{df_name} must not be empty.")
        if prepared[self.config.label_col].nunique(dropna=True) < 2:
            raise ValueError(
                f"{df_name} must contain at least two unique labels. "
                "LogisticRegression cannot be trained on a single class."
            )
        return prepared.reset_index(drop=True)

    def _prepare_pool_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize an unlabeled pool dataframe."""

        self._validate_dataframe(df, df_name="pool_df")
        self._validate_columns(df, required=[self.config.id_col, self.config.text_col], df_name="pool_df")
        self._validate_unique_ids(df, df_name="pool_df")
        self._validate_text_values(df, df_name="pool_df")
        prepared = df.copy(deep=True)
        prepared[self.config.text_col] = prepared[self.config.text_col].astype(str)
        return prepared.reset_index(drop=True)

    def _prepare_eval_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate a labeled evaluation dataframe."""

        self._validate_dataframe(df, df_name="test_df")
        self._validate_columns(
            df,
            required=[self.config.id_col, self.config.text_col, self.config.label_col],
            df_name="test_df",
        )
        self._validate_unique_ids(df, df_name="test_df")
        self._validate_text_values(df, df_name="test_df")
        self._validate_label_values(df, df_name="test_df", label_col=self.config.label_col)
        prepared = df.copy(deep=True)
        prepared[self.config.text_col] = prepared[self.config.text_col].astype(str)
        if prepared.empty:
            raise ValueError("test_df must not be empty.")
        return prepared.reset_index(drop=True)

    def _prepare_inference_df(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """Validate an inference-only dataframe with text and ids."""

        self._validate_dataframe(df, df_name=df_name)
        self._validate_columns(df, required=[self.config.id_col, self.config.text_col], df_name=df_name)
        self._validate_unique_ids(df, df_name=df_name)
        self._validate_text_values(df, df_name=df_name)
        prepared = df.copy(deep=True)
        prepared[self.config.text_col] = prepared[self.config.text_col].astype(str)
        return prepared.reset_index(drop=True)

    def _simulate_human_review(
        self,
        selected_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        oracle_label_col: str | None,
    ) -> pd.DataFrame:
        """Use an oracle label column as a stand-in for a human annotator."""

        resolved_oracle = oracle_label_col or self._infer_oracle_label_col(pool_df)
        if resolved_oracle is None or resolved_oracle not in pool_df.columns:
            raise ValueError(
                "simulation_mode=True requires an oracle label column in pool_df. "
                "Provide `oracle_label_col` or include `true_label` / `label`."
            )

        reviewed = selected_df.copy(deep=True)
        if resolved_oracle not in reviewed.columns:
            oracle_map = pool_df[[self.config.id_col, resolved_oracle]].drop_duplicates()
            reviewed = reviewed.merge(oracle_map, on=self.config.id_col, how="left")
        if reviewed[resolved_oracle].isna().any():
            raise ValueError(f"Oracle label column `{resolved_oracle}` contains missing values for selected rows.")
        reviewed[self.config.human_label_col] = reviewed[resolved_oracle]
        return reviewed

    def _infer_oracle_label_col(self, pool_df: pd.DataFrame) -> str | None:
        """Infer a usable oracle label column for simulation mode."""

        for candidate in ("true_label", self.config.label_col):
            if candidate in pool_df.columns:
                return candidate
        return None

    def _normalize_histories(
        self,
        histories: dict[str, list[dict[str, Any]]] | list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Normalize single- and multi-strategy histories into one mapping."""

        if isinstance(histories, list):
            if not histories:
                return {"unknown": []}
            strategy = str(histories[0].get("strategy", "strategy"))
            return {strategy: histories}
        return histories

    def _save_history_csv(self, history: list[dict[str, Any]], strategy: str) -> Path:
        """Persist a strategy history as CSV."""

        path = self.reports_dir / f"al_history_{strategy}.csv"
        pd.DataFrame(history).to_csv(path, index=False)
        return path.resolve()

    def _save_model(self, strategy: str) -> Path:
        """Persist the current sklearn model as a pickle artifact."""

        self._ensure_fitted()
        path = self.models_dir / f"al_model_{strategy}.pkl"
        with path.open("wb") as file:
            pickle.dump(self.model, file)
        return path.resolve()

    def _write_markdown_report(
        self,
        histories: dict[str, list[dict[str, Any]]],
        final_rows: list[dict[str, Any]],
        sample_efficiency: dict[str, Any],
        plot_path: Path,
    ) -> Path:
        """Create a short markdown report for the AL experiment."""

        final_df = pd.DataFrame(final_rows)
        if final_df.empty:
            metrics_table = "_No history rows available._"
        else:
            metrics_table = self._markdown_table(
                final_df[["strategy", "iteration", "n_labeled", "n_pool", "accuracy", "f1", "f1_average"]]
            )

        first_history = next((history for history in histories.values() if history), [])
        initial_n = int(first_history[0]["n_labeled"]) if first_history else 0
        iterations = max((len(history) - 1 for history in histories.values()), default=0)
        batch_size = int(first_history[0].get("batch_size", 0)) if first_history else 0

        if sample_efficiency.get("target_reached"):
            efficiency_text = (
                f"- Baseline strategy: `{sample_efficiency['baseline_strategy']}`\n"
                f"- Improved strategy: `{sample_efficiency['improved_strategy']}`\n"
                f"- Target metric ({sample_efficiency['metric']}): `{sample_efficiency['target_quality']:.4f}`\n"
                f"- Baseline labeled examples at final point: `{sample_efficiency['n_labeled_baseline_final']}`\n"
                f"- Improved strategy labeled examples at target: `{sample_efficiency['n_labeled_improved_at_target']}`\n"
                f"- Saved examples: `{sample_efficiency['saved_examples']}`"
            )
        else:
            efficiency_text = (
                "- Sample-efficiency target was not reached or the required strategies were not available.\n"
                f"- Saved examples: `{sample_efficiency.get('saved_examples', 0)}`"
            )

        strategies_text = ", ".join(f"`{name}`" for name in histories) or "`random`"
        relative_plot_path = self._relative_to_project(plot_path)
        report_content = "\n".join(
            [
                "# Active Learning Report",
                "",
                "## Model",
                "",
                "- `Pipeline(TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2) + LogisticRegression(max_iter=2000, class_weight='balanced'))`",
                f"- Random state: `{self.config.random_state}`",
                "",
                "## Strategies",
                "",
                f"- Compared strategies: {strategies_text}",
                "- `random`: uniform sampling without replacement",
                "- `entropy`: select highest predictive entropy",
                "- `margin`: select smallest top1-top2 probability gap",
                "",
                "## Experiment Parameters",
                "",
                f"- Initial labeled set size: `{initial_n}`",
                f"- Iterations: `{iterations}`",
                f"- Batch size: `{batch_size}`",
                "",
                "## Final Metrics",
                "",
                metrics_table,
                "",
                "## Sample Efficiency",
                "",
                efficiency_text,
                "",
                "## Human In The Loop",
                "",
                "- After AL selection, examples are exported to a human review queue.",
                "- The agent never assigns ground-truth labels itself in `simulation_mode=False`.",
                f"- Expected manual review column: `{self.config.human_label_col}`.",
                "",
                "## Artifacts",
                "",
                f"- Learning curve: `{relative_plot_path}`",
                "- Per-strategy histories: `reports/al_history_<strategy>.csv`",
                f"- Model artifact directory: `{self._relative_to_project(self.models_dir)}`",
                "",
                "## Config Snapshot",
                "",
                "```json",
                json.dumps(asdict(self.config), indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
        report_path = self.reports_dir / "al_report.md"
        report_path.write_text(report_content, encoding="utf-8")
        return report_path.resolve()

    def _resolve_f1_config(self, labels: pd.Series) -> tuple[str, Any | None]:
        """Choose binary or macro F1 depending on the number of classes."""

        unique_labels = list(pd.unique(labels))
        if len(unique_labels) == 2:
            return "binary", unique_labels[-1]
        return "macro", None

    def _validate_strategy(self, strategy: str) -> str:
        """Validate an AL strategy name."""

        normalized = str(strategy).strip().lower()
        if normalized not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. Supported strategies: {', '.join(SUPPORTED_STRATEGIES)}."
            )
        return normalized

    def _ensure_fitted(self) -> None:
        """Require that the sklearn model is already fitted."""

        if self.model is None:
            raise ValueError("The active-learning model is not fitted yet. Call fit() first.")

    def _validate_dataframe(self, df: pd.DataFrame, df_name: str) -> None:
        """Validate dataframe input type."""

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{df_name} must be a pandas DataFrame.")

    def _validate_columns(self, df: pd.DataFrame, required: list[str], df_name: str) -> None:
        """Validate that all required columns are present."""

        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(
                f"{df_name} is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    def _validate_text_values(self, df: pd.DataFrame, df_name: str) -> None:
        """Reject missing text inputs because they break deterministic preprocessing."""

        if df[self.config.text_col].isna().any():
            raise ValueError(f"{df_name} contains NaN values in `{self.config.text_col}`.")

    def _validate_label_values(self, df: pd.DataFrame, df_name: str, label_col: str) -> None:
        """Reject missing labels in labeled datasets."""

        if df[label_col].isna().any():
            raise ValueError(f"{df_name} contains NaN values in `{label_col}`.")

    def _validate_unique_ids(self, df: pd.DataFrame, df_name: str) -> None:
        """Require unique ids inside each dataframe."""

        if df[self.config.id_col].duplicated().any():
            duplicates = df.loc[df[self.config.id_col].duplicated(), self.config.id_col].astype(str).tolist()
            preview = duplicates[:5]
            raise ValueError(f"{df_name} contains duplicate ids in `{self.config.id_col}`: {preview}")

    def _resolve_path(self, path_value: str | Path) -> Path:
        """Resolve relative project paths against the repository root."""

        path = Path(path_value)
        return path if path.is_absolute() else self.project_root / path

    def _refresh_storage_dirs(self) -> None:
        """Rebuild storage directories after project-root changes."""

        self.output_dir = self._resolve_path(self.config.output_dir)
        self.reports_dir = self._resolve_path(self.config.reports_dir)
        self.review_dir = self.output_dir / "review"
        self.models_dir = self.output_dir / "models"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.review_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _relative_to_project(self, path: Path) -> str:
        """Render a path relative to the repository root when possible."""

        try:
            return str(path.resolve().relative_to(self.project_root.resolve()))
        except ValueError:
            return str(path.resolve())

    def _empty_selection_frame(self) -> pd.DataFrame:
        """Return an empty selection dataframe with the expected service columns."""

        return pd.DataFrame(
            columns=[
                self.config.id_col,
                self.config.text_col,
                "selection_score",
                "strategy",
                "model_pred",
                "model_confidence",
            ]
        )

    def _markdown_table(self, df: pd.DataFrame) -> str:
        """Render a minimal markdown table without optional dependencies."""

        headers = list(df.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in df.iterrows():
            values = []
            for value in row.tolist():
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)
