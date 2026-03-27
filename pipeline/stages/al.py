from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

import numpy as np
import pandas as pd

from agents.al_agent import ALConfig, ActiveLearningAgent

CHECKPOINT_NAME = "al_checkpoint.json"
LABELED_PARQUET = "al_labeled.parquet"
POOL_PARQUET = "al_pool.parquet"
TEST_PARQUET = "al_test.parquet"
HITL_HISTORY_JSON = "al_hitl_history.json"


@dataclass
class ALRunResult:
    review_queue_path: str
    report_path: str
    review_needed: bool
    selected_count: int
    artifact_paths: Dict[str, str]


@dataclass
class ALReviewApplyResult:
    patched_df: pd.DataFrame
    artifact_paths: Dict[str, str]
    al_complete: bool = True


@dataclass(frozen=True)
class ResolvedALConfig:
    enabled: bool
    strategy: str
    query_size: int
    initial_seed_size: int
    n_iterations: int
    compare_against_random: bool
    simulation_mode: bool
    random_state: int
    history_path: str
    test_holdout_fraction: float
    test_holdout_min: int


def resolve_al_config(config_payload: Optional[Mapping[str, Any]]) -> ResolvedALConfig:
    raw = dict((config_payload or {}).get("active_learning") or {})
    return ResolvedALConfig(
        enabled=bool(raw.get("enabled", True)),
        strategy=str(raw.get("strategy", "entropy")).strip().lower(),
        query_size=int(raw.get("query_size", raw.get("batch_size", 20))),
        initial_seed_size=int(raw.get("initial_seed_size", 500)),
        n_iterations=int(raw.get("n_iterations", 5)),
        compare_against_random=bool(raw.get("compare_against_random", False)),
        simulation_mode=bool(raw.get("simulation_mode", False)),
        random_state=int(raw.get("random_state", 42)),
        history_path=str(raw.get("history_path", "reports/al_history.json")),
        test_holdout_fraction=float(raw.get("test_holdout_fraction", 0.2)),
        test_holdout_min=int(raw.get("test_holdout_min", 2)),
    )


def _resolve_label_column(df: pd.DataFrame) -> str:
    for candidate in ("final_label", "human_label", "label", "auto_label"):
        if candidate in df.columns:
            return candidate
    raise ValueError("AL stage requires one of label columns: final_label/human_label/label/auto_label")


def _resolve_text_column(df: pd.DataFrame) -> str:
    for candidate in ("prompt", "text", "content", "input"):
        if candidate in df.columns:
            return candidate
    raise ValueError("AL stage requires a text column (prompt/text/content/input).")


def _resolve_id_column(df: pd.DataFrame) -> str:
    for candidate in ("annotation_id", "id", "record_id", "sample_id"):
        if candidate in df.columns:
            return candidate
    return "annotation_id"


def _prepare_work_frame(annotated_df: pd.DataFrame) -> tuple[pd.DataFrame, str, str, str]:
    text_col = _resolve_text_column(annotated_df)
    label_col = _resolve_label_column(annotated_df)
    id_col = _resolve_id_column(annotated_df)
    work_df = annotated_df.copy()
    if id_col not in work_df.columns:
        work_df[id_col] = [f"row_{i}" for i in range(len(work_df))]
    if label_col != "label":
        work_df["label"] = work_df[label_col].astype(str)
    if text_col != "text":
        work_df["text"] = work_df[text_col].astype(str)
    if id_col != "id":
        work_df["id"] = work_df[id_col].astype(str)
    canonical = work_df[["id", "text", "label"]].copy()
    return canonical, id_col, label_col, text_col


def _deterministic_shuffle(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(df))
    return df.iloc[order].reset_index(drop=True)


def split_for_al(df_lab: pd.DataFrame, cfg: ResolvedALConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratify-friendly: shuffle then slice; ensures room for seed, test, pool."""
    n = len(df_lab)
    if n < 6:
        raise ValueError(f"AL requires at least 6 labeled rows (got {n}).")
    shuffled = _deterministic_shuffle(df_lab, cfg.random_state)
    n_test = max(cfg.test_holdout_min, int(round(n * cfg.test_holdout_fraction)))
    n_test = min(n_test, n - 3)
    n_seed = min(cfg.initial_seed_size, n - n_test - 1)
    n_seed = max(2, n_seed)
    if n_seed + n_test >= n:
        n_seed = max(2, n - n_test - 2)
    test_df = shuffled.iloc[:n_test].copy()
    seed_df = shuffled.iloc[n_test : n_test + n_seed].copy()
    pool_df = shuffled.iloc[n_test + n_seed :].copy()
    if pool_df.empty:
        moved = seed_df.iloc[-1:]
        seed_df = seed_df.iloc[:-1].copy()
        pool_df = pd.concat([pool_df, moved], ignore_index=True)
    seed_df = seed_df.reset_index(drop=True)
    pool_df = pool_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    if seed_df["label"].nunique(dropna=True) < 2:
        raise ValueError(
            "AL seed split must contain at least two distinct labels. "
            "Increase data, adjust active_learning.random_state, or preprocess labels."
        )
    return seed_df, pool_df, test_df


def _write_history_json(stage_dir: Path, rel_path: str, payload: Any) -> Path:
    target = (stage_dir / rel_path).resolve() if not Path(rel_path).is_absolute() else Path(rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return target


def _merge_accumulated_labels(
    base_df: pd.DataFrame,
    labeled_accum: pd.DataFrame,
    *,
    base_id_col: str,
) -> pd.DataFrame:
    out = base_df.copy()
    acc = labeled_accum.copy()
    acc["id"] = acc["id"].astype(str)
    out[base_id_col] = out[base_id_col].astype(str)
    label_map = acc.drop_duplicates("id").set_index("id")["label"].astype(str)
    if "final_label" not in out.columns:
        label_col_src = _resolve_label_column(out)
        out["final_label"] = out[label_col_src].astype(str)
    new_final = []
    for _, row in out.iterrows():
        rid = str(row[base_id_col])
        new_final.append(label_map.get(rid, row["final_label"]))
    out["final_label"] = new_final
    return out


def _run_simulation_mode(
    *,
    stage_dir: Path,
    canonical: pd.DataFrame,
    base_stub: pd.DataFrame,
    base_id_col: str,
    base_label_col: str,
    cfg: ResolvedALConfig,
    strategy_override: Optional[str],
    batch_override: Optional[int],
) -> ALRunResult:
    seed_df, pool_df, test_df = split_for_al(canonical, cfg)
    strat = (strategy_override or cfg.strategy).strip().lower()
    batch_size = int(batch_override if batch_override is not None else cfg.query_size)
    agent = ActiveLearningAgent(
        text_col="text",
        label_col="label",
        id_col="id",
        output_dir=str(stage_dir / "data"),
        random_state=cfg.random_state,
    )
    strategies = [strat]
    if cfg.compare_against_random and "random" not in strategies:
        strategies = strategies + ["random"]
    pool_oracle = pool_df.copy()
    histories = agent.compare_strategies(
        labeled_df=seed_df,
        pool_df=pool_oracle,
        test_df=test_df,
        strategies=strategies,
        n_iterations=cfg.n_iterations,
        batch_size=batch_size,
        simulation_mode=True,
        oracle_label_col="label",
    )
    report_rel = "reports/learning_curve.png"
    curve_path = Path(agent.report(histories, output_path=str(stage_dir / report_rel)))
    history_payload = {"simulation_mode": True, "strategies": histories, "primary_strategy": strat}
    hist_file = _write_history_json(stage_dir, cfg.history_path, history_payload)
    report_md = stage_dir / "reports" / "al_report.md"
    report_md.parent.mkdir(parents=True, exist_ok=True)
    if agent.last_artifacts.get("report_md"):
        report_md = Path(agent.last_artifacts["report_md"])
    else:
        report_md.write_text(
            "\n".join(
                [
                    "# Active Learning Stage (simulation)",
                    "",
                    f"- primary strategy: {strat}",
                    f"- strategies compared: {', '.join(histories.keys())}",
                    f"- learning curve: {report_rel}",
                    f"- history json: {cfg.history_path}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    data_dir = stage_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    seed_df.to_parquet(data_dir / "seed_snapshot.parquet", index=False)
    merged = _merge_accumulated_labels(base_stub, canonical, base_id_col=base_id_col)
    # In simulation, oracle labels for pool are already in canonical; training gets full supervised frame.
    final_path = data_dir / "al_final_dataset.parquet"
    export_df = merged.copy()
    export_df.attrs.clear()
    export_df.to_parquet(final_path, index=False)
    artifacts: Dict[str, str] = dict(agent.last_artifacts)
    artifacts.update(
        {
            "al_review_queue_csv": "",
            "al_report_md": str(report_md.resolve()),
            "al_seed_dataset": str((data_dir / "seed_snapshot.parquet").resolve()),
            "al_final_dataset": str(final_path.resolve()),
            "al_history_json": str(hist_file.resolve()),
            "al_learning_curve_png": str(curve_path.resolve()),
        }
    )
    return ALRunResult(
        review_queue_path="",
        report_path=str(report_md.resolve()),
        review_needed=False,
        selected_count=0,
        artifact_paths=artifacts,
    )


def _checkpoint_path(stage_dir: Path) -> Path:
    return stage_dir / "data" / CHECKPOINT_NAME


def _save_checkpoint(stage_dir: Path, payload: MutableMapping[str, Any]) -> None:
    path = _checkpoint_path(stage_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_checkpoint(stage_dir: Path) -> Optional[dict[str, Any]]:
    path = _checkpoint_path(stage_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _append_hitl_history(stage_dir: Path, row: dict[str, Any]) -> None:
    path = stage_dir / "data" / HITL_HISTORY_JSON
    rows: List[dict[str, Any]] = []
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
    rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _hitl_step_fit_eval_record(
    agent: ActiveLearningAgent,
    labeled_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    stage_dir: Path,
    strategy: str,
    reviews_completed: int,
    cfg: ResolvedALConfig,
    pool_len: int,
) -> dict[str, Any]:
    agent.fit(labeled_df)
    metrics = agent.evaluate(test_df)
    row = {
        **metrics,
        "iteration": reviews_completed,
        "strategy": strategy,
        "n_labeled": int(len(labeled_df)),
        "n_pool": int(pool_len),
        "batch_size": int(cfg.query_size),
        "simulation_mode": False,
    }
    _append_hitl_history(stage_dir, row)
    return row


def run_al_stage(
    *,
    stage_dir: Path,
    annotated_df: pd.DataFrame,
    config_payload: Optional[Mapping[str, Any]] = None,
    strategy: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> ALRunResult:
    cfg = resolve_al_config(config_payload)
    canonical, base_id_col, base_label_col, _text_col = _prepare_work_frame(annotated_df)
    base_stub = annotated_df.copy()
    if base_id_col not in base_stub.columns:
        base_stub[base_id_col] = canonical["id"].values

    if not cfg.enabled:
        data_dir = stage_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        final_path = data_dir / "al_final_dataset.parquet"
        stub = base_stub.copy()
        if "final_label" not in stub.columns:
            stub["final_label"] = stub[base_label_col].astype(str)
        export_df = stub.copy()
        export_df.attrs.clear()
        export_df.to_parquet(final_path, index=False)
        report_md = stage_dir / "reports" / "al_report.md"
        report_md.parent.mkdir(parents=True, exist_ok=True)
        report_md.write_text("# Active Learning\n\nStage disabled in config; passthrough input.\n", encoding="utf-8")
        return ALRunResult(
            review_queue_path="",
            report_path=str(report_md.resolve()),
            review_needed=False,
            selected_count=0,
            artifact_paths={
                "al_report_md": str(report_md.resolve()),
                "al_final_dataset": str(final_path.resolve()),
            },
        )

    if cfg.simulation_mode:
        return _run_simulation_mode(
            stage_dir=stage_dir,
            canonical=canonical,
            base_stub=base_stub,
            base_id_col=base_id_col,
            base_label_col=base_label_col,
            cfg=cfg,
            strategy_override=strategy,
            batch_override=batch_size,
        )

    strat = (strategy or cfg.strategy).strip().lower()
    qsize = int(batch_size if batch_size is not None else cfg.query_size)

    existing = _load_checkpoint(stage_dir)
    data_dir = stage_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    agent = ActiveLearningAgent(
        text_col="text",
        label_col="label",
        id_col="id",
        output_dir=str(data_dir),
        random_state=cfg.random_state,
    )

    if existing is None:
        seed_df, pool_df, test_df = split_for_al(canonical, cfg)
        seed_df.to_parquet(data_dir / "seed_snapshot.parquet", index=False)
        pool_export = pool_df[["id", "text"]].copy()
        pool_export.to_parquet(data_dir / POOL_PARQUET, index=False)
        test_df.to_parquet(data_dir / TEST_PARQUET, index=False)
        labeled_write = seed_df.copy()
        labeled_write.to_parquet(data_dir / LABELED_PARQUET, index=False)
        hist_file = stage_dir / "data" / HITL_HISTORY_JSON
        if hist_file.exists():
            hist_file.unlink()
        _save_checkpoint(
            stage_dir,
            {
                "strategy": strat,
                "n_iterations": cfg.n_iterations,
                "query_size": qsize,
                "reviews_completed": 0,
                "pending_export_iteration": 0,
                "random_state": cfg.random_state,
                "history_path": cfg.history_path,
                "base_id_col": base_id_col,
            },
        )
        existing = _load_checkpoint(stage_dir)
        assert existing is not None
        labeled_df = labeled_write
        pool_only = pool_export
        test_snap = test_df
    else:
        labeled_df = pd.read_parquet(data_dir / LABELED_PARQUET)
        pool_only = pd.read_parquet(data_dir / POOL_PARQUET)
        test_snap = pd.read_parquet(data_dir / TEST_PARQUET)
        strat = str(existing.get("strategy", strat))

    reviews_completed = int(existing.get("reviews_completed", 0))
    pending_it = int(existing.get("pending_export_iteration", 0))
    n_it = int(existing.get("n_iterations", cfg.n_iterations))
    qsize = int(existing.get("query_size", qsize))

    if pending_it > 0:
        qdir = stage_dir / "review" / f"iteration_{pending_it:02d}"
        qcsv = qdir / ALConfig().review_queue_filename
        if qcsv.is_file():
            report_md = stage_dir / "reports" / "al_report.md"
            n_sel = len(pd.read_csv(qcsv))
            return ALRunResult(
                review_queue_path=str(qcsv.resolve()),
                report_path=str(report_md.resolve() if report_md.is_file() else (stage_dir / "reports" / "al_report.md").resolve()),
                review_needed=True,
                selected_count=n_sel,
                artifact_paths={"al_review_queue_csv": str(qcsv.resolve())},
            )

    if pending_it == 0 and reviews_completed == 0:
        _hitl_step_fit_eval_record(
            agent,
            labeled_df,
            test_snap,
            stage_dir=stage_dir,
            strategy=strat,
            reviews_completed=0,
            cfg=cfg,
            pool_len=len(pool_only),
        )

    if n_it <= 0:
        merged = _merge_accumulated_labels(base_stub, labeled_df, base_id_col=base_id_col)
        final_path = data_dir / "al_final_dataset.parquet"
        export_df = merged.copy()
        export_df.attrs.clear()
        export_df.to_parquet(final_path, index=False)
        history_rows = json.loads((stage_dir / "data" / HITL_HISTORY_JSON).read_text(encoding="utf-8"))
        _write_history_json(stage_dir, str(existing.get("history_path", cfg.history_path)), {strat: history_rows})
        agent.report({strat: history_rows}, output_path=str(stage_dir / "reports" / "learning_curve.png"))
        _checkpoint_path(stage_dir).unlink(missing_ok=True)
        report_md = Path(agent.last_artifacts.get("report_md", stage_dir / "reports" / "al_report.md"))
        return ALRunResult(
            review_queue_path="",
            report_path=str(report_md.resolve()),
            review_needed=False,
            selected_count=0,
            artifact_paths={
                "al_report_md": str(report_md.resolve()),
                "al_final_dataset": str(final_path.resolve()),
                "al_history_json": str(
                    _write_history_json(
                        stage_dir, str(existing.get("history_path", cfg.history_path)), {strat: history_rows}
                    ).resolve()
                ),
                "al_learning_curve_png": str(Path(agent.last_artifacts.get("learning_curve_png", "")).resolve())
                if agent.last_artifacts.get("learning_curve_png")
                else "",
            },
        )

    if reviews_completed >= n_it:
        history_rows = json.loads((stage_dir / "data" / HITL_HISTORY_JSON).read_text(encoding="utf-8"))
        hist_path = _write_history_json(stage_dir, str(existing.get("history_path", cfg.history_path)), {strat: history_rows})
        agent.report({strat: history_rows}, output_path=str(stage_dir / "reports" / "learning_curve.png"))
        merged = _merge_accumulated_labels(base_stub, labeled_df, base_id_col=base_id_col)
        final_path = data_dir / "al_final_dataset.parquet"
        export_df = merged.copy()
        export_df.attrs.clear()
        export_df.to_parquet(final_path, index=False)
        _checkpoint_path(stage_dir).unlink(missing_ok=True)
        report_md = Path(agent.last_artifacts.get("report_md", stage_dir / "reports" / "al_report.md"))
        arts = dict(agent.last_artifacts)
        arts.update(
            {
                "al_final_dataset": str(final_path.resolve()),
                "al_history_json": str(hist_path.resolve()),
                "al_report_md": str(report_md.resolve()),
            }
        )
        return ALRunResult(
            review_queue_path="",
            report_path=str(report_md.resolve()),
            review_needed=False,
            selected_count=0,
            artifact_paths=arts,
        )

    selected = agent.query(pool_only, strategy=strat, batch_size=qsize)
    next_export = reviews_completed + 1
    review_dir = stage_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    queue_path = agent.export_review_queue(selected, iteration=next_export, path=review_dir / f"iteration_{next_export:02d}")
    existing["pending_export_iteration"] = next_export
    _save_checkpoint(stage_dir, existing)

    report_md = stage_dir / "reports" / "al_report.md"
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(
        "\n".join(
            [
                "# Active Learning Stage (human-in-the-loop)",
                "",
                f"- strategy: {strat}",
                f"- reviews_completed: {reviews_completed} / {n_it}",
                f"- pending_iteration: {next_export}",
                f"- labeled_size: {len(labeled_df)}",
                f"- pool_size: {len(pool_only)}",
                f"- selected_for_review: {len(selected)}",
                "",
                "Save corrections as `human_label` in:",
                f"- `{queue_path.with_name(agent.config.corrected_review_filename)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    artifacts = dict(agent.last_artifacts)
    artifacts.update(
        {
            "al_review_queue_csv": str(queue_path.resolve()),
            "al_report_md": str(report_md.resolve()),
            "al_seed_dataset": str((data_dir / "seed_snapshot.parquet").resolve()),
        }
    )
    return ALRunResult(
        review_queue_path=str(queue_path.resolve()),
        report_path=str(report_md.resolve()),
        review_needed=bool(len(selected) > 0),
        selected_count=int(len(selected)),
        artifact_paths=artifacts,
    )


def apply_al_review(
    *,
    stage_dir: Path,
    base_annotated_df: pd.DataFrame,
    reviewed_queue_df: pd.DataFrame,
    config_payload: Optional[Mapping[str, Any]] = None,
) -> ALReviewApplyResult:
    """Apply AL human labels. When a multi-iteration checkpoint exists, advances the loop."""

    cfg = resolve_al_config(config_payload)
    ck = _load_checkpoint(stage_dir)
    base = base_annotated_df.copy()
    id_col = _resolve_id_column(base)
    label_col = _resolve_label_column(base)

    if ck is None:
        human_col = "human_label"
        if id_col not in reviewed_queue_df.columns:
            raise ValueError(f"AL reviewed file must include `{id_col}` column.")
        if human_col not in reviewed_queue_df.columns:
            raise ValueError("AL reviewed file must include `human_label` column.")
        patch = reviewed_queue_df[[id_col, human_col]].copy()
        patch[human_col] = patch[human_col].astype(str).str.strip()
        if (patch[human_col] == "").any():
            raise ValueError("AL reviewed file contains empty human_label values.")
        base[id_col] = base[id_col].astype(str)
        patch[id_col] = patch[id_col].astype(str)
        patched = base.merge(patch, on=id_col, how="left")
        patched["final_label"] = patched[human_col].where(patched[human_col].notna(), patched[label_col].astype(str))
        final_path = stage_dir / "data" / "al_final_dataset.parquet"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        export_df = patched.copy()
        export_df.attrs.clear()
        export_df.to_parquet(final_path, index=False)
        return ALReviewApplyResult(
            patched_df=patched,
            artifact_paths={"al_final_dataset": str(final_path.resolve())},
            al_complete=True,
        )

    data_dir = stage_dir / "data"
    pending = int(ck["pending_export_iteration"])
    strat = str(ck["strategy"])
    qsize = int(ck["query_size"])
    n_it = int(ck["n_iterations"])
    base_id_col = str(ck.get("base_id_col", id_col))

    human_col = "human_label"
    if human_col not in reviewed_queue_df.columns:
        raise ValueError("AL reviewed file must include `human_label`.")
    reviewed_queue_df = reviewed_queue_df.copy()
    reviewed_queue_df["id"] = reviewed_queue_df[id_col].astype(str) if id_col != "id" else reviewed_queue_df["id"].astype(str)
    if "text" not in reviewed_queue_df.columns:
        pool_full = pd.read_parquet(data_dir / POOL_PARQUET)
        id_to_text = pool_full.set_index("id")["text"].to_dict()
        reviewed_queue_df["text"] = reviewed_queue_df["id"].map(id_to_text)
    reviewed_queue_df[human_col] = reviewed_queue_df[human_col].astype(str).str.strip()
    if (reviewed_queue_df[human_col] == "").any():
        raise ValueError("AL reviewed file contains empty human_label values.")

    agent = ActiveLearningAgent(
        text_col="text",
        label_col="label",
        id_col="id",
        output_dir=str(data_dir),
        random_state=int(ck.get("random_state", 42)),
    )
    newly = agent.ingest_human_labels(reviewed_queue_df)

    labeled_df = pd.read_parquet(data_dir / LABELED_PARQUET)
    pool_df = pd.read_parquet(data_dir / POOL_PARQUET)
    test_df = pd.read_parquet(data_dir / TEST_PARQUET)

    labeled_df = pd.concat([labeled_df, newly], ignore_index=True)
    pool_df = pool_df[~pool_df["id"].astype(str).isin(newly["id"].astype(str))].reset_index(drop=True)
    labeled_df.to_parquet(data_dir / LABELED_PARQUET, index=False)
    pool_df.to_parquet(data_dir / POOL_PARQUET, index=False)

    reviews_completed = int(ck["reviews_completed"]) + 1
    ck["reviews_completed"] = reviews_completed
    ck["pending_export_iteration"] = 0
    _save_checkpoint(stage_dir, ck)

    _hitl_step_fit_eval_record(
        agent,
        labeled_df,
        test_df,
        stage_dir=stage_dir,
        strategy=strat,
        reviews_completed=reviews_completed,
        cfg=cfg,
        pool_len=len(pool_df),
    )

    artifact_paths: Dict[str, str] = {"al_labeled_accum": str((data_dir / LABELED_PARQUET).resolve())}

    if reviews_completed >= n_it or pool_df.empty:
        history_rows = json.loads((stage_dir / "data" / HITL_HISTORY_JSON).read_text(encoding="utf-8"))
        hp = str(ck.get("history_path", cfg.history_path))
        hist_path = _write_history_json(stage_dir, hp, {strat: history_rows})
        agent.report({strat: history_rows}, output_path=str(stage_dir / "reports" / "learning_curve.png"))
        merged = _merge_accumulated_labels(base, labeled_df, base_id_col=base_id_col)
        final_path = data_dir / "al_final_dataset.parquet"
        export_df = merged.copy()
        export_df.attrs.clear()
        export_df.to_parquet(final_path, index=False)
        _checkpoint_path(stage_dir).unlink(missing_ok=True)
        artifact_paths.update(
            {
                "al_final_dataset": str(final_path.resolve()),
                "al_history_json": str(hist_path.resolve()),
                "al_learning_curve_png": str(Path(agent.last_artifacts.get("learning_curve_png", "")).resolve())
                if agent.last_artifacts.get("learning_curve_png")
                else "",
                "al_report_md": str(Path(agent.last_artifacts.get("report_md", stage_dir / "reports" / "al_report.md")).resolve()),
            }
        )
        return ALReviewApplyResult(patched_df=merged, artifact_paths=artifact_paths, al_complete=True)

    selected = agent.query(pool_df, strategy=strat, batch_size=qsize)
    next_export = reviews_completed + 1
    queue_path = agent.export_review_queue(selected, iteration=next_export, path=stage_dir / "review" / f"iteration_{next_export:02d}")
    ck["pending_export_iteration"] = next_export
    _save_checkpoint(stage_dir, ck)
    artifact_paths.update(
        {
            "al_review_queue_csv": str(queue_path.resolve()),
            "al_report_md": str((stage_dir / "reports" / "al_report.md").resolve()),
        }
    )
    return ALReviewApplyResult(
        patched_df=base,
        artifact_paths=artifact_paths,
        al_complete=False,
    )


def al_find_latest_corrected_csv(stage_dir: Path) -> Optional[Path]:
    review_root = stage_dir / "review"
    if not review_root.is_dir():
        return None
    candidates = sorted(review_root.glob("iteration_*/review_queue_al_corrected.csv"))
    return candidates[-1] if candidates else None
