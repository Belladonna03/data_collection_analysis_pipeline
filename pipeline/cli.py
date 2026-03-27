from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple, Union

from agents.data_quality.schemas import QualityStageResult
from pipeline.artifacts import (
    load_config,
    read_dataframe,
    resolve_annotate_output_path,
    resolve_quality_input_path,
    resolve_quality_output_path,
    resolve_train_input_path,
)
from pipeline.collect_snapshots import (
    UserSelection,
    assign_stable_keys,
    build_recommendation_combinations,
    format_collect_status_block,
    load_user_selection,
    parse_candidate_ids,
    print_discover_catalog,
    print_recommendation_combinations,
    save_discovery_snapshot,
    save_recommendations,
    save_user_selection,
    serialize_collection_plans,
    spec_id_to_candidate_number,
    try_load_discovery_snapshot,
)
from pipeline.cli_syntax import (
    al_review_file_placeholder,
    annotate_review_file_placeholder,
    cli,
    collect_discover,
    pipeline_artifacts,
    pipeline_status,
    quality_review_decision_placeholder,
    quality_run_input_placeholder,
    stage_run,
)
from pipeline.orchestration import apply_annotation_stage_skip, skip_annotation_policy
from pipeline.registry import STAGE_BY_SHORT_NAME, STAGE_REGISTRY, StageDefinition
from pipeline.render import render_current_stage_summary, render_pipeline_summary
from pipeline.state import PipelineState, PipelineStateManager


def _maybe_skip_annotate_after_quality(
    manager: PipelineStateManager, state: PipelineState, cfg: Dict[str, Any], cleaned_path: Path
) -> Tuple[bool, str]:
    """If policy applies, mark ANNOTATE skipped and advance to AL. Returns (True, reason) or (False, "")."""

    skip, reason = skip_annotation_policy(cfg)
    if not skip:
        return False, ""
    ann_dir = Path(state.run_dir) / STAGE_BY_SHORT_NAME["annotate"].artifact_dir_name
    apply_annotation_stage_skip(
        manager,
        state,
        passthrough_dataset_path=str(cleaned_path.resolve()),
        annotate_stage_dir=ann_dir,
        reason=reason,
    )
    return True, reason


def _resolve_state(manager: PipelineStateManager, run_id: Optional[str], *, create_if_missing: bool) -> Optional[PipelineState]:
    if run_id:
        return manager.load_run(run_id)
    state = manager.load_active_or_latest()
    if state is not None:
        return state
    if create_if_missing:
        return manager.create_run()
    return None


def _arg_run_id(args: argparse.Namespace) -> Optional[str]:
    return getattr(args, "run_id_cmd", None) or getattr(args, "run_id", None)


def _arg_config(args: argparse.Namespace) -> Optional[str]:
    return getattr(args, "config_cmd", None) or getattr(args, "config", None)


def _print_final_state(state: PipelineState) -> None:
    print("final status:")
    print(f"  pipeline_status={state.pipeline_status}, current_stage={state.current_stage}")
    if state.next_action:
        print("next action:")
        print(f"  {state.next_action}")


def _get_stage_state(state: PipelineState, stage_short_name: str):
    stage_id = STAGE_BY_SHORT_NAME[stage_short_name].stage_id
    for stage in state.stages:
        if stage.stage_id == stage_id:
            return stage
    raise ValueError("Stage is missing from pipeline state. Re-run reset.")


STAGE_CLI_TOPLEVEL: FrozenSet[str] = frozenset(STAGE_BY_SHORT_NAME.keys())

# Shell-style placeholder for help text (avoid backslashes inside f-string expressions).
_EXAMPLE_TOPIC_CLI = '"..."'


def _load_dotenv_for_config(config_path: Optional[str]) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if config_path:
        config_dir = Path(config_path).expanduser().resolve().parent
        load_dotenv(config_dir / ".env", override=False)
    load_dotenv(Path.cwd() / ".env", override=False)


def resolve_stage_invocation(args: argparse.Namespace) -> Tuple[str, str]:
    if args.command == "stage":
        return str(args.stage_name), str(args.stage_command)
    if str(args.command) in STAGE_CLI_TOPLEVEL:
        return str(args.command), str(args.stage_command)
    raise ValueError("Internal CLI routing error: expected a stage namespace command.")


def stage_handler_namespace(args: argparse.Namespace) -> argparse.Namespace:
    stage_name, _ = resolve_stage_invocation(args)
    merged = argparse.Namespace(**vars(args))
    merged.stage_name = stage_name
    return merged


def _collect_stage_dir(state: PipelineState) -> Path:
    collect = STAGE_BY_SHORT_NAME["collect"]
    return Path(state.run_dir) / collect.artifact_dir_name


def _truncate_query_hint(text: str, max_len: int = 72) -> str:
    s = str(text).replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _build_data_collection_agent(cfg: Dict[str, Any], stage_dir: Path) -> Any:
    from agents.data_collection_agent import DataCollectionAgent

    merged = dict(cfg)
    storage_cfg = dict(merged.get("storage", {}))
    storage_cfg["artifacts_dir"] = str(stage_dir.resolve())
    merged["storage"] = storage_cfg
    return DataCollectionAgent(config=merged)


def cmd_collect_discover(args: argparse.Namespace, manager: PipelineStateManager) -> int:
    _load_dotenv_for_config(_arg_config(args))
    state = _resolve_state(manager, _arg_run_id(args), create_if_missing=False)
    if state is None:
        print("No pipeline runs found yet.")
        print(f"Next action: {cli('run')}")
        return 0
    stage_dir = _collect_stage_dir(state)
    cfg = load_config(_arg_config(args))
    from pipeline.stages.collect import infer_topic_profile

    try:
        profile = infer_topic_profile(
            cfg,
            topic=getattr(args, "topic", None),
            modality=getattr(args, "modality", None),
            language=getattr(args, "language", None),
            task_type=getattr(args, "task_type", None),
            size_target=getattr(args, "size_target", None),
            needs_labels=getattr(args, "needs_labels", None),
        )
    except ValueError as exc:
        print(exc)
        return 2
    agent = _build_data_collection_agent(cfg, stage_dir)
    agent.session.topic_profile = profile
    candidates = agent.discover_sources(agent.session.topic_profile)
    journal = getattr(agent.discovery_service, "last_journal", None)
    discover_verbose = bool(getattr(args, "verbose", False))
    if journal is not None:
        print("Provider capabilities:")
        for capability in journal.provider_capabilities:
            status = "ok" if capability.available else "unavailable"
            reason = f" ({capability.reason})" if capability.reason else ""
            print(f"  - {capability.provider.value}: {status}{reason}")
        nq = len(journal.queries or [])
        if discover_verbose:
            print(f"queries ({nq}): {journal.queries}")
        elif nq:
            head = list(journal.queries[:6])
            extra = f" … (+{nq - len(head)} more)" if nq > len(head) else ""
            short = ", ".join(_truncate_query_hint(q) for q in head)
            print(f"queries ({nq}): {short}{extra}")
        else:
            print("queries: (none)")
        if journal.used_demo_fallback:
            print("demo_fallback: enabled and used")

    tagged = assign_stable_keys(candidates)
    _, ordered = save_discovery_snapshot(stage_dir, topic_profile=profile, tagged=tagged)
    print_discover_catalog(ordered, verbose=discover_verbose)

    collect_stage = _get_stage_state(state, "collect")
    disc_path = stage_dir / "discovery" / "discovery_snapshot.json"
    manager.record_artifact(state, collect_stage.stage_id, "collect_discovery_snapshot", str(disc_path.resolve()))
    manager.save(state)

    print("next action:")
    print(f"  {cli('collect', 'recommend')}")
    return 0


def cmd_collect_recommend(args: argparse.Namespace, manager: PipelineStateManager) -> int:
    from agents.data_collection.planner import StrategyPlanner

    _load_dotenv_for_config(_arg_config(args))
    state = _resolve_state(manager, _arg_run_id(args), create_if_missing=False)
    if state is None:
        print("No pipeline runs found yet.")
        print(f"Next action: {cli('run')}")
        return 0
    stage_dir = _collect_stage_dir(state)
    snapshot = try_load_discovery_snapshot(stage_dir)
    if snapshot is None:
        print("No discovery snapshot for this run. Run collect discover first, e.g.:")
        print(f"  {cli('collect', 'discover', '--topic', _EXAMPLE_TOPIC_CLI, '--config', 'config.yaml')}")
        return 2

    candidates = [c for _, _, c in snapshot.ordered]
    planner = StrategyPlanner()
    plans = planner.build_plans(snapshot.topic_profile, candidates)
    spec_map = spec_id_to_candidate_number(snapshot)
    combos = build_recommendation_combinations(plans, spec_id_to_number=spec_map)
    print_recommendation_combinations(combos)
    rec_path = save_recommendations(
        stage_dir,
        combinations=combos,
        plans_serialized=serialize_collection_plans(plans),
    )

    collect_stage = _get_stage_state(state, "collect")
    manager.record_artifact(state, collect_stage.stage_id, "collect_recommendations", str(rec_path.resolve()))
    manager.save(state)

    print("next action:")
    print(f"  {cli('collect', 'select', '--ids', '1,2')}")
    return 0


def cmd_collect_select(args: argparse.Namespace, manager: PipelineStateManager) -> int:
    from agents.data_collection.planner import StrategyPlanner

    _load_dotenv_for_config(_arg_config(args))
    state = _resolve_state(manager, _arg_run_id(args), create_if_missing=False)
    if state is None:
        print("No pipeline runs found yet.")
        print(f"Next action: {cli('run')}")
        return 0
    stage_dir = _collect_stage_dir(state)
    snapshot = try_load_discovery_snapshot(stage_dir)
    if snapshot is None:
        print("No discovery snapshot for this run. Run collect discover first, e.g.:")
        print(f"  {cli('collect', 'discover', '--topic', _EXAMPLE_TOPIC_CLI, '--config', 'config.yaml')}")
        return 2

    try:
        nums = parse_candidate_ids(str(getattr(args, "ids", "")))
    except ValueError as exc:
        print(exc)
        return 2

    key_by_num = snapshot.key_by_number()
    valid_numbers = set(key_by_num.keys())
    bad_nums = [n for n in nums if n not in valid_numbers]
    if bad_nums:
        hi = max(valid_numbers) if valid_numbers else 0
        print(f"Invalid candidate id(s): {bad_nums}. Valid ids are 1..{hi} from the last discover output.")
        return 2

    by_key = snapshot.candidates_by_key()
    non_exec: list[tuple[int, str, str]] = []
    seen_ne: set[str] = set()
    for n in nums:
        k = key_by_num[n]
        cand = by_key[k]
        if k in seen_ne:
            continue
        seen_ne.add(k)
        if not cand.is_executable:
            non_exec.append((n, cand.name, cand.non_executable_reason or "discovery-only or no connector"))
    if non_exec:
        print("Cannot select non-executable source(s); fix connectors or choose different ids:")
        for nid, name, reason in non_exec:
            print(f"  - id {nid}: {name} — {reason}")
        exec_nums = sorted(num for num, key in key_by_num.items() if by_key[key].is_executable)
        if exec_nums:
            shown = ", ".join(str(x) for x in exec_nums[:40])
            more = " …" if len(exec_nums) > 40 else ""
            print()
            print(
                f"In this discovery snapshot there are {len(exec_nums)} executable candidate(s). "
                f"Ids: {shown}{more}"
            )
            pair = (
                f"{exec_nums[0]},{exec_nums[1]}" if len(exec_nums) >= 2 else str(exec_nums[0])
            )
            print(
                f"Tip: pick scrape targets only after adding a selector/scraper_spec, or choose HF/GitHub/etc. "
                f"Example: {cli('collect', 'select', '--ids', pair, '--run-id', state.run_id)}"
            )
        return 2

    ordered_keys: list[str] = []
    seen_k: set[str] = set()
    for n in nums:
        k = key_by_num[n]
        if k in seen_k:
            continue
        seen_k.add(k)
        ordered_keys.append(k)

    try:
        StrategyPlanner().build_plan_from_selected_candidates(snapshot.topic_profile, by_key, ordered_keys)
    except ValueError as exc:
        print(exc)
        return 2

    entries: list[dict[str, Any]] = []
    for k in ordered_keys:
        first_n = next(n for n in nums if key_by_num[n] == k)
        cand = by_key[k]
        entries.append(
            {
                "candidate_number": first_n,
                "candidate_key": k,
                "name": cand.name,
                "source_type": cand.source_type.value,
                "executable": cand.is_executable,
            }
        )

    uniq_nums = [e["candidate_number"] for e in entries]
    uniq_keys = list(ordered_keys)
    sel = UserSelection(candidate_numbers=uniq_nums, candidate_keys=uniq_keys, entries=entries)
    sel_path = save_user_selection(stage_dir, sel)
    print(f"Saved selection for {len(uniq_keys)} source(s): {uniq_nums}")
    print(f"  file: {sel_path.resolve()}")

    collect_stage = _get_stage_state(state, "collect")
    manager.record_artifact(state, collect_stage.stage_id, "collect_user_selection", str(sel_path.resolve()))
    manager.save(state)

    print("next action:")
    print(f"  {stage_run('collect')}")
    return 0


def dispatch_stage_namespace(args: argparse.Namespace, manager: PipelineStateManager) -> int:
    """Route ``collect …``, ``quality …``, legacy ``stage …``, etc. to existing handlers."""
    stage_name, action = resolve_stage_invocation(args)
    hargs = stage_handler_namespace(args)
    if stage_name == "collect":
        if action == "discover":
            return cmd_collect_discover(hargs, manager)
        if action == "recommend":
            return cmd_collect_recommend(hargs, manager)
        if action == "select":
            return cmd_collect_select(hargs, manager)
    if action == "status":
        return cmd_stage_status(hargs, manager)
    if action == "run":
        return cmd_stage_run(hargs, manager)
    if action == "review":
        return cmd_stage_review(hargs, manager)
    print(f"Unknown stage action: {action}")
    return 2


def cmd_status(args: argparse.Namespace, manager: PipelineStateManager) -> int:
    state = _resolve_state(manager, _arg_run_id(args), create_if_missing=False)
    if state is None:
        print("No pipeline runs found yet.")
        print(f"Next action: {cli('run')}")
        return 0
    print(render_pipeline_summary(state))
    return 0


def cmd_reset(args: argparse.Namespace, manager: PipelineStateManager) -> int:
    state = manager.create_run(run_id=args.new_run_id)
    print("Initialized new pipeline run.")
    print(render_pipeline_summary(state))
    print("created stage directories:")
    for stage in STAGE_REGISTRY:
        print(f"  - {Path(state.run_dir) / stage.artifact_dir_name}")
    return 0


def cmd_artifacts(args: argparse.Namespace, manager: PipelineStateManager) -> int:
    state = _resolve_state(manager, _arg_run_id(args), create_if_missing=False)
    if state is None:
        print("No pipeline runs found yet.")
        print(f"Next action: {cli('run')}")
        return 0
    print(f"run_id: {state.run_id}")
    print(f"pipeline_status: {state.pipeline_status}")
    print("artifacts by stage:")
    any_artifacts = False
    for stage in sorted(state.stages, key=lambda s: s.ordinal):
        print(f"  {stage.stage_id} {stage.display_name} [{stage.status}]")
        if stage.review_supported:
            print(f"    review_required: {stage.review_required}")
            print(f"    review_file: {stage.review_file or '-'}")
        if not stage.artifacts:
            print("    - (no artifacts yet)")
            continue
        any_artifacts = True
        for key, value in sorted(stage.artifacts.items()):
            print(f"    - {key}: {value}")
    if not any_artifacts:
        print("  (no artifacts recorded yet in any stage)")
    print("artifact tree:")
    run_dir = Path(state.run_dir)
    for stage in sorted(state.stages, key=lambda s: s.ordinal):
        stage_dir = run_dir / stage.artifact_dir_name
        print(f"  - {stage.stage_id}: {stage_dir}")
    print("next action:")
    print(f"  {state.next_action}")
    return 0


def cmd_stage_status(args: argparse.Namespace, manager: PipelineStateManager) -> int:
    state = _resolve_state(manager, _arg_run_id(args), create_if_missing=False)
    if state is None:
        print("No pipeline runs found yet.")
        print(f"Next action: {cli('run')}")
        return 0
    stage = _get_stage_state(state, args.stage_name)
    state.current_stage = stage.stage_id
    print(render_current_stage_summary(state))
    if args.stage_name == "quality":
        print("quality details:")
        print(f"  input_dataset: {stage.artifacts.get('quality_input_dataset', '(not set)')}")
        print(f"  review_status: {'required' if stage.review_required else 'not required'}")
        print(f"  review_decision: {stage.review_file or '(not set)'}")
        print(f"  final_output: {stage.artifacts.get('stage_cleaned_output', '(not set)')}")
    if args.stage_name == "annotate":
        print("annotate details:")
        print(f"  input_dataset: {stage.artifacts.get('annotate_input_dataset', '(not set)')}")
        print(f"  review_queue: {stage.artifacts.get('annotate_review_queue_csv', '(not set)')}")
        print(f"  review_file: {stage.review_file or '(not set)'}")
        print(f"  final_output: {stage.artifacts.get('annotate_final_reviewed', '(not set)')}")
    if args.stage_name == "al":
        print("al details:")
        print(f"  input_dataset: {stage.artifacts.get('al_input_dataset', '(not set)')}")
        print(f"  review_queue: {stage.artifacts.get('al_review_queue_csv', '(not set)')}")
        print(f"  review_file: {stage.review_file or '(not set)'}")
        print(f"  final_output: {stage.artifacts.get('al_final_dataset', '(not set)')}")
    if args.stage_name == "train":
        print("train details:")
        print(f"  input_dataset: {stage.artifacts.get('train_input_dataset', '(not set)')}")
        print(f"  model_info: {stage.artifacts.get('train_model_info_json', '(not set)')}")
        print(f"  metrics: {stage.artifacts.get('train_metrics_json', '(not set)')}")
        print(f"  classification_report: {stage.artifacts.get('train_classification_report_txt', '(not set)')}")
    if args.stage_name == "report":
        print("report details:")
        print(f"  final_report: {stage.artifacts.get('final_pipeline_report_md', '(not set)')}")
    if args.stage_name == "collect":
        stage_dir = _collect_stage_dir(state)
        print(format_collect_status_block(stage_dir, stage))
    print("next action:")
    print(f"  {state.next_action}")
    return 0


def cmd_stage_run(args: argparse.Namespace, manager: PipelineStateManager) -> int:
    state = _resolve_state(manager, _arg_run_id(args), create_if_missing=False)
    if state is None:
        print("No pipeline runs found yet.")
        print(f"Next action: {cli('run')}")
        return 0
    # Load `.env` next to config and CWD so `GITHUB_TOKEN`, Kaggle keys, etc. apply to `run` / `quality run`, …
    _load_dotenv_for_config(_arg_config(args))
    stage_def = STAGE_BY_SHORT_NAME[args.stage_name]
    stage = _get_stage_state(state, args.stage_name)
    stage_dir = Path(state.run_dir) / stage_def.artifact_dir_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    manager.set_current_stage(state, stage_def.stage_id)

    if args.stage_name == "collect":
        from pipeline.stages.collect import infer_topic_profile, run_collect_stage

        cfg = load_config(_arg_config(args))
        coll = cfg.get("collection") or {}
        interactive = bool(
            getattr(args, "interactive_collect", False)
            or coll.get("interactive")
            or coll.get("interactive_collect")
        )
        print(f"[1/6] {stage_def.display_name} — DataCollectionAgent")
        _collect_note = (
            "COLLECT: terminal Q&A for topic profile (customer)."
            if interactive
            else "Running DataCollectionAgent with profile from config/CLI."
        )
        manager.update_stage_status(state, stage_def.stage_id, "running", note=_collect_note)
        manager.save(state)
        sel_path = stage_dir / "selection" / "user_selection.json"
        if sel_path.exists():
            profile = None
        elif interactive:
            profile = None
        else:
            try:
                profile = infer_topic_profile(
                    cfg,
                    topic=getattr(args, "topic", None),
                    modality=getattr(args, "modality", None),
                    language=getattr(args, "language", None),
                    task_type=getattr(args, "task_type", None),
                    size_target=getattr(args, "size_target", None),
                    needs_labels=getattr(args, "needs_labels", None),
                )
            except ValueError as exc:
                stage.last_error = str(exc)
                manager.update_stage_status(state, stage_def.stage_id, "failed", note=f"COLLECT failed: {exc}")
                manager.save(state)
                print(f"COLLECT failed: {exc}")
                return 1
        try:
            result = run_collect_stage(
                config_payload=cfg,
                stage_dir=stage_dir,
                profile=profile,
                interactive_terminal=interactive,
            )
        except Exception as exc:
            stage.last_error = str(exc)
            manager.update_stage_status(state, stage_def.stage_id, "failed", note=f"COLLECT failed: {exc}")
            manager.save(state)
            print(f"COLLECT failed: {exc}")
            return 1
        manager.record_artifacts(state, stage_def.stage_id, result.artifact_paths)
        manager.update_stage_status(state, stage_def.stage_id, "completed", note=f"COLLECT completed: {result.row_count} rows from {result.source_count} sources.")
        manager.set_current_stage(state, STAGE_BY_SHORT_NAME["quality"].stage_id)
        manager.set_next_action(state, quality_run_input_placeholder())
        manager.save(state)
        if result.terminal_summary:
            print(result.terminal_summary.rstrip("\n"))
        else:
            print("COLLECT completed.")
            print(f"merged_output: {result.merged_output_path}")
            print("next action:")
            print(f"  {state.next_action}")
        return 0

    if args.stage_name == "annotate":
        from pipeline.stages.annotate import run_annotate_stage

        input_str = getattr(args, "input", None) or resolve_quality_output_path(state)
        if not input_str:
            print("ANNOTATE run needs QUALITY output dataset, but none was found.")
            print("Run QUALITY review first or pass --input <csv|parquet>.")
            return 2
        input_path = Path(input_str)
        if not input_path.is_absolute():
            input_path = Path.cwd() / input_path
        if not input_path.exists():
            print(f"Input not found: {input_path}")
            return 2
        cfg = load_config(_arg_config(args))
        skip_ann, skip_reason = skip_annotation_policy(cfg)
        if skip_ann:
            print(f"[3/6] {stage_def.display_name} — skipped")
            print(skip_reason)
            ann_dir = Path(state.run_dir) / stage_def.artifact_dir_name
            apply_annotation_stage_skip(
                manager,
                state,
                passthrough_dataset_path=str(input_path.resolve()),
                annotate_stage_dir=ann_dir,
                reason=skip_reason,
            )
            print(f"passthrough_dataset: {input_path.resolve()}")
            print("next action:")
            print(f"  {state.next_action}")
            return 0
        print(f"[3/6] {stage_def.display_name} — AnnotationAgent")
        try:
            input_df = read_dataframe(input_path)
        except Exception as exc:
            print(f"Failed to read annotate input: {exc}")
            return 2
        if len(input_df) == 0:
            alt_str = resolve_quality_input_path(state)
            alt_path = Path(alt_str) if alt_str else None
            if alt_path and not alt_path.is_absolute():
                alt_path = Path.cwd() / alt_path
            primary_resolved = input_path.resolve()
            if (
                alt_path
                and alt_path.is_file()
                and alt_path.resolve() != primary_resolved
            ):
                try:
                    alt_df = read_dataframe(alt_path)
                except Exception:
                    alt_df = None
                if alt_df is not None and len(alt_df) > 0:
                    print(
                        f"WARNING: QUALITY cleaned output has 0 rows ({primary_resolved}). "
                        f"Using upstream dataset ({alt_path.resolve()}) with {len(alt_df)} rows for ANNOTATE. "
                        "Re-run QUALITY with a gentler decision or fix upstream data if this is unintended.",
                        flush=True,
                    )
                    input_df = alt_df
                    input_path = alt_path
            if len(input_df) == 0:
                tried = [str(primary_resolved)]
                if alt_path and alt_path.is_file() and alt_path.resolve() != primary_resolved:
                    tried.append(str(alt_path.resolve()))
                print(
                    "ANNOTATE: input dataframe has no rows.\n"
                    f"  Tried: {tried}\n"
                    "  QUALITY likely removed every row (strict filters / drop-all strategy). "
                    "Use a less aggressive quality decision, or pass a non-empty dataset explicitly, e.g.\n"
                    f"  python run_pipeline.py annotate run --input <path/to/merged_dataset.parquet>"
                )
                stage.last_error = "annotate_input_empty"
                manager.update_stage_status(
                    state,
                    stage_def.stage_id,
                    "failed",
                    note="ANNOTATE failed: input has 0 rows (quality output empty; see CLI message).",
                )
                manager.save(state)
                return 1
        manager.update_stage_status(state, stage_def.stage_id, "running", note=f"Running ANNOTATE on {input_path}")
        manager.record_artifact(state, stage_def.stage_id, "annotate_input_dataset", str(input_path.resolve()))
        manager.save(state)
        try:
            result = run_annotate_stage(config_payload=cfg, stage_dir=stage_dir, input_df=input_df)
        except Exception as exc:
            stage.last_error = str(exc)
            manager.update_stage_status(state, stage_def.stage_id, "failed", note=f"ANNOTATE failed: {exc}")
            manager.save(state)
            print(f"ANNOTATE failed: {exc}")
            return 1
        manager.record_artifacts(state, stage_def.stage_id, result.artifact_paths)
        if result.review_needed:
            stage.review_required = True
            stage.review_file = result.review_queue_path
            manager.update_stage_status(state, stage_def.stage_id, "awaiting_review", note=f"Review required for {result.review_count} low-confidence examples.")
            manager.set_next_action(state, annotate_review_file_placeholder())
            manager.save(state)
            print("ANNOTATE review artifacts created:")
            print(f"  - review_queue: {result.review_queue_path}")
            print("expected columns (CSV round-trip):")
            print("  - annotation_id, auto_label, confidence, review_reason, human_label")
            print(
                "Label Studio: import review/labelstudio_import.json (or review_queue_labelstudio.json); "
                "paste label_config.xml in the UI; export JSON and use annotate review --file …"
            )
            hr_hint = load_config(_arg_config(args)).get("human_review") or {}
            rel_hint = hr_hint.get("corrected_file") or "review/review_queue_corrected.csv"
            print(f"After review: save CSV/JSON then  annotate review --file …  or place file at {(Path(state.run_dir) / stage_def.artifact_dir_name / rel_hint).resolve()}")
            print("stage status: awaiting_review")
            print("next action:")
            print(f"  {state.next_action}")
            return 0
        stage.review_required = False
        manager.update_stage_status(state, stage_def.stage_id, "completed", note="ANNOTATE completed without manual review.")
        manager.set_current_stage(state, STAGE_BY_SHORT_NAME["al"].stage_id)
        manager.set_next_action(state, stage_run("al"))
        manager.save(state)
        print("ANNOTATE completed (no review required).")
        print("next action:")
        print(f"  {state.next_action}")
        return 0

    if args.stage_name == "al":
        from pipeline.stages.al import run_al_stage

        print(f"[4/6] {stage_def.display_name} — ActiveLearningAgent")
        input_str = getattr(args, "input", None) or resolve_annotate_output_path(state)
        if not input_str:
            print("AL run needs ANNOTATE output dataset, but none was found.")
            print("Run ANNOTATE review first or pass --input <csv|parquet>.")
            return 2
        input_path = Path(input_str)
        if not input_path.is_absolute():
            input_path = Path.cwd() / input_path
        if not input_path.exists():
            print(f"Input not found: {input_path}")
            return 2
        try:
            input_df = read_dataframe(input_path)
        except Exception as exc:
            print(f"Failed to read AL input: {exc}")
            return 2
        manager.update_stage_status(state, stage_def.stage_id, "running", note=f"Running AL on {input_path}")
        manager.record_artifact(state, stage_def.stage_id, "al_input_dataset", str(input_path.resolve()))
        manager.save(state)
        try:
            cfg_al = load_config(_arg_config(args))
            result = run_al_stage(stage_dir=stage_dir, annotated_df=input_df, config_payload=cfg_al)
        except Exception as exc:
            stage.last_error = str(exc)
            manager.update_stage_status(state, stage_def.stage_id, "failed", note=f"AL failed: {exc}")
            manager.save(state)
            print(f"AL failed: {exc}")
            return 1
        manager.record_artifacts(state, stage_def.stage_id, result.artifact_paths)
        if result.review_needed:
            stage.review_required = True
            stage.review_file = result.review_queue_path
            manager.update_stage_status(state, stage_def.stage_id, "awaiting_review", note=f"Manual labeling required for {result.selected_count} AL-selected examples.")
            manager.set_next_action(state, al_review_file_placeholder())
            manager.save(state)
            print("AL review artifacts created:")
            print(f"  - review_queue: {result.review_queue_path}")
            print("expected columns:")
            print("  - id, text, human_label")
            print("stage status: awaiting_review")
            print("next action:")
            print(f"  {state.next_action}")
            return 0
        stage.review_required = False
        manager.update_stage_status(state, stage_def.stage_id, "completed", note="AL completed without manual review.")
        manager.set_current_stage(state, STAGE_BY_SHORT_NAME["train"].stage_id)
        manager.set_next_action(state, stage_run("train"))
        manager.save(state)
        print("AL completed (no review required).")
        print("next action:")
        print(f"  {state.next_action}")
        return 0

    if args.stage_name == "train":
        from pipeline.stages.train import run_train_stage

        print(f"[5/6] {stage_def.display_name} — Trainer")
        input_str = getattr(args, "input", None) or resolve_train_input_path(state)
        if not input_str:
            print("TRAIN run needs labeled dataset from AL/ANNOTATE, but none was found.")
            print("Run AL review first or pass --input <csv|parquet>.")
            return 2
        input_path = Path(input_str)
        if not input_path.is_absolute():
            input_path = Path.cwd() / input_path
        if not input_path.exists():
            print(f"Input not found: {input_path}")
            return 2
        try:
            input_df = read_dataframe(input_path)
        except Exception as exc:
            print(f"Failed to read TRAIN input: {exc}")
            return 2
        manager.update_stage_status(state, stage_def.stage_id, "running", note=f"Running TRAIN on {input_path}")
        manager.record_artifact(state, stage_def.stage_id, "train_input_dataset", str(input_path.resolve()))
        manager.save(state)
        try:
            cfg_train = load_config(_arg_config(args))
            result = run_train_stage(stage_dir=stage_dir, dataset_df=input_df, config_payload=cfg_train)
        except Exception as exc:
            stage.last_error = str(exc)
            manager.update_stage_status(state, stage_def.stage_id, "failed", note=f"TRAIN failed: {exc}")
            manager.save(state)
            print(f"TRAIN failed: {exc}")
            return 1
        manager.record_artifacts(state, stage_def.stage_id, result.artifact_paths)
        manager.update_stage_status(state, stage_def.stage_id, "completed", note="TRAIN completed.")
        manager.set_current_stage(state, STAGE_BY_SHORT_NAME["report"].stage_id)
        manager.set_next_action(state, stage_run("report"))
        manager.save(state)
        print("TRAIN completed.")
        print(f"model_info: {result.model_info_path}")
        print(f"metrics: {result.metrics_path}")
        print("next action:")
        print(f"  {state.next_action}")
        return 0

    if args.stage_name == "report":
        from pipeline.stages.report import run_report_stage

        print(f"[6/6] {stage_def.display_name} — Final Reporter")
        manager.update_stage_status(state, stage_def.stage_id, "running", note="Generating final report.")
        manager.save(state)
        try:
            result = run_report_stage(stage_dir=stage_dir, state=state)
        except Exception as exc:
            stage.last_error = str(exc)
            manager.update_stage_status(state, stage_def.stage_id, "failed", note=f"REPORT failed: {exc}")
            manager.save(state)
            print(f"REPORT failed: {exc}")
            return 1
        manager.record_artifacts(state, stage_def.stage_id, result.artifact_paths)
        manager.update_stage_status(state, stage_def.stage_id, "completed", note="REPORT completed.")
        manager.set_next_action(state, pipeline_status())
        manager.save(state)
        print("REPORT completed.")
        print(f"final_report: {result.report_path}")
        print("next action:")
        print(f"  {state.next_action}")
        return 0

    if args.stage_name != "quality":
        manager.update_stage_status(state, stage_def.stage_id, "pending", note=f"{stage_def.display_name} stage run is not implemented yet.")
        manager.set_next_action(state, stage_run(args.stage_name))
        manager.save(state)
        print(f"{stage_def.display_name} stage run is not implemented yet.")
        print(f"stage_dir: {stage_dir}")
        print("next action:")
        print(f"  {state.next_action}")
        return 0

    from pipeline.stages.quality import build_quality_agent

    print(f"[2/6] {stage_def.display_name} — DataQualityAgent")
    input_str = getattr(args, "input", None) or resolve_quality_input_path(state)
    if not input_str:
        print("QUALITY run needs an input dataset, but none was found.")
        print(f"Provide one explicitly: {quality_run_input_placeholder()} (.csv or .parquet)")
        print("or run COLLECT stage first.")
        return 2
    input_path = Path(input_str)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 2

    decision_payload: Optional[Union[Dict[str, Any], str, Path]] = None
    decision_arg = getattr(args, "decision", None)
    if decision_arg:
        decision_path = Path(decision_arg)
        if not decision_path.is_absolute():
            decision_path = Path.cwd() / decision_path
        if not decision_path.exists():
            print(f"Decision file not found: {decision_path}")
            return 2
        decision_payload = decision_path

    try:
        raw_df = read_dataframe(input_path)
    except Exception as exc:
        print(f"Failed to read input dataset: {exc}")
        return 2

    manager.update_stage_status(state, stage_def.stage_id, "running", note=f"Running QUALITY on {input_path}")
    manager.record_artifact(state, stage_def.stage_id, "quality_input_dataset", str(input_path.resolve()))
    manager.save(state)
    print(f"Running QUALITY stage on {input_path} ...")
    agent = build_quality_agent(_arg_config(args), Path(state.run_dir))

    try:
        result = agent.run_stage(
            raw_df,
            decision=decision_payload,
            task_description=getattr(args, "task_description", None),
            raise_on_awaiting_review=False,
        )
    except Exception as exc:
        stage.last_error = str(exc)
        manager.update_stage_status(state, stage_def.stage_id, "failed", note=f"QUALITY failed: {exc}")
        manager.save(state)
        print(f"QUALITY stage failed: {exc}")
        return 1

    if isinstance(result, QualityStageResult):
        stage.review_required = True
        stage.review_file = result.decision_template_json or None
        manager.record_artifacts(state, stage_def.stage_id, result.artifacts)
        manager.update_stage_status(state, stage_def.stage_id, "awaiting_review", note=result.message)
        manager.set_next_action(state, quality_review_decision_placeholder())
        manager.save(state)
        print("review artifacts created:")
        for key, value in sorted(result.artifacts.items()):
            if "review" in key or "decision" in key:
                print(f"  - {key}: {value}")
        print("stage status: awaiting_review")
        print("next action:")
        print(f"  {state.next_action}")
        return 0

    cleaned_path = stage_dir / "cleaned_output.parquet"
    export_df = result.copy()
    export_df.attrs.clear()
    export_df.to_parquet(cleaned_path, index=False)
    stage.review_required = False
    stage.review_file = None
    manager.record_artifacts(state, stage_def.stage_id, {**agent.last_artifacts, "stage_cleaned_output": str(cleaned_path.resolve())})
    manager.update_stage_status(state, stage_def.stage_id, "completed", note="QUALITY completed and final artifacts saved.")
    cfg_after_quality = load_config(_arg_config(args))
    skipped_ann, ann_skip_reason = _maybe_skip_annotate_after_quality(manager, state, cfg_after_quality, cleaned_path)
    if skipped_ann:
        print("QUALITY stage completed.")
        print(f"cleaned_output: {cleaned_path.resolve()}")
        print("ANNOTATE stage skipped. See 03_annotate/reports/annotation_skipped.md.")
        print(ann_skip_reason)
        print("next action:")
        print(f"  {state.next_action}")
        return 0
    manager.set_next_action(state, stage_run("annotate"))
    manager.save(state)
    print("QUALITY stage completed.")
    print(f"cleaned_output: {cleaned_path.resolve()}")
    print("next action:")
    print(f"  {state.next_action}")
    return 0


def cmd_stage_review(args: argparse.Namespace, manager: PipelineStateManager) -> int:
    state = _resolve_state(manager, _arg_run_id(args), create_if_missing=False)
    if state is None:
        print("No pipeline runs found yet.")
        print(f"Next action: {cli('run')}")
        return 0
    stage_def = STAGE_BY_SHORT_NAME[args.stage_name]
    stage = _get_stage_state(state, args.stage_name)

    if args.stage_name == "quality":
        from pipeline.stages.quality import build_quality_agent

        raw_decision = Path(args.decision)
        if raw_decision.is_absolute():
            decision_candidates = [raw_decision]
        else:
            decision_candidates = [Path.cwd() / raw_decision]
            # Typical typo: runs/<run_id>/... vs artifacts/runs/<run_id>/...
            if raw_decision.parts and raw_decision.parts[0] == "runs":
                decision_candidates.append(Path.cwd() / "artifacts" / raw_decision)
        decision_path = next((p for p in decision_candidates if p.exists()), None)
        if decision_path is None:
            print("Decision file not found. Tried:")
            for p in decision_candidates:
                print(f"  - {p}")
            return 2
        try:
            decision_payload = json.loads(decision_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Failed to parse decision JSON: {exc}")
            return 2
        approved = bool(decision_payload.get("approved"))
        manager.record_review_file(state, stage_def.stage_id, str(decision_path.resolve()))
        manager.record_artifact(state, stage_def.stage_id, "quality_decision", str(decision_path.resolve()))
        if not approved:
            manager.update_stage_status(state, stage_def.stage_id, "awaiting_review", note="Decision present but approved=false.")
            manager.set_next_action(state, quality_review_decision_placeholder())
            print("QUALITY remains awaiting_review (approved=false).")
            manager.save(state)
            print("next action:")
            print(f"  {state.next_action}")
            return 0

        input_str = resolve_quality_input_path(state)
        if not input_str:
            print("Could not resolve input dataset for QUALITY review apply step.")
            print(f"Set it via `{cli('quality', 'run')} --input ...` first.")
            return 2
        input_path = Path(input_str)
        if not input_path.is_absolute():
            input_path = Path.cwd() / input_path
        if not input_path.exists():
            print(f"Input dataset for QUALITY not found: {input_path}")
            return 2
        print(f"QUALITY review: loading {input_path} ...", flush=True)
        try:
            raw_df = read_dataframe(input_path)
        except Exception as exc:
            print(f"Failed to read quality input dataset: {exc}")
            return 2
        print(
            "QUALITY review: applying approved decision (fix, compare, save) — may take several minutes on wide tables ...",
            flush=True,
        )

        manager.update_stage_status(state, stage_def.stage_id, "running", note="Applying approved quality decision.")
        manager.save(state)
        agent = build_quality_agent(_arg_config(args), Path(state.run_dir))
        try:
            cleaned_df = agent.run_stage(raw_df, decision=decision_payload, raise_on_awaiting_review=False)
        except Exception as exc:
            stage.last_error = str(exc)
            manager.update_stage_status(state, stage_def.stage_id, "failed", note=f"QUALITY review apply failed: {exc}")
            manager.save(state)
            print(f"QUALITY review apply failed: {exc}")
            return 1
        if isinstance(cleaned_df, QualityStageResult):
            manager.update_stage_status(state, stage_def.stage_id, "awaiting_review", note=cleaned_df.message)
            manager.record_artifacts(state, stage_def.stage_id, cleaned_df.artifacts)
            manager.save(state)
            print("QUALITY still awaits review.")
            print("next action:")
            print(f"  {state.next_action}")
            return 0

        out_dir = Path(state.run_dir) / stage_def.artifact_dir_name / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        final_path = out_dir / "cleaned_final_pipeline.parquet"
        out_df = cleaned_df.copy()
        out_df.attrs.clear()
        out_df.to_parquet(final_path, index=False)
        manager.record_artifacts(state, stage_def.stage_id, agent.last_artifacts)
        manager.record_artifact(state, stage_def.stage_id, "stage_cleaned_output", str(final_path.resolve()))
        stage.review_required = False
        manager.update_stage_status(state, stage_def.stage_id, "completed", note="QUALITY approved and applied.")
        cfg_review = load_config(_arg_config(args))
        skipped_ann_r, ann_skip_reason_r = _maybe_skip_annotate_after_quality(
            manager, state, cfg_review, final_path
        )
        if skipped_ann_r:
            print("QUALITY review decision applied successfully.")
            print(f"final_output: {final_path.resolve()}")
            print("ANNOTATE stage skipped. See 03_annotate/reports/annotation_skipped.md.")
            print(ann_skip_reason_r)
            print("next action:")
            print(f"  {state.next_action}")
            return 0
        manager.set_current_stage(state, STAGE_BY_SHORT_NAME["annotate"].stage_id)
        manager.set_next_action(state, stage_run("annotate"))
        manager.save(state)
        print("QUALITY review decision applied successfully.")
        print(f"final_output: {final_path.resolve()}")
        print("next action:")
        print(f"  {state.next_action}")
        return 0

    if args.stage_name in {"annotate", "al"}:
        stage_dir = Path(state.run_dir) / stage_def.artifact_dir_name
        review_file_arg = getattr(args, "file", None)
        if not review_file_arg:
            cfg_hitl = load_config(_arg_config(args))
            hr = dict(cfg_hitl.get("human_review") or {})
            rel = hr.get("corrected_file") or "review/review_queue_corrected.csv"
            primary = (stage_dir / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
            candidates = [primary]
            if args.stage_name == "al":
                from pipeline.stages.al import al_find_latest_corrected_csv

                latest_corr = al_find_latest_corrected_csv(stage_dir)
                if latest_corr is not None:
                    candidates.append(latest_corr.resolve())
                candidates.append((stage_dir / "review" / "iteration_01" / "review_queue_al_corrected.csv").resolve())
            found = next((p for p in candidates if p.exists()), None)
            if found:
                review_file_arg = str(found)
                print(f"Using corrected reviews file: {found}")
            else:
                print(
                    "Missing review file. Pass --file <path> with either:\n"
                    "  - ANNOTATE: CSV/Parquet (annotation_id + human_label) or Label Studio export JSON\n"
                    "  - AL: CSV with id + human_label\n"
                    f"Or save corrections at one of: {', '.join(str(p) for p in candidates)}"
                )
                return 2
        review_file = Path(review_file_arg)
        if not review_file.is_absolute():
            review_file = Path.cwd() / review_file
        if not review_file.exists():
            print(f"Review file not found: {review_file}")
            return 2
        manager.record_review_file(state, stage_def.stage_id, str(review_file.resolve()))
        manager.record_artifact(state, stage_def.stage_id, f"{args.stage_name}_review_file", str(review_file.resolve()))
        if args.stage_name == "annotate":
            from pipeline.stages.annotate import apply_annotate_review, load_annotate_review_corrections

            auto_path = _get_stage_state(state, "annotate").artifacts.get("annotate_auto_labeled")
            if not auto_path:
                print("ANNOTATE review apply failed: missing auto-labeled dataset path.")
                return 2
            auto_df = read_dataframe(Path(auto_path))
            cfg = load_config(_arg_config(args))
            reviewed_df = load_annotate_review_corrections(review_file, config_payload=cfg)
            try:
                _, artifacts = apply_annotate_review(
                    config_payload=cfg,
                    stage_dir=Path(state.run_dir) / stage_def.artifact_dir_name,
                    auto_labeled_df=auto_df,
                    reviewed_df=reviewed_df,
                )
            except Exception as exc:
                stage.last_error = str(exc)
                manager.update_stage_status(state, stage_def.stage_id, "failed", note=f"ANNOTATE review apply failed: {exc}")
                manager.save(state)
                print(f"ANNOTATE review apply failed: {exc}")
                return 1
            manager.record_artifacts(state, stage_def.stage_id, artifacts)
            stage.review_required = False
            manager.update_stage_status(state, stage_def.stage_id, "completed", note="ANNOTATE review applied and completed.")
            manager.set_current_stage(state, STAGE_BY_SHORT_NAME["al"].stage_id)
            manager.set_next_action(state, stage_run("al"))
            manager.save(state)
            print("ANNOTATE review applied successfully.")
            print(f"final_output: {artifacts.get('annotate_final_reviewed')}")
            print("next action:")
            print(f"  {state.next_action}")
            return 0

        from pipeline.stages.al import apply_al_review

        al_input = _get_stage_state(state, "al").artifacts.get("al_input_dataset") or resolve_annotate_output_path(state)
        if not al_input:
            print("AL review apply failed: missing AL input dataset.")
            return 2
        base_df = read_dataframe(Path(al_input))
        reviewed_df = read_dataframe(review_file)
        try:
            cfg_al = load_config(_arg_config(args))
            al_result = apply_al_review(
                stage_dir=Path(state.run_dir) / stage_def.artifact_dir_name,
                base_annotated_df=base_df,
                reviewed_queue_df=reviewed_df,
                config_payload=cfg_al,
            )
        except Exception as exc:
            stage.last_error = str(exc)
            manager.update_stage_status(state, stage_def.stage_id, "failed", note=f"AL review apply failed: {exc}")
            manager.save(state)
            print(f"AL review apply failed: {exc}")
            return 1
        artifacts = al_result.artifact_paths
        manager.record_artifacts(state, stage_def.stage_id, artifacts)
        if not al_result.al_complete:
            from agents.al_agent import ALConfig

            stage.review_required = True
            next_queue = artifacts.get("al_review_queue_csv")
            if next_queue:
                stage.review_file = next_queue
            manager.update_stage_status(
                state,
                stage_def.stage_id,
                "awaiting_review",
                note="AL loop: another labeling batch is ready.",
            )
            manager.set_next_action(state, al_review_file_placeholder())
            manager.save(state)
            print("AL review batch applied; more iterations pending.")
            if next_queue:
                pq = Path(next_queue)
                print(f"  - next review_queue: {next_queue}")
                print(f"  - save corrections to: {pq.parent / ALConfig().corrected_review_filename}")
            print("next action:")
            print(f"  {state.next_action}")
            return 0

        stage.review_required = False
        manager.update_stage_status(state, stage_def.stage_id, "completed", note="AL review applied and completed.")
        manager.set_current_stage(state, STAGE_BY_SHORT_NAME["train"].stage_id)
        manager.set_next_action(state, stage_run("train"))
        manager.save(state)
        print("AL review applied successfully.")
        print(f"final_output: {artifacts.get('al_final_dataset')}")
        print("next action:")
        print(f"  {state.next_action}")
        return 0

    print(f"{stage_def.display_name} review is not implemented yet.")
    return 0


def cmd_run(args: argparse.Namespace, manager: PipelineStateManager) -> int:
    state = _resolve_state(manager, _arg_run_id(args), create_if_missing=True)
    if state is None:
        print("Could not initialize pipeline state.")
        return 1
    while True:
        state = _resolve_state(manager, _arg_run_id(args), create_if_missing=False)
        if state is None:
            print("Could not load pipeline state.")
            return 1
        awaiting = next((s for s in sorted(state.stages, key=lambda s: s.ordinal) if s.status == "awaiting_review"), None)
        if awaiting is not None:
            print(f"Pipeline is awaiting review at stage {awaiting.display_name}.")
            if awaiting.short_name == "quality":
                print(f"Use: {quality_review_decision_placeholder()}")
            elif awaiting.short_name in {"annotate", "al"}:
                print(f"Use: {cli(awaiting.short_name, 'review', '--file', '<path>')}")
            _print_final_state(state)
            return 0
        failed = next((s for s in sorted(state.stages, key=lambda s: s.ordinal) if s.status == "failed"), None)
        if failed is not None:
            print(f"Pipeline stopped: stage {failed.display_name} failed.")
            if failed.last_error:
                print(f"error: {failed.last_error}")
            _print_final_state(state)
            return 1
        next_stage = next((s for s in sorted(state.stages, key=lambda s: s.ordinal) if s.status in {"pending", "approved"}), None)
        if next_stage is None:
            print("Pipeline completed.")
            _print_final_state(state)
            return 0
        fake = argparse.Namespace(**vars(args))
        fake.stage_name = next_stage.short_name
        rc = cmd_stage_run(fake, manager)
        if rc != 0:
            state_after = _resolve_state(manager, _arg_run_id(args), create_if_missing=False)
            if state_after is not None:
                _print_final_state(state_after)
            return rc


def cmd_resume(args: argparse.Namespace, manager: PipelineStateManager) -> int:
    state = _resolve_state(manager, _arg_run_id(args), create_if_missing=False)
    if state is None:
        print("No pipeline runs found to resume.")
        print(f"Start a new run with: {cli('run', '--config', '<path>')}")
        return 0
    pending = any(s.status in {"pending", "approved"} for s in state.stages)
    awaiting = any(s.status == "awaiting_review" for s in state.stages)
    if not pending and not awaiting:
        print("Nothing to resume: all stages are already completed/skipped.")
        _print_final_state(state)
        return 0
    return cmd_run(args, manager)


def _add_collect_topic_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--topic", default=None, help="Topic/domain for data collection.")
    parser.add_argument("--modality", default=None, help="Data modality (e.g. text).")
    parser.add_argument("--language", default=None, help="Dataset language.")
    parser.add_argument("--task-type", default=None, dest="task_type", help="Task type (e.g. classification).")
    parser.add_argument("--size-target", default=None, type=int, dest="size_target", help="Desired row count.")
    parser.add_argument(
        "--needs-labels",
        default=None,
        type=lambda x: str(x).strip().lower() in {"1", "true", "yes", "y", "да"},
        dest="needs_labels",
        help="Whether pre-labeled data is required (true/false).",
    )


def _stage_parser_description(stage: StageDefinition, *, legacy: bool) -> str:
    lines = [
        f"{stage.display_name} stage.",
        "",
        "Examples:",
        f"  {cli(stage.short_name, 'status', '--run-id', '<run_id>')}",
        f"  {cli(stage.short_name, 'run', '--run-id', '<run_id>')}",
    ]
    if stage.short_name == "collect":
        lines.extend(
            [
                f"  {cli('collect', 'discover', '--topic', _EXAMPLE_TOPIC_CLI, '--config', 'config.yaml')}",
                f"  {cli('collect', 'recommend')}",
                f"  {cli('collect', 'select', '--ids', '1,2')}",
            ]
        )
    if legacy:
        lines.append("")
        lines.append("Prefer the top-level command without `stage` (same behavior).")
    return "\n".join(lines)


def register_stage_nested_parsers(nested: Any, stage: StageDefinition) -> None:
    """Attach stage subcommands (status / run / review, plus collect discover|recommend|select)."""
    status_p = nested.add_parser("status", help=f"Show {stage.display_name} stage status.")
    status_p.add_argument("--run-id", default=None, dest="run_id_cmd", help="Specific run id override.")

    if stage.short_name == "collect":
        discover_p = nested.add_parser(
            "discover",
            help="Discover sources and write a snapshot (--topic or config topic required; other fields optional).",
        )
        discover_p.add_argument("--run-id", default=None, dest="run_id_cmd", help="Specific run id override.")
        discover_p.add_argument("--config", default=None, dest="config_cmd", help="Config path override.")
        discover_p.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Print full discovery rationale, scores, risks, and all query strings (default: name + resource only).",
        )
        _add_collect_topic_args(discover_p)
        recommend_p = nested.add_parser(
            "recommend",
            help="Build 1–3 recommended combinations from the last discovery snapshot (no re-discovery).",
        )
        recommend_p.add_argument("--run-id", default=None, dest="run_id_cmd", help="Specific run id override.")
        select_p = nested.add_parser(
            "select",
            help="Choose executable sources by discover id numbers before collect run.",
        )
        select_p.add_argument(
            "--ids",
            required=True,
            help="Comma-separated candidate numbers from discover output, e.g. 1,2,4",
        )
        select_p.add_argument("--run-id", default=None, dest="run_id_cmd", help="Specific run id override.")

    run_sub = nested.add_parser("run", help=f"Run {stage.display_name} stage.")
    run_sub.add_argument("--run-id", default=None, dest="run_id_cmd", help="Specific run id override.")
    run_sub.add_argument("--config", default=None, dest="config_cmd", help="Config path override.")
    if stage.short_name in {"annotate", "al"}:
        run_sub.add_argument("--input", default=None, help="Optional input dataset override (.csv/.parquet).")
    if stage.short_name == "train":
        run_sub.add_argument("--input", default=None, help="Optional training input dataset override (.csv/.parquet).")
    if stage.short_name == "collect":
        _add_collect_topic_args(run_sub)
        run_sub.add_argument(
            "--interactive",
            action="store_true",
            dest="interactive_collect",
            help="Ask for topic, modality, language, etc. in the terminal (customer Q&A) instead of inferring from config.",
        )
    if stage.short_name == "quality":
        run_sub.add_argument("--input", default=None, help="Input dataframe path (.csv/.parquet).")
        run_sub.add_argument("--decision", default=None, help="Decision JSON path.")
        run_sub.add_argument("--task-description", default=None, help="Task description for recommendation artifact.")
    if stage.review_supported:
        review = nested.add_parser("review", help=f"Register review decision/file for {stage.display_name}.")
        review.add_argument("--run-id", default=None, dest="run_id_cmd", help="Specific run id override.")
        review.add_argument("--config", default=None, dest="config_cmd", help="Config path override.")
        if stage.short_name == "quality":
            review.add_argument("--decision", required=True, help="Decision JSON path.")
        elif stage.short_name in {"annotate", "al"}:
            review.add_argument(
                "--file",
                required=False,
                default=None,
                help="Corrected reviews (CSV/Parquet with human_label, or Label Studio JSON for annotate). "
                "If omitted, uses human_review.corrected_file under the stage directory when present.",
            )


def build_parser() -> argparse.ArgumentParser:
    desc = (
        "Main project interface: orchestrate collect → quality → annotate → active learning → train → report.\n"
        "Start data acquisition with `collect discover` (topic + config), then `collect recommend`, "
        "`collect select --ids …`, and `collect run`. Each stage has `status`, `run`, and (where applicable) "
        "`review` subcommands."
    )
    examples = "\n".join(
        [
            "typical flow after `reset`:",
            f"  {cli('collect', 'discover', '--topic', _EXAMPLE_TOPIC_CLI, '--config', 'config.yaml')}",
            f"  {cli('collect', 'recommend')} && {cli('collect', 'select', '--ids', '1,2')} && {cli('collect', 'run', '--config', 'config.yaml')}",
            f"  {cli('quality', 'run')}  # then annotate, al, train, report as needed",
            "",
            "other examples:",
            f"  {cli('status')}",
            f"  {cli('quality', 'review', '--decision', 'artifacts/runs/<run_id>/02_quality/review/quality_review_decision_template.json')}",
            f"  {cli('run', '--config', 'config.yaml')}",
            f"  {cli('resume', '--run-id', '<run_id>')}",
            f"  {cli('artifacts', '--run-id', '<run_id>')}",
            "",
            "Legacy alias (supported, not preferred):",
            f"  {cli('stage', 'quality', 'status', '--run-id', '<run_id>')}",
        ]
    )
    parser = argparse.ArgumentParser(
        description=desc,
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to JSON/YAML config.")
    parser.add_argument("--run-id", default=None, help="Specific run id in artifacts/runs/<run_id>.")
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="Show overall pipeline status.")
    status.add_argument("--run-id", default=None, dest="run_id_cmd", help="Specific run id override.")
    run = sub.add_parser("run", help="Run pipeline until complete / awaiting_review / failed.")
    run.add_argument("--config", default=None, dest="config_cmd", help="Config path override.")
    run.add_argument("--run-id", default=None, dest="run_id_cmd", help="Specific run id override.")
    run.add_argument("--input", default=None, help="Optional input path for quality stage (.csv/.parquet).")
    run.add_argument("--decision", default=None, help="Optional decision path for quality stage.")
    run.add_argument("--task-description", default=None, help="Optional task description for quality strategy note.")
    run.add_argument(
        "--interactive-collect",
        action="store_true",
        dest="interactive_collect",
        help="For COLLECT: same as `collect run --interactive` (terminal topic profile with customer).",
    )
    resume = sub.add_parser("resume", help="Resume latest run from first actionable stage.")
    resume.add_argument("--config", default=None, dest="config_cmd", help="Config path override.")
    resume.add_argument("--run-id", default=None, dest="run_id_cmd", help="Specific run id override.")
    resume.add_argument("--input", default=None, help="Optional input path for quality stage (.csv/.parquet).")
    resume.add_argument("--decision", default=None, help="Optional decision path for quality stage.")
    resume.add_argument("--task-description", default=None, help="Optional task description for quality strategy note.")
    resume.add_argument(
        "--interactive-collect",
        action="store_true",
        dest="interactive_collect",
        help="For COLLECT: terminal topic profile (same as collect run --interactive).",
    )
    reset = sub.add_parser("reset", help="Create a new run and reset state.")
    reset.add_argument("--new-run-id", default=None, help="Explicit run id (optional).")
    artifacts = sub.add_parser("artifacts", help="Show artifact pointers from pipeline state.")
    artifacts.add_argument("--run-id", default=None, dest="run_id_cmd", help="Specific run id override.")

    for stage in STAGE_REGISTRY:
        stage_top = sub.add_parser(
            stage.short_name,
            help=f"{stage.display_name}: status, run" + (", review" if stage.review_supported else "") + (", discover, …" if stage.short_name == "collect" else ""),
            description=_stage_parser_description(stage, legacy=False),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        stage_top_nested = stage_top.add_subparsers(dest="stage_command", required=True)
        register_stage_nested_parsers(stage_top_nested, stage)

    stage_cmd = sub.add_parser(
        "stage",
        help="Backward-compatible alias for `<stage> <action>` (same handlers; prefer top-level stage names).",
        description=(
            "Legacy routing: `stage <collect|quality|…> <status|run|review|…>`.\n"
            f"Example: {cli('stage', 'quality', 'status', '--run-id', '<run_id>')}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    stage_sub = stage_cmd.add_subparsers(dest="stage_name", required=True)
    for stage in STAGE_REGISTRY:
        one = stage_sub.add_parser(
            stage.short_name,
            help=f"{stage.display_name} (legacy namespace).",
            description=_stage_parser_description(stage, legacy=True),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        one_sub = one.add_subparsers(dest="stage_command", required=True)
        register_stage_nested_parsers(one_sub, stage)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manager = PipelineStateManager()
    try:
        if args.command == "status":
            return cmd_status(args, manager)
        if args.command == "reset":
            return cmd_reset(args, manager)
        if args.command == "artifacts":
            return cmd_artifacts(args, manager)
        if args.command == "run":
            return cmd_run(args, manager)
        if args.command == "resume":
            return cmd_resume(args, manager)
        if args.command == "stage" or str(args.command) in STAGE_CLI_TOPLEVEL:
            return dispatch_stage_namespace(args, manager)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2
    except FileNotFoundError as exc:
        print(f"File error: {exc}")
        return 2
    print("Unknown command. Use --help.")
    return 2

