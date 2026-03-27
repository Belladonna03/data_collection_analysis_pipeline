"""INTERNAL DEBUG / DEVELOPMENT REPL — not the primary user interface.

This module runs an interactive slash-command session (``/discover``, ``/plans``, …)
against :class:`~agents.data_collection_agent.DataCollectionAgent` only.

**For coursework, demos, and reproducible runs, use the canonical CLI**::

    python run_pipeline.py --help
    python run_pipeline.py collect discover ...

Do not treat this entrypoint as part of the supported end-to-end pipeline UX.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from agents.data_collection.schemas import CollectionPlan, TopicProfile
from agents.data_collection_agent import DataCollectionAgent


def _load_dotenv_files(config_path: str | None) -> None:
    """Load `.env` into the process so Kaggle/HF/GitHub env vars apply to the CLI."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if config_path:
        config_dir = Path(config_path).expanduser().resolve().parent
        load_dotenv(config_dir / ".env", override=False)
    load_dotenv(Path.cwd() / ".env", override=False)


HELP_TEXT = """
Available commands:
  /help       Show this help message
  /status     Show current session status and topic profile
  /discover   Discover source candidates for the current topic profile
  /plans      Build and show collection plans
  /select N   Select plan number N for execution
  /run        Execute the selected plan or best available plan
  /artifacts  Show saved artifact paths from the last run
  /reset      Reset the current session
  /exit       Exit the CLI
""".strip()


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "[INTERNAL] Debug REPL for DataCollectionAgent (slash commands). "
            "Not the main pipeline interface — use `python run_pipeline.py` for orchestration."
        ),
        epilog=(
            "Canonical workflow: python run_pipeline.py collect discover --topic \"…\" --config config.yaml "
            "→ collect recommend → collect select --ids … → collect run."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON or YAML config file.",
    )
    return parser


def summarize_topic_profile(topic_profile: TopicProfile) -> str:
    """Return a short topic profile summary."""

    fields = [
        ("topic", topic_profile.topic),
        ("modality", topic_profile.modality),
        ("language", topic_profile.language),
        ("task_type", topic_profile.task_type),
        ("size_target", topic_profile.size_target),
        ("needs_labels", topic_profile.needs_labels),
    ]
    parts = [f"{name}={value}" for name, value in fields if value is not None]
    return ", ".join(parts) if parts else "empty"


def print_status(agent: DataCollectionAgent) -> None:
    """Print current session status."""

    print(f"status: {agent.session.status.value}")
    print(f"topic_profile: {summarize_topic_profile(agent.session.topic_profile)}")
    print(f"candidates: {len(agent.session.candidates)}")
    print(f"plans: {len(agent.session.proposed_plans)}")
    print(f"selected_plan: {'yes' if agent.session.selected_plan else 'no'}")


def print_candidates(candidates: list[Any]) -> None:
    """Print discovered source candidates."""

    if not candidates:
        print("No candidates found.")
        return

    print("Candidates:")
    for index, candidate in enumerate(candidates, start=1):
        location = candidate.url or candidate.endpoint or "-"
        score = candidate.relevance_score if candidate.relevance_score is not None else "-"
        demo_suffix = " [DEMO]" if getattr(candidate, "is_demo_fallback", False) else ""
        execution_suffix = ""
        if not getattr(candidate, "is_executable", True):
            reason = getattr(candidate, "non_executable_reason", None)
            execution_suffix = f" | non-executable: {reason or 'no connector yet'}"
        print(
            f"  {index}. [{candidate.source_type.value}] {candidate.name}{demo_suffix} "
            f"| score={score} | {location}{execution_suffix}"
        )


def print_plans(plans: list[CollectionPlan], selected_plan: CollectionPlan | None) -> None:
    """Print proposed plans."""

    if not plans:
        print("No plans available.")
        return

    print("Plans:")
    for index, plan in enumerate(plans, start=1):
        marker = "*" if selected_plan is plan else " "
        source_names = ", ".join(
            f"{source.name} [{'exec' if source.is_executable else 'discovery-only'}]"
            for source in plan.sources
        ) or "(no sources)"
        warnings = "; ".join(plan.warnings) if plan.warnings else "none"
        print(f"{marker} {index}. sources: {source_names}")
        print(f"     rationale: {plan.rationale}")
        print(f"     warnings: {warnings}")


def print_result(agent: DataCollectionAgent, result: Any) -> None:
    """Print execution result summary."""

    print(f"run status: {agent.session.status.value}")
    if agent.session.selected_plan and agent.session.selected_plan.warnings:
        print("Plan warnings:")
        for warning in agent.session.selected_plan.warnings:
            print(f"  - {warning}")
    if result.dataframe is not None:
        print(f"merged rows: {len(result.dataframe)}")
        print(f"merged columns: {', '.join(result.dataframe.columns)}")

    if result.per_source_stats:
        print("Per-source stats:")
        for source_id, stats in result.per_source_stats.items():
            print(f"  - {source_id}: {stats['rows']} rows")

    if result.validation_report.warnings:
        print("Validation warnings:")
        for warning in result.validation_report.warnings:
            print(f"  - {warning}")

    print_artifacts(agent)


def print_artifacts(agent: DataCollectionAgent) -> None:
    """Print stored artifact paths."""

    if not agent.session.artifacts:
        print("No artifacts saved yet.")
        return

    print("Artifacts:")
    for name, path in sorted(agent.session.artifacts.items()):
        print(f"  - {name}: {path}")


def build_plans(agent: DataCollectionAgent) -> list[CollectionPlan]:
    """Build and store plans for the current session."""

    if not agent.session.candidates:
        candidates = agent.discover_sources(agent.session.topic_profile)
    else:
        candidates = agent.session.candidates

    plans = agent.planner.build_plans(agent.session.topic_profile, candidates)
    agent.session.proposed_plans = plans
    return plans


def handle_chat_message(agent: DataCollectionAgent, user_message: str) -> None:
    """Handle a plain user message."""

    response = agent.chat_step(user_message)
    if response.get("ready_for_discovery"):
        print(response["message"])
        print("Tip: use /discover, /plans, or /run next.")
        return

    print(response["next_question"])


def handle_command(agent: DataCollectionAgent, command_line: str) -> bool:
    """Handle a slash command. Return False to exit."""

    parts = command_line[1:].split()
    command = parts[0].lower() if parts else ""

    if command == "help":
        print(HELP_TEXT)
        return True
    if command == "status":
        print_status(agent)
        return True
    if command == "discover":
        candidates = agent.discover_sources(agent.session.topic_profile)
        journal = getattr(agent.discovery_service, "last_journal", None)
        if journal is not None:
            print("Provider capabilities:")
            for capability in journal.provider_capabilities:
                status = "ok" if capability.available else "unavailable"
                reason = f" ({capability.reason})" if capability.reason else ""
                print(f"  - {capability.provider.value}: {status}{reason}")
            print(f"queries: {journal.queries}")
            if journal.used_demo_fallback:
                print("demo_fallback: enabled and used")
        print_candidates(candidates)
        return True
    if command == "plans":
        plans = build_plans(agent)
        print_plans(plans, agent.session.selected_plan)
        return True
    if command == "select":
        if len(parts) != 2 or not parts[1].isdigit():
            print("Usage: /select N")
            return True
        plans = agent.session.proposed_plans or build_plans(agent)
        plan_index = int(parts[1]) - 1
        if plan_index < 0 or plan_index >= len(plans):
            print(f"Invalid plan number: {parts[1]}")
            return True
        agent.session.selected_plan = plans[plan_index]
        print(f"Selected plan {parts[1]}.")
        print_plans(plans, agent.session.selected_plan)
        return True
    if command == "run":
        try:
            result = agent.interactive_run()
        except Exception as exc:
            print(f"Run failed: {exc}")
            return True
        print_result(agent, result)
        return True
    if command == "artifacts":
        print_artifacts(agent)
        return True
    if command == "reset":
        agent.reset_session()
        print("Session reset.")
        print_status(agent)
        return True
    if command == "exit":
        print("Bye.")
        return False

    print(f"Unknown command: /{command}")
    print("Use /help to see available commands.")
    return True


def main() -> int:
    """Run the terminal REPL."""

    args = build_parser().parse_args()
    _load_dotenv_files(args.config)
    agent = DataCollectionAgent(config=args.config)

    print("DataCollectionAgent CLI")
    print("Type a message to clarify the topic, or use /help.")
    print_status(agent)

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\nBye.")
            return 0

        if not line:
            continue

        if line.startswith("/"):
            should_continue = handle_command(agent, line)
            if not should_continue:
                return 0
            continue

        handle_chat_message(agent, line)


if __name__ == "__main__":
    sys.exit(main())
