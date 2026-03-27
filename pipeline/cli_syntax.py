"""Canonical `run_pipeline.py` command-line snippets for status and UX hints."""

from __future__ import annotations

ENTRYPOINT = "python run_pipeline.py"


def cli(*parts: str) -> str:
    """Build `python run_pipeline.py <part> ...` without duplicating the program name."""
    return f"{ENTRYPOINT} {' '.join(parts)}"


def quality_run_input_placeholder() -> str:
    return cli("quality", "run", "--input", "<path>")


def quality_review_decision_placeholder() -> str:
    return cli("quality", "review", "--decision", "<path>")


def annotate_review_file_placeholder() -> str:
    return cli("annotate", "review", "--file", "<path>")


def al_review_file_placeholder() -> str:
    return cli("al", "review", "--file", "<path>")


def stage_run(stage_short: str) -> str:
    return cli(stage_short, "run")


def collect_discover() -> str:
    return cli("collect", "discover")


def pipeline_status() -> str:
    return cli("status")


def pipeline_artifacts() -> str:
    return cli("artifacts")
