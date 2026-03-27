"""Canonical CLI entrypoint for this repo: pipeline orchestration and per-stage commands.

Run with ``python run_pipeline.py --help``. Stage namespaces (e.g. ``collect``, ``quality``) are
top-level subcommands; ``stage <name> …`` remains a backward-compatible alias.
"""

from __future__ import annotations

import sys

from pipeline.cli import (
    build_parser,
    cmd_artifacts,
    cmd_collect_discover,
    cmd_collect_recommend,
    cmd_collect_select,
    cmd_reset,
    cmd_resume,
    cmd_run,
    cmd_stage_review,
    cmd_stage_run,
    cmd_stage_status,
    cmd_status,
    dispatch_stage_namespace,
    main,
)
from pipeline.stages.annotate import apply_annotate_review, run_annotate_stage
from pipeline.stages.al import apply_al_review

__all__ = [
    "build_parser",
    "cmd_artifacts",
    "cmd_collect_discover",
    "cmd_collect_recommend",
    "cmd_collect_select",
    "cmd_reset",
    "cmd_resume",
    "cmd_run",
    "cmd_stage_review",
    "cmd_stage_run",
    "cmd_stage_status",
    "cmd_status",
    "dispatch_stage_namespace",
    "main",
    "apply_al_review",
    "apply_annotate_review",
    "run_annotate_stage",
]


if __name__ == "__main__":
    sys.exit(main())
