from __future__ import annotations

from pathlib import Path
from typing import Optional

from pipeline.artifacts import build_quality_config


def build_quality_agent(config_path: Optional[str], run_dir: Path):
    from agents.data_quality_agent import DataQualityAgent

    cfg = build_quality_config(config_path, run_dir)
    return DataQualityAgent(config=cfg)

