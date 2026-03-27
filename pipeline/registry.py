from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class StageDefinition:
    """Static metadata for one pipeline stage."""

    stage_id: str
    short_name: str
    display_name: str
    ordinal: int
    review_supported: bool
    artifact_dir_name: str


STAGE_REGISTRY: List[StageDefinition] = [
    StageDefinition("01_collect", "collect", "COLLECT", 1, False, "01_collect"),
    StageDefinition("02_quality", "quality", "QUALITY", 2, True, "02_quality"),
    StageDefinition("03_annotate", "annotate", "ANNOTATE", 3, True, "03_annotate"),
    StageDefinition("04_al", "al", "AL", 4, True, "04_al"),
    StageDefinition("05_train", "train", "TRAIN", 5, False, "05_train"),
    StageDefinition("06_report", "report", "REPORT", 6, False, "06_report"),
]

STAGE_BY_SHORT_NAME: Dict[str, StageDefinition] = {stage.short_name: stage for stage in STAGE_REGISTRY}
STAGE_BY_ID: Dict[str, StageDefinition] = {stage.stage_id: stage for stage in STAGE_REGISTRY}

