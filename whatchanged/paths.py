"""Filesystem layout shared by every stage.

Inputs and outputs live under three roots, each overridable with an environment variable:

    data/     WHATCHANGED_DATA     one folder per scene: frames, photos, poses, splat, inventory, checker pairs, changes
    runs/     WHATCHANGED_RUNS     trained checkers, calibrations, films, logs
    weights/  WHATCHANGED_WEIGHTS  third-party weights (SAM 3, SigLIP 2, DINOv2 hub, MoGe-3)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("WHATCHANGED_DATA", REPO / "data"))
RUNS = Path(os.environ.get("WHATCHANGED_RUNS", REPO / "runs"))
WEIGHTS = Path(os.environ.get("WHATCHANGED_WEIGHTS", REPO / "weights"))
CONFIGS = REPO / "configs"


def scene_dir(scene: str) -> Path:
    return DATA / scene


def runs_dir(scene: str) -> Path:
    return RUNS / scene
