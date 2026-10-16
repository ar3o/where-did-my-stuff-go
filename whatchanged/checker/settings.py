"""Settings of the checker stage (configs/checker.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from whatchanged import config


@dataclass
class PairsConfig:
    per_frame: int = 4
    need_mib: int = 2500


@dataclass
class FeaturesConfig:
    need_mib: int = 3000


@dataclass
class TrainConfig:
    epochs: int = 40
    batch: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_minutes: float = 25.0
    need_mib: int = 4000


@dataclass
class InferConfig:
    need_mib: int = 2500


@dataclass
class CalibrateConfig:
    need_mib: int = 2500


@dataclass
class CheckerSettings:
    pairs: PairsConfig = field(default_factory=PairsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    calibrate: CalibrateConfig = field(default_factory=CalibrateConfig)


def load_settings(path: str | Path | None = None) -> CheckerSettings:
    return config.load(CheckerSettings, "checker", path)
