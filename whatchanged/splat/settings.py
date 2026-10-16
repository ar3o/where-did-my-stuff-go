"""Settings of the splat stage (configs/splat.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from whatchanged import config


@dataclass
class TrainConfig:
    iters: int = 30000
    max_gaussians: int = 2_500_000
    grow_grad2d: float = 2e-4
    reset_every: int = 3000
    refine_every: int = 100
    checkpoint_every: int = 5000
    affine_l2: float = 1e-2
    ssim_weight: float = 0.2
    need_mib: int = 5000


@dataclass
class CleanupConfig:
    enabled: bool = False
    min_opacity: float = 0.005
    max_scale_m: float = 1.0
    room_margin_m: float = 0.5
    floor_reflection_m: float = -0.3
    min_views: int = 3
    need_mib: int = 2500


@dataclass
class LocalizeConfig:
    scales: list = field(default_factory=lambda: [0.25, 0.5])
    steps: list = field(default_factory=lambda: [60, 40])
    blur_sigmas: list = field(default_factory=lambda: [2.0, 1.0])
    lr_rotation: float = 2e-3
    lr_translation: float = 4e-3
    need_mib: int = 2500


@dataclass
class SupportConfig:
    min_views: int = 3
    max_angle_deg: float = 60.0
    tol_abs_m: float = 0.02
    tol_rel: float = 0.03
    min_range_m: float = 0.15
    max_range_m: float = 6.0
    need_mib: int = 3000


@dataclass
class SplatSettings:
    train: TrainConfig = field(default_factory=TrainConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    localize: LocalizeConfig = field(default_factory=LocalizeConfig)
    support: SupportConfig = field(default_factory=SupportConfig)


def load_settings(path: str | Path | None = None) -> SplatSettings:
    return config.load(SplatSettings, "splat", path)
