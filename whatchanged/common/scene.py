"""File layout of one scene. Every stage reads and writes through these paths.

<data>/<scene>/
  frames/ frames_hi/ frames.json        reference walk frames (960 px work size, 1920 px for SAM 3 tiles)
  photos/ photos_hi/ photos.json        query photos, EXIF orientation applied
  groups.json names.json                photo groups (one state of the room each) and the object name list
  depth/{frames,photos}/                MoGe-3 depth
  sfm/ poses.json photos_sfm.json       COLMAP model, camera poses in the world frame
  splat/ frame_known/                   Gaussian splat, per-frame known masks (support from the other cameras)
  predicted/<photo>/                    refined pose, predicted view, depth, support / known maps
  features/                             DINOv2 PCA features of frames, photos and predicted views
  inventory/                            SAM 3 detections, labelled 3D objects
  checker/                              training pairs and per-photo change maps
  changes/                              re-identification inputs, one changes.json per group
<runs>/<scene>/                         checker weights, calibrations, films, stills
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from whatchanged import paths
from whatchanged.common.io import read_json


@dataclass(frozen=True)
class Scene:
    name: str

    @property
    def root(self) -> Path:
        return paths.scene_dir(self.name)

    @property
    def runs(self) -> Path:
        return paths.runs_dir(self.name)

    # ingest
    frames = property(lambda s: s.root / "frames")
    frames_hi = property(lambda s: s.root / "frames_hi")
    frames_json = property(lambda s: s.root / "frames.json")
    photos = property(lambda s: s.root / "photos")
    photos_hi = property(lambda s: s.root / "photos_hi")
    photos_json = property(lambda s: s.root / "photos.json")
    groups_json = property(lambda s: s.root / "groups.json")
    names_json = property(lambda s: s.root / "names.json")
    depth = property(lambda s: s.root / "depth")
    # sfm
    sfm = property(lambda s: s.root / "sfm")
    poses_json = property(lambda s: s.root / "poses.json")
    photos_sfm_json = property(lambda s: s.root / "photos_sfm.json")
    sfm_points = property(lambda s: s.root / "sfm_points.npz")
    # splat
    splat_dir = property(lambda s: s.root / "splat")
    splat = property(lambda s: s.root / "splat" / "splat.npz")
    ref_depth = property(lambda s: s.root / "splat" / "ref_depth.npz")
    frame_known = property(lambda s: s.root / "frame_known")
    predicted = property(lambda s: s.root / "predicted")
    # later stages
    features = property(lambda s: s.root / "features")
    inventory = property(lambda s: s.root / "inventory")
    checker = property(lambda s: s.root / "checker")
    changes = property(lambda s: s.root / "changes")

    def frame_image(self, name: str) -> Path:
        return self.frames / f"{name}.jpg"

    def photo_image(self, name: str) -> Path:
        return self.photos / f"{name}.jpg"

    def poses(self) -> dict:
        return read_json(self.poses_json)

    def groups(self) -> dict[int, list[str]]:
        return {int(k): v for k, v in read_json(self.groups_json)["groups"].items()}

    def names(self) -> dict:
        return read_json(self.names_json)

    def phrases(self, with_static: bool = True) -> list[str]:
        """SAM 3 prompts in list order: furniture, items, then static parts of the room."""
        n = self.names()
        return n["furniture"] + n["items"] + (n["static"] if with_static else [])

    def phrase_class(self) -> dict[str, str]:
        n = self.names()
        return {p: c for c in ("furniture", "items", "static") for p in n[c]}

    def walk_split(self) -> dict[str, str]:
        """'val' for the last 20 % of the walk by time, 'train' for the rest (frame name -> split)."""
        import numpy as np

        frames = self.poses()["frames"]
        t80 = float(np.percentile([f["t"] for f in frames], 80))
        return {f["name"]: ("val" if f["t"] >= t80 else "train") for f in frames}
