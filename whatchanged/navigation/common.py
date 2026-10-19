"""Shared paths, settings and small helpers of the navigation module (numpy / stdlib only, so every environment can
import it).

Frames:
  splat frame S  the memory splat's frame (poses.json T_wc, Gaussians, inventory boxes);
  floor frame F  S rotated and translated so that the RANSAC floor plane is z = 0 with +z up (memory_map.py writes
                 T_F_S).
Every navigation output (grid, poses, paths, boxes) is in F unless a key says otherwise. Camera poses T_wc are OpenCV
camera-to-world.

Files: <data>/<scene>/navigation/ holds the memory map, run logs and caches; <runs>/navigation/ holds films, stills and
result tables; <runs>/<scene>/navigation/ holds the scene's no-change baseline and fingerprint match threshold.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np

from whatchanged import paths
from whatchanged.common.geometry import apply  # noqa: F401  (re-exported for the navigation modules)
from whatchanged.common.io import log, read_json
from whatchanged.common.scene import Scene

CFG_PATH = paths.CONFIGS / "navigation.json"
RUNS = paths.RUNS / "navigation"
_OVERRIDES: dict[tuple[str, str], object] = {}


def cfg() -> dict:
    """Navigation settings (configs/navigation.json), with any overrides set by override()."""
    c = read_json(CFG_PATH)
    for (section, key), value in _OVERRIDES.items():
        c[section][key] = copy.deepcopy(value)
    return c


def override(section: str, key: str, value) -> None:
    """Replace one setting for the rest of this process (used for short debugging runs)."""
    _OVERRIDES[(section, key)] = value


def nav_dir(scene: str) -> Path:
    return paths.scene_dir(scene) / "navigation"


def scene_runs(scene: str) -> Path:
    return paths.runs_dir(scene) / "navigation"


def frame_file(root: Path, name: str) -> Path:
    """Image of a walk frame under a scene root; frame names are stored without the extension."""
    return root / "frames" / (name if Path(name).suffix else f"{name}.jpg")


# ------------------------------------------------------------------ geometry
def look_at(eye, target, up=(0, 0, 1.0)) -> np.ndarray:
    """OpenCV camera-to-world looking from eye to target (x = forward x up, y = forward x x)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


# ------------------------------------------------------------------ memories
def memory_paths(scene: str) -> dict:
    """The robot's memory of a scene: splat, per-Gaussian inventory ids, inventory json and navigation map folder."""
    s = Scene(scene)
    return dict(
        name="base",
        splat_dir=s.splat_dir,
        splat=s.splat,
        gid=s.inventory / "inventory.npz",
        inv=s.inventory / "inventory.json",
        map_dir=nav_dir(scene),
    )


def doorway_start(frames: list[dict], M: dict, min_clear: float, min_cam_h: float = 1.2):
    """The robot's start: the first frame (time order) of a walk where a standing camera (more than min_cam_h above the
    floor) is over memory-map floor with clearance >= min_clear (the smallest footprint of the shrink ladder).
    Returns (frame, T_wc in F) or (None, None)."""
    from whatchanged.navigation.planner import to_cell

    T_F_S = M["T_F_S"]
    for f in sorted(frames, key=lambda f: f["t"]):
        T = T_F_S @ np.array(f["T_wc"])
        i, j = to_cell(T[:2, 3], M["origin"], float(M["cell"]), M["shape"])
        if T[2, 3] - M["floor_level"][i, j] > min_cam_h and M["free"][i, j] and M["dist"][i, j] >= min_clear:
            return f, T
    return None, None


def _runs_path(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else paths.RUNS / p


def scene_spec(scene: str) -> dict:
    """Model and calibration files of a scene: its own by default, replaced by configs/navigation.json "scenes"."""
    s = Scene(scene)
    spec = dict(
        checker_ckpt=s.runs / "checker" / "best.pt",
        feats_pca=s.features / "pca.npz",
        calib=s.runs / "checker" / "conf_calibration.json",
        nochange=scene_runs(scene) / "nochange_tau.json",
        move_match=scene_runs(scene) / "move_match.json",
    )
    extra = cfg().get("scenes", {}).get(scene, {})
    for k, v in extra.items():
        if k in spec:
            spec[k] = _runs_path(v)
    if "move_match_from" in extra:
        spec["move_match"] = scene_runs(extra["move_match_from"]) / "move_match.json"
    for k in ("checker_ckpt", "feats_pca", "calib"):
        if not Path(spec[k]).exists():
            log(f"WARNING: {scene}: {k} file {spec[k]} does not exist")
    return spec


def file_sha(path, n: int = 12) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def inventory_sha(scene: str) -> str:
    """Identity of the current inventory (json + per-Gaussian ids). Every product keyed by object id (map, fingerprint
    references, visibility tables) carries it, because ids are re-numbered when the inventory is rebuilt."""
    m = memory_paths(scene)
    return file_sha(m["inv"]) + file_sha(m["gid"])
