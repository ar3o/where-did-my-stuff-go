"""Episode worlds: what the robot's simulated head camera renders.

A real episode is another scene of the same room (a later walk, its own splat and walk frames) registered into the
memory scene's splat frame: its poses_memory.json, or else its poses.json, must hold camera poses in the memory frame,
and its splat and reference-depth cache must be in that frame too. Its own walk gives the robot's doorway start and the
real frame shown next to the render. 'memory' is the memory itself (the no-change control). A prepared world folder
<data>/<memory>/navigation/worlds/<world>/ with splat.npz (and optionally edit.json, read only by the evaluation) is
used as it is.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from whatchanged import paths
from whatchanged.common.io import read_json
from whatchanged.common.scene import Scene
from whatchanged.navigation.common import cfg, doorway_start, frame_file, memory_paths, nav_dir


def world_dir(scene: str, world: str) -> Path:
    return nav_dir(scene) / "worlds" / world


def resolve_world(scene: str, world: str) -> tuple[Path, dict | None]:
    """(world splat, start pose or None) of an episode world seen from the memory scene `scene`. A start of None means
    the robot starts from the memory walk's doorway pose."""
    from whatchanged.navigation.memory_map import load_map

    if world == "memory":
        return memory_paths(scene)["splat"], None
    wd = world_dir(scene, world)
    if (wd / "splat.npz").exists():
        return wd / "splat.npz", None
    ep = Scene(world)
    pj = ep.root / "poses_memory.json" if (ep.root / "poses_memory.json").exists() else ep.poses_json
    Pw = read_json(pj)
    M, _ = load_map(scene)
    f0, T0 = doorway_start(Pw["frames"], M, min(cfg()["robots"]["shrink_ladder_m"]))
    if f0 is None:
        raise RuntimeError(f"{world}: no frame of its walk stands on memory floor")
    start = dict(
        xy=T0[:2, 3].tolist(),
        yaw=float(math.atan2(T0[1, 2], T0[0, 2])),
        source=f"{world} video frame {f0['name']} (t {f0['t']} s)",
        real_image=str(frame_file(ep.root, f0["name"])),
        T_S=f0["T_wc"],
        poses=str(pj),
    )
    return ep.splat, start


def episode_root(start: dict) -> Path:
    """Scene root of the episode whose poses file a start pose names."""
    return Path(start["poses"]).parent if start.get("poses") else paths.DATA


def nearest_real(start: dict | None, T_S: np.ndarray):
    """The episode walk's own frame whose camera is closest to the pose T_S (position + 0.5 m per radian of viewing
    direction difference), so a verdict can be checked against a real photo; None for worlds without a walk.
    Returns (rgb, frame name, distance m, angle deg)."""
    import cv2

    if not start or not start.get("poses"):
        return None
    Pw = read_json(start["poses"])
    best, bd = None, 1e9
    for f in Pw["frames"]:
        T = np.array(f["T_wc"], float)
        ang = math.acos(float(np.clip(T[:3, 2] @ T_S[:3, 2], -1, 1)))
        d = float(np.linalg.norm(T[:3, 3] - T_S[:3, 3])) + 0.5 * ang
        if d < bd:
            best, bd = f, d
    T = np.array(best["T_wc"], float)
    img = cv2.imread(str(frame_file(episode_root(start), best["name"])))
    if img is None:
        return None
    ang = math.degrees(math.acos(float(np.clip(T[:3, 2] @ T_S[:3, 2], -1, 1))))
    return img[..., ::-1], best["name"], float(np.linalg.norm(T[:3, 3] - T_S[:3, 3])), ang
