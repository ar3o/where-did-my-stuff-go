"""Appearance fingerprints of inventory objects from the training part of the walk (the last 20 % is held out for the
calibrations): the DINOv2 PCA embedding pooled over the object's rendered mask on known pixels in every training frame
where at least 800 px are visible (weighted by the visible area), the mean colour of those pixels, and the box size."""

from __future__ import annotations

import cv2
import numpy as np

from whatchanged.common.features import pool
from whatchanged.common.scene import Scene
from whatchanged.splat import gaussians as gs

MIN_PX = 800


def fingerprints(scene: Scene, tsp: dict, objects: list[dict], gid: np.ndarray) -> dict[str, dict]:
    P = scene.poses()
    K, W, H = np.array(P["K"]), P["W"], P["H"]
    split = scene.walk_split()
    movable = {int(o["id"][1:]): o for o in objects if o["cls"] != "static"}
    ids = [0] + sorted(int(o["id"][1:]) for o in objects)
    acc = {k: [] for k in movable}
    cols = {k: [] for k in movable}
    for f in P["frames"]:
        n = f["name"]
        feat_path = scene.features / "frames" / f"{n}.npy"
        if split[n] != "train" or not feat_path.exists():
            continue
        inst = gs.render_ids(tsp, gid, K, np.array(f["T_wc"]), W, H, ids)
        feat = np.load(feat_path)
        img = cv2.imread(str(scene.frame_image(n)))[..., ::-1].astype(np.float32) / 255
        known = np.load(scene.frame_known / f"{n}.npy")
        for k in np.unique(inst):
            if int(k) not in movable:
                continue
            m = (inst == k) & known
            if m.sum() < MIN_PX:
                continue
            e = pool(feat, m)
            if e is not None:
                acc[int(k)].append(e * m.sum())
                cols[int(k)].append(img[m].mean(0))
    refs = {}
    for k, L in acc.items():
        if L:
            e = np.sum(L, 0)
            refs[f"I{k}"] = dict(
                emb=e / np.linalg.norm(e), col=np.mean(cols[k], 0), size=np.array(movable[k]["box"]["size"])
            )
    return refs
