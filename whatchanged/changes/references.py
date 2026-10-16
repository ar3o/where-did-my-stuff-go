"""Inputs for re-identification and instance counting.

* References of every movable inventory object from the training part of the walk only: its DINOv2 fingerprint and
  mean colour (inventory/fingerprint.py rule) and crops of its 4 best views (visible px / depth x sqrt(sharpness))
  from the 1920 px frames.
* The before image of every photo: the reference splat rendered at the photo's pose at 2x resolution and
  colour-matched to the photo. A detection that the same phrase also finds in the before image at the same place
  existed before.
* SAM 3 jobs, furniture and item phrases only (static parts of the room are never objects), min score 0.2, top 8 per
  phrase: photos and their before images (full image + 2 x 2 tiles), and the held-out walk frames (the real frame and
  every validation pair render) for the calibration.
"""

from __future__ import annotations

import cv2
import numpy as np

from whatchanged.common import geometry as geo
from whatchanged.common.images import colour_match, crop_masked
from whatchanged.common.io import log, read_json, write_json
from whatchanged.common.scene import Scene
from whatchanged.inventory import detect
from whatchanged.inventory.fingerprint import fingerprints
from whatchanged.splat import gaussians as gs

MIN_VIEW_PX = 200
BEST_VIEWS = 4
BEFORE_SCALE = 2


def movable_objects(scene: Scene) -> tuple[list[dict], np.ndarray]:
    objs = [o for o in read_json(scene.inventory / "inventory.json")["objects"] if o["cls"] != "static"]
    return objs, np.load(scene.inventory / "inventory.npz")["gid"]


def reference_crops(scene: Scene, tsp: dict, objs: list[dict], gid: np.ndarray) -> list[dict]:
    P = scene.poses()
    K, W, H = np.array(P["K"]), P["W"], P["H"]
    split = scene.walk_split()
    ids = [0] + [int(o["id"][1:]) for o in objs]
    views = {o["id"]: [] for o in objs}
    for f in P["frames"]:
        n = f["name"]
        if split[n] != "train":
            continue
        T = np.array(f["T_wc"])
        inst = gs.render_ids(tsp, gid, K, T, W, H, ids)
        D = gs.render(tsp, K, T, W, H)["depth"]
        cnt = np.bincount(inst.ravel().astype(np.int64), minlength=max(ids) + 1)
        for o in objs:
            k = int(o["id"][1:])
            if cnt[k] < MIN_VIEW_PX:
                continue
            m = inst == k
            dep = float(np.median(D[m & (D > 0)])) if (m & (D > 0)).any() else 3.0
            views[o["id"]].append((cnt[k] / max(dep, 0.3) * np.sqrt(f.get("sharpness", 500.0)), n))
    want = {}
    for o in objs:
        for r, (_, n) in enumerate(sorted(views[o["id"]], reverse=True)[:BEST_VIEWS]):
            want.setdefault(n, []).append((o["id"], r))
    T_of = {f["name"]: np.array(f["T_wc"]) for f in P["frames"]}
    out_dir = scene.changes / "refcrops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops = []
    for n, rows in want.items():
        hires = scene.frames_hi / f"{n}.jpg"
        img = cv2.imread(str(hires if hires.exists() else scene.frame_image(n)))
        s = img.shape[1] / W
        inst = gs.render_ids(tsp, gid, geo.scale_K(K, s), T_of[n], img.shape[1], img.shape[0], ids)
        for oid, r in rows:
            out = out_dir / f"{oid}_{r}.jpg"
            if crop_masked(img, inst == int(oid[1:]), out):
                crops.append(dict(key=f"ref:{oid}:{r}", path=str(out), object=oid))
    return crops


def before_renders(scene: Scene, tsp: dict, photos: list[str]) -> None:
    out_dir = scene.changes / "before"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in photos:
        out = out_dir / f"{name}.jpg"
        pdir = scene.predicted / name
        if out.exists() or not (pdir / "pose.json").exists():
            continue
        pose = read_json(pdir / "pose.json")
        w, h = pose["size"]
        raw = cv2.imread(str(pdir / "rgb_raw.jpg"))[..., ::-1].astype(np.float32) / 255
        real = cv2.imread(str(scene.photo_image(name)))[..., ::-1].astype(np.float32) / 255
        a, b = colour_match(raw, real, np.load(pdir / "known.npy"))
        r = gs.render(
            tsp,
            geo.scale_K(np.array(pose["K"]), BEFORE_SCALE),
            np.array(pose["T_wc"]),
            BEFORE_SCALE * w,
            BEFORE_SCALE * h,
            depth=False,
        )
        rgb = np.clip(r["rgb"] / np.maximum(r["alpha"], 1e-3)[..., None] * a + b, 0, 1)
        rgb[r["alpha"] < 0.05] = 0
        cv2.imwrite(str(out), (rgb[..., ::-1] * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])


def sam3_items(scene: Scene, photos: list[str]) -> list[dict]:
    items = []
    tiles = scene.changes / "tiles"
    for name in photos:
        pdir = scene.predicted / name
        if not (pdir / "pose.json").exists():
            continue
        w, h = read_json(pdir / "pose.json")["size"]
        for kind, src in (
            ("photo", scene.photos_hi / f"{name}.jpg"),
            ("before", scene.changes / "before" / f"{name}.jpg"),
        ):
            items += detect.jobs_for_image(src, src, scene.changes / "sam3" / f"{kind}_{name}", tiles, w, h)
    P = scene.poses()
    split = scene.walk_split()
    pairs = scene.checker / "pairs" / "val"
    for n in sorted(x for x, s in split.items() if s == "val" and (pairs / f"{x}_0").exists()):
        items.append(
            dict(
                image=str(scene.frame_image(n)),
                out=str(scene.changes / "sam3" / f"valreal_{n}.npz"),
                mask_wh=[P["W"], P["H"]],
            )
        )
        for pd in sorted(pairs.glob(f"{n}_*")):
            items.append(
                dict(
                    image=str(pd / "render.jpg"),
                    out=str(scene.changes / "sam3" / f"valrender_{pd.name}.npz"),
                    mask_wh=[P["W"], P["H"]],
                )
            )
    return items


def prepare(scene: Scene, tsp: dict, photos: list[str]) -> list[dict]:
    """Writes changes/references.json and the before renders; returns the SAM 3 job items."""
    objs, gid = movable_objects(scene)
    refs = fingerprints(scene, tsp, objs, gid)
    crops = reference_crops(scene, tsp, objs, gid)
    ref = {oid: dict(emb=r["emb"].tolist(), col=r["col"].tolist()) for oid, r in refs.items()}
    write_json(dict(ref=ref, crops=crops, rule="training part of the walk only"), scene.changes / "references.json")
    log(f"references: {len(ref)} fingerprints, {len(crops)} crops")
    before_renders(scene, tsp, photos)
    return sam3_items(scene, photos)
