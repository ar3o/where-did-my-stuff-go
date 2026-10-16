"""Per-detection features for the photos and the held-out walk frames.

Each SAM 3 detection gets: its 3D box lifted with the aligned depth (photos: MoGe aligned to the predicted depth; walk
frames: the same alignment against the unedited render), its DINOv2 PCA embedding pooled over the mask, its mean colour
mapped into the render's colour space (inverse of the photo's colour match), its z-normalised SAM 3 score, whether it
touches the image border, and a crop for SigLIP 2. Photos also store their before-image detections and the inventory
id map at the photo pose.
Outputs changes/dets/{photo_<name>,val_<frame>}.{json,npz} and the SigLIP 2 crop list.
"""

from __future__ import annotations

import cv2
import numpy as np

from whatchanged.changes.reid import zscore
from whatchanged.common import depth as depthlib
from whatchanged.common import geometry as geo
from whatchanged.common.features import pool
from whatchanged.common.images import colour_match, crop_masked
from whatchanged.common.io import log, pack_masks, read_json, write_json
from whatchanged.common.photoview import PhotoView
from whatchanged.common.scene import Scene
from whatchanged.inventory import detect
from whatchanged.splat import gaussians as gs

BORDER_PX = 12
MIN_DET_PX = 30


def touches_border(m: np.ndarray, W: int, H: int) -> bool:
    ys, xs = np.nonzero(m)
    return bool(xs.min() < BORDER_PX or ys.min() < BORDER_PX or xs.max() >= W - BORDER_PX or ys.max() >= H - BORDER_PX)


def lift_box(m, D, K, T) -> dict | None:
    return geo.box3d(geo.lift(m, D, K, T))


def hires_mask(m: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return cv2.resize(m.astype(np.uint8), (hi.shape[1], hi.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def photo_rows(scene: Scene, tsp, gid, ids, name: str, stats: dict, phrases: set, crops: list) -> bool:
    out = scene.changes / "dets" / f"photo_{name}.json"
    if out.exists():
        return True
    pv = PhotoView(scene, name)
    if not pv.exists:
        return False
    hi = cv2.imread(str(scene.photos_hi / f"{name}.jpg"))
    raw = cv2.imread(str(scene.predicted / name / "rgb_raw.jpg"))[..., ::-1].astype(np.float32) / 255
    ga, gb = colour_match(raw, pv.rgb, pv.known)
    feat = np.load(scene.features / "photos" / f"{name}_real.npy")
    pz = np.load(scene.checker / "photos" / name / "probs.npz")
    P_app = cv2.resize(pz["appeared"].astype(np.float32), (pv.W, pv.H))
    inst = gs.render_ids(tsp, gid, pv.K, pv.T, pv.W, pv.H, ids)
    D = detect.merged(scene.changes / "sam3" / f"photo_{name}", pv.W, pv.H, phrases, MIN_DET_PX)
    Db = detect.merged(scene.changes / "sam3" / f"before_{name}", pv.W, pv.H, phrases, MIN_DET_PX)
    rows = []
    crop_dir = scene.changes / "detcrops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for j, (phrase, s, m) in enumerate(D):
        e = pool(feat, m)
        key = f"det:{name}:{j}"
        if crop_masked(hi, hires_mask(m, hi), crop_dir / f"{name}_{j}.jpg"):
            crops.append(dict(key=key, path=str(crop_dir / f"{name}_{j}.jpg")))
        rows.append(
            dict(
                phrase=phrase,
                score=s,
                z=zscore(stats, phrase, s),
                box=lift_box(m & pv.known, pv.D_photo, pv.K, pv.T),
                emb=None if e is None else e.tolist(),
                col=((pv.rgb[m].mean(0) - gb) / ga).tolist(),
                touches=touches_border(m, pv.W, pv.H),
                p_appeared=float(P_app[m].mean()),
                known=float(pv.known[m].mean()),
                px=int(m.sum()),
                key=key,
            )
        )
    bits, shape = pack_masks([m for _, _, m in D])
    bbits, bshape = pack_masks([m for _, _, m in Db])
    np.savez_compressed(
        out.with_suffix(".npz"), masks=bits, shape=shape, bmasks=bbits, bshape=bshape, inst=inst.astype(np.uint16)
    )
    write_json(
        dict(
            rows=rows,
            before=[dict(phrase=p, score=s, z=zscore(stats, p, s)) for p, s, _ in Db],
            gain=[ga.tolist(), gb.tolist()],
        ),
        out,
    )
    log(f"dets {name}: {len(D)} in the photo, {len(Db)} in the before image")
    return True


def val_rows(scene: Scene, tsp, gid, ids, n: str, stats: dict, phrases: set, crops: list) -> None:
    out = scene.changes / "dets" / f"val_{n}.json"
    if out.exists():
        return
    P = scene.poses()
    K, W, H = np.array(P["K"]), P["W"], P["H"]
    T = np.array(next(f["T_wc"] for f in P["frames"] if f["name"] == n))
    known = np.load(scene.frame_known / f"{n}.npy")
    D_render = gs.render(tsp, K, T, W, H)["depth"]
    D_real, _ = depthlib.align_to_render(
        D_render, depthlib.load_mono(scene.depth / "frames" / f"{n}.npz", (W, H)), known
    )
    feat = np.load(scene.features / "frames" / f"{n}.npy")
    img = cv2.imread(str(scene.frame_image(n)))
    hires = scene.frames_hi / f"{n}.jpg"
    hi = cv2.imread(str(hires)) if hires.exists() else img
    rgb = img[..., ::-1].astype(np.float32) / 255
    inst = gs.render_ids(tsp, gid, K, T, W, H, ids)
    D = detect.merged(scene.changes / "sam3" / f"valreal_{n}", W, H, phrases, MIN_DET_PX, tiles=False)
    rows = []
    crop_dir = scene.changes / "detcrops"
    for j, (phrase, s, m) in enumerate(D):
        e = pool(feat, m)
        key = f"val:{n}:{j}"
        if crop_masked(hi, hires_mask(m, hi), crop_dir / f"val_{n}_{j}.jpg"):
            crops.append(dict(key=key, path=str(crop_dir / f"val_{n}_{j}.jpg")))
        rows.append(
            dict(
                phrase=phrase,
                score=s,
                z=zscore(stats, phrase, s),
                box=lift_box(m & known, D_real, K, T),
                emb=None if e is None else e.tolist(),
                col=rgb[m].mean(0).tolist(),
                px=int(m.sum()),
                key=key,
                touches=touches_border(m, W, H),
            )
        )
    renders = {}
    arrays = {}
    for pd in sorted((scene.checker / "pairs" / "val").glob(f"{n}_*")):
        stem = scene.changes / "sam3" / f"valrender_{pd.name}"
        if not stem.with_suffix(".npz").exists():
            continue
        L = detect.merged(stem, W, H, phrases, MIN_DET_PX, tiles=False)
        renders[pd.name] = [dict(phrase=p, score=s) for p, s, _ in L]
        arrays[f"r_{pd.name}"], arrays[f"rs_{pd.name}"] = pack_masks([m for _, _, m in L])
    bits, shape = pack_masks([m for _, _, m in D])
    np.savez_compressed(out.with_suffix(".npz"), masks=bits, shape=shape, inst=inst.astype(np.uint16), **arrays)
    write_json(dict(rows=rows, renders=renders), out)
    log(f"dets val {n}: {len(D)} detections, {len(renders)} pair renders")


def compute(scene: Scene, tsp, photos: list[str]) -> list[dict]:
    """All detection features; returns the SigLIP 2 crop list (detections and reference crops)."""
    stats = read_json(scene.inventory / "inventory.json")["phrase_score_stats"]
    inv = read_json(scene.inventory / "inventory.json")["objects"]
    gid = np.load(scene.inventory / "inventory.npz")["gid"]
    ids = sorted({0} | {int(o["id"][1:]) for o in inv})
    phrases = set(scene.phrases(with_static=False))
    (scene.changes / "dets").mkdir(parents=True, exist_ok=True)
    crops_path = scene.changes / "det_crops.json"
    crops = read_json(crops_path) if crops_path.exists() else []
    for name in photos:
        photo_rows(scene, tsp, gid, ids, name, stats, phrases, crops)
    split = scene.walk_split()
    for n in sorted(
        x for x, s in split.items() if s == "val" and (scene.changes / "sam3" / f"valreal_{x}.npz").exists()
    ):
        val_rows(scene, tsp, gid, ids, n, stats, phrases, crops)
    seen = set()
    crops = [c for c in crops if not (c["key"] in seen or seen.add(c["key"]))]
    write_json(crops, crops_path)
    return crops + read_json(scene.changes / "references.json")["crops"]
