"""Geometry first, then names: merge fragments of one physical object, then name each object from two independent
sources.

1. Merge: two objects merge when their Gaussians touch (< 1.5 cm), they look alike (mean colours within 0.12 and, when
   both have a fingerprint, DINOv2 cosine >= 0.75) and the merged box stays object-sized (<= 0.5 m). Objects larger
   than 0.7 m (furniture, room structure) are never merged. A looser "colour or DINOv2" rule merged a milk jug with a
   fruit bowl and a shoe box. Two faces of one box are adjacent and alike, so they join before any naming.
2. SAM 3 votes per merged object as in voting.py, turned into shares p_sam (with 1e-3 smoothing over all phrases).
3. SigLIP 2 zero-shot on the object's 4 best views (visible pixels / depth x sqrt(frame sharpness), so close, clear
   views decide), cropped from the 1920 px frames with the outside dimmed; views averaged with those weights -> p_sig.
4. p = normalise(p_sam x p_sig). Name = argmax with its probability as the name confidence; the runner-up is kept and
   "(uncertain)" is added when (p1 - p2) / p1 < 0.25.
"""

from __future__ import annotations

import cv2
import numpy as np

from whatchanged.common import geometry as geo
from whatchanged.common.images import crop_masked, ochiai
from whatchanged.common.scene import Scene
from whatchanged.splat import gaussians as gs

MERGE_TOUCH_M = 0.015
MERGE_COLOUR = 0.12
MERGE_DINO_COS = 0.75
MERGE_MAX_SIZE_M = 0.5
NO_MERGE_ABOVE_M = 0.7
MIN_VIEW_PX = 200
BEST_VIEWS = 4
SAM_SMOOTHING = 1e-3


def num(oid: str) -> int:
    return int(oid[1:])


def geometric_merge(objs: list[dict], gid: np.ndarray, sp: dict, emb: dict) -> tuple[dict, list, np.ndarray]:
    """Returns (surviving objects by id, merges [(dropped, kept)], merged gid)."""
    from scipy.spatial import cKDTree

    gid = gid.copy()
    means, rgb = sp["means"], gs.rgb_of(sp)
    ext = {o["id"]: max(o["box"]["size"]) for o in objs}
    alive = {o["id"]: dict(o) for o in objs}
    merges = []
    changed = True
    while changed:
        changed = False
        ids = sorted(alive, key=num)
        for ai, a in enumerate(ids):
            if ext[a] > NO_MERGE_ABOVE_M:
                continue
            for b in ids[ai + 1 :]:
                if ext[b] > NO_MERGE_ABOVE_M:
                    continue
                Pa, Pb = means[gid == num(a)], means[gid == num(b)]
                if len(Pa) == 0 or len(Pb) == 0:
                    continue
                ca, cb = rgb[gid == num(a)].mean(0), rgb[gid == num(b)].mean(0)
                alike = np.abs(ca - cb).max() < MERGE_COLOUR and (
                    a not in emb or b not in emb or float(emb[a] @ emb[b]) >= MERGE_DINO_COS
                )
                if not alike:
                    continue
                if len(Pa) > len(Pb):
                    Pa, Pb = Pb, Pa
                sub = Pa[np.random.default_rng(0).choice(len(Pa), min(len(Pa), 3000), replace=False)]
                if cKDTree(Pb).query(sub, k=1)[0].min() >= MERGE_TOUCH_M:
                    continue
                Q = np.concatenate([means[gid == num(a)], means[gid == num(b)]])
                cc, sz, yaw = geo.obb_zup(Q, 2, 98)
                if max(sz) > MERGE_MAX_SIZE_M:
                    continue
                keep, drop = (a, b) if (gid == num(a)).sum() >= (gid == num(b)).sum() else (b, a)
                gid[gid == num(drop)] = num(keep)
                alive[keep]["box"] = dict(
                    centre=cc.round(4).tolist(), size=sz.round(4).tolist(), yaw=round(float(yaw), 4)
                )
                ext[keep] = float(max(sz))
                alive[keep].setdefault("merged_from", []).append(drop)
                alive.pop(drop)
                merges.append((drop, keep))
                changed = True
                break
            if changed:
                break
    return alive, merges, gid


def votes_and_views(scene: Scene, tsp: dict, gid: np.ndarray, keep_ids: list[str], dets_of: dict, stats: dict):
    """SAM 3 votes and candidate views (weight, frame, visible px) of every object over the detection frames."""
    P = scene.poses()
    K, W, H = np.array(P["K"]), P["W"], P["H"]
    T_of = {f["name"]: np.array(f["T_wc"]) for f in P["frames"]}
    sharp = {f["name"]: f.get("sharpness", 500.0) for f in P["frames"]}
    ids = [0] + [num(i) for i in keep_ids]
    votes = {i: {} for i in keep_ids}
    views = {i: [] for i in keep_ids}
    counts = {i: [] for i in keep_ids}
    for name, d in dets_of.items():
        T = T_of[name]
        inst = gs.render_ids(tsp, gid, K, T, W, H, ids)
        D = gs.render(tsp, K, T, W, H)["depth"]
        cnt = np.bincount(inst.ravel().astype(np.int64), minlength=max(ids) + 1)
        for i in keep_ids:
            c = int(cnt[num(i)])
            if c >= MIN_VIEW_PX:
                m = inst == num(i)
                dep = float(np.median(D[m & (D > 0)])) if (m & (D > 0)).any() else 3.0
                views[i].append((c / max(dep, 0.3) * np.sqrt(sharp.get(name, 500.0)), name, c))
                counts[i].append(c)
        for phrase, score, mk in d:
            labs, inter = np.unique(inst[mk], return_counts=True)
            for lab, n_in in zip(labs, inter):
                if lab == 0 or f"I{lab}" not in votes:
                    continue
                och = ochiai(n_in, mk.sum(), cnt[lab])
                if och < 0.05:
                    continue
                mu, sd = stats.get(phrase, (0.5, 0.2))
                z = (float(score) - mu) / sd
                votes[f"I{lab}"].setdefault(name, []).append((phrase, och / (1 + np.exp(-z)), int(cnt[lab])))
    total = {}
    for i, per in votes.items():
        top = max(counts[i]) if counts[i] else 1
        vv = {}
        for rows in per.values():
            for p, w, c in rows:
                vv[p] = vv.get(p, 0.0) + w * c / top
        total[i] = vv
    return total, views


def best_view_crops(scene: Scene, tsp: dict, gid: np.ndarray, keep_ids: list[str], views: dict, crop_dir) -> list[dict]:
    P = scene.poses()
    K, W, H = np.array(P["K"]), P["W"], P["H"]
    T_of = {f["name"]: np.array(f["T_wc"]) for f in P["frames"]}
    ids = [0] + [num(i) for i in keep_ids]
    crop_dir.mkdir(parents=True, exist_ok=True)
    crops = []
    for i in keep_ids:
        for r, (w, name, _) in enumerate(sorted(views[i], reverse=True)[:BEST_VIEWS]):
            T = T_of[name]
            hires = scene.frames_hi / f"{name}.jpg"
            if hires.exists():
                img = cv2.imread(str(hires))
                s = img.shape[1] / W
                inst = gs.render_ids(tsp, gid, geo.scale_K(K, s), T, img.shape[1], img.shape[0], ids)
            else:
                img = cv2.imread(str(scene.frame_image(name)))
                inst = gs.render_ids(tsp, gid, K, T, W, H, ids)
            out = crop_dir / f"{i}_{r}.jpg"
            if crop_masked(img, inst == num(i), out):
                crops.append(dict(key=f"{i}_{r}", path=str(out), weight=float(w), object=i))
    return crops


def combine_probs(sam_votes: dict, sig: np.ndarray | None, phrases: list[str], margin: float) -> dict:
    tot = sum(sam_votes.values())
    if tot <= 0 and sig is None:
        return dict(
            name="object (uncertain)",
            name_conf=0.0,
            runner_up=None,
            runner_up_conf=0.0,
            sam_top=None,
            sam_share=0.0,
            siglip_top=None,
            siglip_p=None,
        )
    psam = np.full(len(phrases), SAM_SMOOTHING)
    for p, v in sam_votes.items():
        if p in phrases and tot > 0:
            psam[phrases.index(p)] += v / tot
    psam /= psam.sum()
    p = psam * sig if sig is not None else psam
    p /= p.sum()
    o = np.argsort(-p)
    top, run = phrases[o[0]], phrases[o[1]]
    m = (p[o[0]] - p[o[1]]) / max(p[o[0]], 1e-9)
    return dict(
        name=top if m >= margin else f"{top} (uncertain)",
        name_conf=round(float(p[o[0]]), 3),
        runner_up=run,
        runner_up_conf=round(float(p[o[1]]), 3),
        sam_top=phrases[int(np.argmax(psam))],
        sam_share=round(float(psam.max()), 3),
        siglip_top=phrases[int(np.argmax(sig))] if sig is not None else None,
        siglip_p=round(float(sig.max()), 3) if sig is not None else None,
    )


def name_objects(
    alive: dict,
    keep_ids: list[str],
    votes: dict,
    crops: list[dict],
    siglip: dict,
    phrases: list[str],
    phrase_class: dict,
    margin: float,
) -> list[dict]:
    """siglip: crop key -> phrase probabilities."""
    out = []
    for i in keep_ids:
        cs = [c for c in crops if c["object"] == i and c["key"] in siglip]
        sig = None
        if cs:
            w = np.array([c["weight"] for c in cs])
            sig = (np.array([siglip[c["key"]] for c in cs]) * (w / w.sum())[:, None]).sum(0)
        r = combine_probs(votes.get(i, {}), sig, phrases, margin)
        out.append(
            dict(
                alive[i],
                **r,
                cls=phrase_class.get(r["name"].replace(" (uncertain)", ""), "items"),
                n_views_named=len(cs),
            )
        )
    return out
