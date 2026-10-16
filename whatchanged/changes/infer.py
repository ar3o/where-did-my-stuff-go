"""What changed in each photo group, compared with the reference splat and inventory only. Groups are independent
because the room was reset after each change.

Per usable photo:
  * object state from the checker's maps on the object's rendered mask (known pixels, away from the border):
    q_missing = calibrated P(missing | mean P(missing), size) and q_covered = calibrated P(appeared | mean P(appeared),
    size). The state is "missing" if q_missing >= 0.5, else "covered" (something in front, cannot verify) if
    q_covered >= 0.5, else "present". A view is clear when >= 50 % of the object (rendered alone) is visible.
  * SAM 3 detections. A detection lying on an inventory object's rendered mask (Ochiai >= 0.5) is claimed by it: the
    object is there, whatever the phrase. If that object reads missing, only when p_same >= 0.5. A detection that a
    same-phrase detection of the before image matches (IoU >= 0.3) existed before and is dropped (instance counting).
Per group:
  * unclaimed detections -> regions (one photo, IoU > 0.5, any phrase) -> 3D clusters across photos, with centres
    within the calibrated same-place radius.
  * a present object absorbs a cluster at its own place that carries its own name with p_same >= 0.5.
  * objects to search: clearly missing (>= 2 missing photos or one clear view; more missing than present photos and no
    clear present view) and unconfirmed ones (never seen present, but missing or covered somewhere).
  * one-to-one assignment (Hungarian, cost -log p_same over the cluster's detections with the object's name or
    runner-up) between searched objects and clusters, accepted at p_same >= 0.5. Found elsewhere means MOVED, found at
    its own place means unchanged. Missing, found nowhere and seen clearly means REMOVED; otherwise there is no change
    to report (cannot verify).
  * remaining clusters are ADDED if p_added (noisy-OR over photos) >= 0.5 and >= 2 photos agree, or one detection away
    from the border with no other photo seeing that place; at most 0.8 m. Regions with P(appeared) >= 0.5 and no
    detection are ADDED when >= 2 photos agree and the calibrated P(added) >= 0.5.
  * confidences: removed = noisy-OR of calibrated P(missing) over its photos; added = noisy-OR of p_added; moved =
    p_same (noisy-OR over photos) x P(left its place).
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from whatchanged.changes.language import sentence, summary_line
from whatchanged.changes.reid import Appearance, pair_features, phrases_of, predict_logit, sigmoid
from whatchanged.common import geometry as geo
from whatchanged.common.images import border_mask, ochiai, outline
from whatchanged.common.io import log, read_json, save_npz, unpack_masks, write_json
from whatchanged.common.photoview import PhotoView
from whatchanged.common.scene import Scene
from whatchanged.inventory.naming import combine_probs
from whatchanged.splat import gaussians as gs

MIN_VISIBLE_PX = 300
MIN_DET_PX = 150
MAX_ADDED_M = 0.8
BORDER_PX = 24
ACCEPT = 0.5
CLAIM_OCHIAI = 0.5
COUNTED_IOU = 0.3
REGION_IOU = 0.5
MIN_HEAT_AREA_M2 = 0.0025  # a 5 x 5 cm patch


def noisy_or(ps) -> float:
    return float(1 - np.prod([1 - p for p in ps])) if len(ps) else 0.0


class Context:
    """Splat, inventory, calibrations and appearance models shared by all groups."""

    def __init__(self, scene: Scene, tsp: dict, sp: dict, name_margin: float):
        self.scene, self.tsp, self.np = scene, tsp, gs.core(sp)
        inv = read_json(scene.inventory / "inventory.json")
        self.objs = [o for o in inv["objects"] if o["cls"] != "static"]
        self.gid = np.load(scene.inventory / "inventory.npz")["gid"]
        self.byid = {o["id"]: o for o in self.objs}
        self.static_ids = {int(o["id"][1:]) for o in inv["objects"] if o["cls"] == "static"}
        self.furniture = [o for o in inv["objects"] if o["cls"] == "furniture"]
        self.phrases = scene.phrases(with_static=False)
        self.static = set(scene.names()["static"])
        cal = read_json(scene.runs / "checker" / "conf_calibration.json")["types"]
        self.platt = {t: np.array(cal[t]["weights"]) for t in ("missing", "appeared")}
        rc = read_json(scene.runs / "changes" / "reid_calibration.json")
        self.m_same, self.m_add = rc["p_same"]["model"], rc["p_added"]["model"]
        self.r_rel, self.r_min = rc["r_ref_rel"], rc["r_ref_min_size_m"]
        self.app = Appearance(scene)
        self.name_margin = name_margin

    def radius(self, size) -> float:
        """Same-place radius for an object or region of this size."""
        return self.r_rel * max(float(np.max(size)), self.r_min)

    def q(self, typ: str, heat: float, size: float) -> float:
        w = self.platt[typ]
        return float(sigmoid(w[0] * heat + w[1] * np.log(max(size, 0.02)) + w[2]))

    def p_same(self, row: dict, o: dict) -> float:
        if row.get("box") is None or row.get("emb") is None:
            return 0.0
        return float(predict_logit(self.m_same, [pair_features(row, o, self.app)])[0])

    def p_added(self, row: dict) -> float:
        x = [row["p_appeared"], row["z"], np.log(max(max(row["box"]["size"]), 0.02))]
        return float(predict_logit(self.m_add, [x])[0])

    def alone_area(self, k: int, K, T, W: int, H: int) -> int:
        sub = {kk: v[self.gid == k] for kk, v in self.np.items()}
        return int((gs.render(gs.to_torch(sub), K, T, W, H, depth=False)["alpha"] > 0.5).sum())

    def same_place(self, c1, s1, c2, s2) -> bool:
        """Within the same-place radius, or the two boxes overlap (a shift too small to resolve)."""
        c1, c2 = np.asarray(c1, float), np.asarray(c2, float)
        s1, s2 = np.asarray(s1, float), np.asarray(s2, float)
        if np.linalg.norm(c1 - c2) < self.radius(np.maximum(s1, s2)):
            return True
        h1 = np.array([max(s1[:2])] * 2 + [s1[2]]) / 2
        h2 = np.array([max(s2[:2])] * 2 + [s2[2]]) / 2
        return bool((np.abs(c1 - c2) < h1 + h2).all())


class PhotoEvidence:
    """Object states, detections and claims of one photo."""

    def __init__(self, ctx: Context, name: str):
        scene = ctx.scene
        self.pv = pv = PhotoView(scene, name)
        self.name, self.good = name, pv.good
        if not pv.good:
            return
        J = read_json(scene.changes / "dets" / f"photo_{name}.json")
        z = np.load(scene.changes / "dets" / f"photo_{name}.npz")
        self.rows = J["rows"]
        self.M = unpack_masks(z["masks"], z["shape"])
        self.Mb = unpack_masks(z["bmasks"], z["bshape"])
        self.before_phrases = [b["phrase"] for b in J["before"]]
        self.inst = z["inst"].astype(np.int64)
        pz = np.load(scene.checker / "photos" / name / "probs.npz")
        self.P_app = cv2.resize(pz["appeared"].astype(np.float32), (pv.W, pv.H))
        self.P_mis = cv2.resize(pz["missing"].astype(np.float32), (pv.W, pv.H))
        inner = border_mask(pv.H, pv.W, BORDER_PX)
        self.states = {}
        for o in ctx.objs:
            k = int(o["id"][1:])
            R = (self.inst == k) & pv.known & inner
            n = int(R.sum())
            if n < MIN_VISIBLE_PX:
                self.states[o["id"]] = dict(state="cant_verify", px=n, vis=0.0, clear=False)
                continue
            vis = min(n / max(ctx.alone_area(k, pv.K, pv.T, pv.W, pv.H), 1), 1.0)
            size = max(o["box"]["size"])
            pm, pa = float(self.P_mis[R].mean()), float(self.P_app[R].mean())
            qm, qc = ctx.q("missing", pm, size), ctx.q("appeared", pa, size)
            st = "missing" if qm >= ACCEPT else "covered" if qc >= ACCEPT else "present"
            self.states[o["id"]] = dict(
                state=st,
                p_missing=round(pm, 3),
                p_appeared=round(pa, 3),
                q_missing=round(qm, 3),
                q_covered=round(qc, 3),
                px=n,
                vis=round(vis, 3),
                clear=bool(vis >= 0.5),
                mask=R,
            )
        self.claim_detections(ctx)

    def claim_detections(self, ctx: Context) -> None:
        cnt = np.bincount(self.inst.ravel(), minlength=int(self.inst.max()) + 1)
        self.claim, self.counted, self.usable = {}, set(), []
        for j, (row, m) in enumerate(zip(self.rows, self.M)):
            if row["phrase"] in ctx.static or row["box"] is None or row["known"] < 0.5 or row["px"] < MIN_DET_PX:
                continue
            self.usable.append(j)
            labs, inter = np.unique(self.inst[m], return_counts=True)
            best, owner = 0.0, None
            for lab, n_in in zip(labs, inter):
                if lab == 0 or lab in ctx.static_ids or f"I{lab}" not in ctx.byid:
                    continue
                och = ochiai(n_in, m.sum(), cnt[lab])
                if och > best:
                    best, owner = och, f"I{lab}"
            if owner is not None and best >= CLAIM_OCHIAI:
                if self.states[owner]["state"] != "missing":
                    self.claim[j] = (owner, "on_place")
                elif ctx.p_same(row, ctx.byid[owner]) >= ACCEPT:
                    self.claim[j] = (owner, "still_there")
            if j not in self.claim:
                ious = [
                    (m & mb).sum() / max((m | mb).sum(), 1)
                    for q, mb in zip(self.before_phrases, self.Mb)
                    if q == row["phrase"]
                ]
                if ious and max(ious) >= COUNTED_IOU:
                    self.counted.add(j)

    def centre(self, j: int) -> np.ndarray:
        return np.array(self.rows[j]["box"]["centre"])

    def claimed_by(self, oid: str) -> bool:
        return any(c[0] == oid for c in self.claim.values())


def regions_of(pe: PhotoEvidence) -> list[dict]:
    """Unclaimed, uncounted detections of one photo grouped by mask overlap (any phrase), largest first."""
    js = sorted((j for j in pe.usable if j not in pe.claim and j not in pe.counted), key=lambda j: -pe.rows[j]["px"])
    regs = []
    for j in js:
        m = pe.M[j]
        for r in regs:
            mm = pe.M[r["main"]]
            if (m & mm).sum() / max((m | mm).sum(), 1) > REGION_IOU:
                r["dets"].append(j)
                break
        else:
            regs.append(dict(main=j, dets=[j], photo=pe.name))
    for r in regs:
        r["centre"] = pe.centre(r["main"])
        r["size"] = np.array(pe.rows[r["main"]]["box"]["size"])
    return regs


def cluster_regions(ctx: Context, regs: list[dict], by_photo: dict) -> list[dict]:
    clusters = []
    for r in sorted(regs, key=lambda r: -by_photo[r["photo"]].rows[r["main"]]["px"]):
        for c in clusters:
            if r["photo"] in {x["photo"] for x in c["regs"]}:
                continue
            if np.linalg.norm(c["centre"] - r["centre"]) < ctx.radius(np.maximum(c["size"], r["size"])):
                c["regs"].append(r)
                break
        else:
            clusters.append(dict(regs=[r], centre=r["centre"], size=r["size"]))
    for c in clusters:
        c["centre"] = np.median([x["centre"] for x in c["regs"]], 0)
        c["photos"] = sorted({x["photo"] for x in c["regs"]})
        c["dets"] = [(x["photo"], j) for x in c["regs"] for j in x["dets"]]
    return clusters


def heat_regions(pe: PhotoEvidence, claimed_px: np.ndarray) -> list[dict]:
    """P(appeared) >= 0.5 regions that no detection and no present object explains, at least 5 x 5 cm."""
    pv = pe.pv
    present = np.zeros((pv.H, pv.W), bool)
    for s in pe.states.values():
        if s["state"] != "missing" and s.get("mask") is not None:
            present |= s["mask"]
    comp = ((pe.P_app >= ACCEPT) & pv.known & border_mask(pv.H, pv.W, BORDER_PX)).astype(np.uint8)
    comp = cv2.morphologyEx(comp, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    nl, lab, _, _ = cv2.connectedComponentsWithStats(comp)
    out = []
    for i in range(1, nl):
        m = lab == i
        if (m & claimed_px).sum() >= 0.5 * m.sum() or (m & present).sum() >= 0.5 * m.sum():
            continue
        d = pv.D_photo[m & (pv.D_photo > 0)]
        if len(d) == 0 or (np.median(d) / pv.K[0, 0]) ** 2 * m.sum() < MIN_HEAT_AREA_M2:
            continue
        b = geo.box3d(pv.lift(m, pv.D_photo))
        if b and max(b["size"]) <= MAX_ADDED_M:
            out.append(
                dict(
                    mask=m,
                    box=b,
                    centre=np.array(b["centre"]),
                    size=np.array(b["size"]),
                    p_appeared=float(pe.P_app[m].mean()),
                    photo=pe.name,
                )
            )
    return out


def evidence_at_place(ctx: Context, good: list[PhotoEvidence]) -> dict:
    ev = {}
    for o in ctx.objs:
        oid = o["id"]
        missing = [
            (pe.name, pe.states[oid]) for pe in good if pe.states[oid]["state"] == "missing" and not pe.claimed_by(oid)
        ]
        present = [
            (pe.name, pe.states[oid]) for pe in good if pe.states[oid]["state"] == "present" or pe.claimed_by(oid)
        ]
        covered = [
            (pe.name, pe.states[oid]) for pe in good if pe.states[oid]["state"] == "covered" and not pe.claimed_by(oid)
        ]
        strong = (
            (len(missing) >= 2 or any(s["clear"] for _, s in missing))
            and len(present) < len(missing)
            and not any(s.get("clear") for _, s in present)
        )
        ev[oid] = dict(
            missing=missing,
            present=present,
            covered=covered,
            strong_missing=strong,
            clear_missing=[i for i, s in missing if s["clear"]],
        )
    return ev


def name_cluster(ctx: Context, c: dict, by_photo: dict) -> dict:
    """SAM 3 votes on the cluster's detections (z-score x overlap with the region's main mask) x SigLIP 2 zero-shot."""
    votes, sig, wsum = {}, None, 0.0
    for x in c["regs"]:
        pe = by_photo[x["photo"]]
        mm = pe.M[x["main"]]
        for j in x["dets"]:
            row = pe.rows[j]
            iou = (pe.M[j] & mm).sum() / max((pe.M[j] | mm).sum(), 1)
            votes[row["phrase"]] = votes.get(row["phrase"], 0.0) + sigmoid(row["z"]) * iou
            if row["key"] in ctx.app.k:
                w = row["px"]
                p = ctx.app.prob[ctx.app.k[row["key"]]].astype(np.float64)
                sig = p * w if sig is None else sig + p * w
                wsum += w
    return combine_probs(votes, sig / wsum if sig is not None else None, ctx.phrases, ctx.name_margin)


def lifted_points(pe: PhotoEvidence, m: np.ndarray) -> dict | None:
    """Coloured points of an added thing (aligned photo depth inside its mask), for the film."""
    pv = pe.pv
    v, u = np.nonzero(m & (pv.D_photo > 0) & pv.known)
    if len(u) < 30:
        return None
    return dict(xyz=pv.lift(m, pv.D_photo).astype(np.float32), rgb=pv.rgb[v, u].astype(np.float32), photo=pe.name)


def run_group(ctx: Context, photos: list[str], k: int) -> tuple[list[dict], dict, dict, list[str]]:
    """Events of one group, the room state after it, the lifted points of added things and the usable photos."""
    pes = [PhotoEvidence(ctx, name) for name in photos]
    good = [pe for pe in pes if pe.good]
    by_photo = {pe.name: pe for pe in good}
    events, added, blobs = [], [], {}
    ev = evidence_at_place(ctx, good)
    clusters = cluster_regions(ctx, [r for pe in good for r in regions_of(pe)], by_photo)

    def cluster_p(c, o, own_name_only=False):
        allowed = {o["name"].replace(" (uncertain)", "")} if own_name_only else phrases_of(o)
        ps = {}
        for img, j in c["dets"]:
            row = by_photo[img].rows[j]
            if row["phrase"] in allowed:
                ps[img] = max(ps.get(img, 0.0), ctx.p_same(row, o))
        return (max(ps.values()) if ps else 0.0), ps

    used = set()
    for o in ctx.objs:
        if ev[o["id"]]["strong_missing"]:
            continue
        for ci, c in enumerate(clusters):
            if ci in used or not ctx.same_place(c["centre"], c["size"], o["box"]["centre"], o["box"]["size"]):
                continue
            if cluster_p(c, o, own_name_only=True)[0] >= ACCEPT:  # a runner-up phrase must not hide another object
                used.add(ci)
    search = []
    for o in ctx.objs:
        e = ev[o["id"]]
        if e["strong_missing"]:
            search.append(("missing", o))
        elif not e["present"] and (e["missing"] or e["covered"]):
            search.append(("unconfirmed", o))
    free = [ci for ci in range(len(clusters)) if ci not in used]
    Pm = np.zeros((len(search), len(free)))
    support = {}
    for si, (_, o) in enumerate(search):
        for fi, ci in enumerate(free):
            Pm[si, fi], support[(si, fi)] = cluster_p(clusters[ci], o)
    match = {}
    if Pm.size:
        ri, cj = linear_sum_assignment(np.where(Pm >= ACCEPT, -np.log(np.maximum(Pm, 1e-6)), 1e3))
        match = {a: free[b] for a, b in zip(ri, cj) if Pm[a, b] >= ACCEPT}
    for si, (why, o) in enumerate(search):
        oid = o["id"]
        src = np.array(o["box"]["centre"])
        if si in match:
            c = clusters[match[si]]
            used.add(match[si])
            ps = support[(si, free.index(match[si]))]
            dest = c["centre"]
            if ctx.same_place(dest, c["size"], src, o["box"]["size"]):
                continue
            p_reid = noisy_or([p for p in ps.values() if p >= ACCEPT])
            if why == "missing":
                p_left = noisy_or([s["q_missing"] for _, s in ev[oid]["missing"]])
            else:
                p_left = noisy_or(
                    [max(s["q_missing"], s["q_covered"]) for _, s in ev[oid]["missing"] + ev[oid]["covered"]]
                )
            events.append(
                dict(
                    kind="moved",
                    object=oid,
                    centre_before=src.tolist(),
                    centre_after=dest.tolist(),
                    box_after=dict(centre=dest.tolist(), size=np.asarray(c["size"]).tolist(), yaw=0.0),
                    photos=sorted(ps),
                    dist_m=round(float(np.linalg.norm(dest - src)), 3),
                    p_same=round(p_reid, 3),
                    change_conf=round(p_reid * p_left, 3),
                    after_dets={img: j for img, j in c["dets"] if img in ps},
                    why=why,
                )
            )
        elif why == "missing" and ev[oid]["clear_missing"]:
            conf = noisy_or([s["q_missing"] for _, s in ev[oid]["missing"]])
            events.append(
                dict(
                    kind="removed",
                    object=oid,
                    centre_before=o["box"]["centre"],
                    photos=[i for i, _ in ev[oid]["missing"]],
                    clear_photos=ev[oid]["clear_missing"],
                    change_conf=round(conf, 3),
                )
            )
    for ci, c in enumerate(clusters):
        if ci in used:
            continue
        if any(ctx.same_place(a["loc"], a["size"], c["centre"], c["size"]) for a in added):
            continue
        if any(
            ctx.same_place(
                e["centre_after"], (e.get("box_after") or {}).get("size") or c["size"], c["centre"], c["size"]
            )
            for e in events
            if e["kind"] == "moved"
        ):
            continue  # another view of an object just re-identified
        if max(c["size"]) > MAX_ADDED_M:
            continue
        pa = {}
        for img, j in c["dets"]:
            pa[img] = max(pa.get(img, 0.0), ctx.p_added(by_photo[img].rows[j]))
        conf = noisy_or(list(pa.values()))
        others = [pe.name for pe in good if pe.name not in c["photos"] and pe.pv.can_see(c["centre"])]
        touches = all(by_photo[x["photo"]].rows[x["main"]]["touches"] for x in c["regs"])
        if conf < ACCEPT or not (len(c["photos"]) >= 2 or (not touches and not others)):
            continue
        nm = name_cluster(ctx, c, by_photo)
        aid = f"g{k:02d}A{len(added) + 1}"
        main = max(c["regs"], key=lambda x: by_photo[x["photo"]].rows[x["main"]]["px"])
        box = dict(centre=c["centre"].tolist(), size=np.asarray(c["size"]).tolist(), yaw=0.0)
        added.append(
            dict(
                id=aid,
                name=nm["name"],
                name_conf=nm["name_conf"],
                runner_up=nm["runner_up"],
                runner_up_conf=nm["runner_up_conf"],
                loc=c["centre"].tolist(),
                size=box["size"],
                box=box,
                conf=round(conf, 3),
            )
        )
        blobs[aid] = lifted_points(by_photo[main["photo"]], by_photo[main["photo"]].M[main["main"]])
        events.append(
            dict(
                kind="added",
                object=aid,
                centre_after=c["centre"].tolist(),
                box_after=box,
                photos=c["photos"],
                change_conf=round(conf, 3),
                source="counting",
                after_dets={x["photo"]: x["main"] for x in c["regs"]},
                blob_photo=main["photo"],
            )
        )
    heat = []
    for pe in good:
        claimed_px = np.zeros((pe.pv.H, pe.pv.W), bool)
        for j in pe.usable:
            claimed_px |= pe.M[j]
        heat += heat_regions(pe, claimed_px)
    heat_clusters = []
    for r in heat:
        for c in heat_clusters:
            if r["photo"] not in {x["photo"] for x in c} and np.linalg.norm(c[0]["centre"] - r["centre"]) < ctx.radius(
                np.maximum(c[0]["size"], r["size"])
            ):
                c.append(r)
                break
        else:
            heat_clusters.append([r])
    for c in heat_clusters:
        if len({x["photo"] for x in c}) < 2:
            continue
        cen = np.median([x["centre"] for x in c], 0)
        main = max(c, key=lambda x: x["mask"].sum())
        if any(ctx.same_place(a["loc"], a["size"], cen, main["size"]) for a in added):
            continue
        if any(
            np.linalg.norm(np.array(e.get("centre_after") or [9, 9, 9]) - cen) < ctx.radius(e.get("size") or [0.1])
            for e in events
        ):
            continue
        conf = noisy_or([ctx.q("appeared", x["p_appeared"], max(x["size"])) for x in c])
        if conf < ACCEPT:
            continue
        aid = f"g{k:02d}A{len(added) + 1}"
        box = dict(centre=cen.tolist(), size=main["size"].tolist(), yaw=0.0)
        added.append(
            dict(
                id=aid,
                name="object",
                name_conf=0.0,
                runner_up=None,
                runner_up_conf=0.0,
                loc=cen.tolist(),
                size=box["size"],
                box=box,
                conf=round(conf, 3),
                heat_only=True,
            )
        )
        blobs[aid] = lifted_points(by_photo[main["photo"]], main["mask"])
        events.append(
            dict(
                kind="added",
                object=aid,
                centre_after=cen.tolist(),
                box_after=box,
                photos=sorted({x["photo"] for x in c}),
                change_conf=round(conf, 3),
                source="heatmap",
                heat_masks={x["photo"]: outline(x["mask"]) for x in c},
                blob_photo=main["photo"],
            )
        )
    describe(ctx, events, added, by_photo)
    events.sort(key=lambda e: -e["change_conf"])
    state = dict(
        removed=[e["object"] for e in events if e["kind"] == "removed"],
        moved={e["object"]: e["centre_after"] for e in events if e["kind"] == "moved"},
        added=added,
    )
    return events, state, blobs, [pe.name for pe in good]


def describe(ctx: Context, events: list[dict], added: list[dict], by_photo: dict) -> None:
    """Names, sizes, per-photo outlines (before / after) and per-photo evidence of every event."""
    adds = {a["id"]: a for a in added}
    for e in events:
        oid = e["object"]
        o = ctx.byid.get(oid) or adds.get(oid)
        e.update(
            name=o["name"],
            name_conf=o.get("name_conf", 0.0),
            runner_up=o.get("runner_up"),
            runner_up_conf=o.get("runner_up_conf", 0.0),
        )
        e["size"] = o["box"]["size"] if "box" in o else o["size"]
        outlines = {}
        for img in set(e.get("photos", [])) | set(e.get("after_dets", {})):
            if img not in by_photo:
                continue
            pe = by_photo[img]
            d = {}
            st = pe.states.get(oid)
            if st is not None and st.get("mask") is not None and e["kind"] in ("removed", "moved"):
                d["before"] = outline(st["mask"])
            if img in e.get("after_dets", {}):
                d["after"] = outline(pe.M[e["after_dets"][img]])
            if img in e.get("heat_masks", {}):
                d["after"] = e["heat_masks"][img]
            if d:
                outlines[img] = d
        e["outlines"] = outlines
        e.pop("heat_masks", None)
    for e in events:
        e["evidence"] = {
            i: {kk: v for kk, v in by_photo[i].states.get(e["object"], {}).items() if kk != "mask"}
            for i in e.get("photos", [])
            if i in by_photo
        }


def run_all(ctx: Context, groups: dict[int, list[str]]) -> dict:
    t0 = time.time()
    summary = {}
    for k in sorted(groups):
        events, state, blobs, good = run_group(ctx, groups[k], k)
        od = ctx.scene.changes / f"g{k:02d}"
        od.mkdir(parents=True, exist_ok=True)
        for e in events:
            e["sentence"] = sentence(e, ctx.furniture)
        for aid, b in blobs.items():
            if b is not None:
                save_npz(od / f"blob_{aid}.npz", compressed=True, **b)
        write_json(
            dict(
                group=k,
                photos=groups[k],
                good_photos=good,
                events=events,
                state_after=state,
                same_place_rule=dict(r_ref_rel=ctx.r_rel, min_size_m=ctx.r_min),
            ),
            od / "changes.json",
            indent=None,
        )
        summary[k] = [summary_line(e) for e in events]
        log(f"g{k:02d}: {summary[k]}")
    write_json(dict(groups=summary, wall_s=round(time.time() - t0, 1)), ctx.scene.changes / "summary.json")
    return summary
