"""Training and validation pairs for the change checker: edit the reference, not reality.

    python -m whatchanged.checker.pairs --scene config1

after  = a real walk frame (never a query photo);
before = a render of an edited splat at that frame's pose, with 0-2 cm / 0-0.5 degree pose jitter so tiny misalignment
         is not a change.
Edits on the render side flip their meaning:
  remove X from the render (X is in the real frame)    -> "appeared" on X's pixels at the true pose
  transplant a copy of X to a visible horizontal spot  -> "missing" on the copy's pixels in the edited render
  move X 2-10 or 15-60 cm (optional yaw), rotate 5-90  -> "missing" at the destination, "appeared" at the start,
                                                          the overlap ignored
  no edit (about 35 %)                                 -> real render-vs-photo nuisance (floaters, blur, colour)
Vacated footprints are filled with flat inpainted Gaussians. Decoy fills of the same kind go into 30 % of unedited and
15 % of edited renders, so "a filled hole means a change" is not a shortcut. Pixels the other reference cameras did not
see (frame_known), and a 16 px border, are ignored (255). The real side gets colour gain / gamma / saturation / white
balance, blur or sharpening, noise and JPEG 60-95. Split: the last 20 % of the walk is validation.
Per pair (checker/pairs/<split>/<frame>_<j>/): render.jpg, real.jpg, depth_r.npy, depth_q.npy (MoGe aligned to the
render depth, as at test time), known.npy, label.png (0 / 1 appeared / 2 missing / 255), edit_masks.npz, meta.json.
"""

from __future__ import annotations

import argparse
import math
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch

from whatchanged.checker.settings import load_settings
from whatchanged.common import depth as depthlib
from whatchanged.common.geometry import quat_wxyz_from_R, rot_z
from whatchanged.common.gpu import gpu_stage
from whatchanged.common.io import log, read_json, write_json
from whatchanged.common.scene import Scene
from whatchanged.splat import gaussians as gs

BORDER = 16
MIN_PX = 600
MAX_EDIT_SIZE_M = 0.8
DUP_ID0 = 1000
APPEARED, MISSING, IGNORE = 1, 2, 255


def qmul_wxyz_t(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def point_in_poly(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Points inside the convex hull of poly, by rasterising the hull at 2 mm."""
    path = cv2.convexHull(poly.astype(np.float32))[:, 0]
    lo = path.min(0) - 0.01
    cell = 0.002
    sh = np.ceil((path.max(0) + 0.01 - lo) / cell).astype(int) + 1
    img = np.zeros((sh[1], sh[0]), np.uint8)
    cv2.fillConvexPoly(img, np.round((path - lo) / cell).astype(np.int32), 1)
    ij = np.floor((xy - lo) / cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < sh[0]) & (ij[:, 1] < sh[1])
    out = np.zeros(len(xy), bool)
    out[ok] = img[ij[ok, 1], ij[ok, 0]] > 0
    return out


class World:
    """The reference splat with its objects (Gaussian groups) and the edits applied to it."""

    def __init__(self, sp: dict, gid: np.ndarray):
        from scipy.spatial import cKDTree

        self.np = gs.core(sp)
        self.base = gs.to_torch(self.np)
        self.gid = gid.astype(np.int64)
        means = self.np["means"]
        self.obj_ids = sorted(int(x) for x in np.unique(self.gid) if x > 0)
        self.idx = {o: np.flatnonzero(self.gid == o) for o in self.obj_ids}
        self.cen = {o: np.median(means[self.idx[o]], 0) for o in self.obj_ids}
        self.bot = {o: float(np.percentile(means[self.idx[o]][:, 2], 2)) for o in self.obj_ids}
        self.ext = {
            o: np.percentile(means[self.idx[o]], 98, 0) - np.percentile(means[self.idx[o]], 2, 0) for o in self.obj_ids
        }
        self.bg = np.flatnonzero(self.gid == 0)
        self.tree = cKDTree(means[self.bg][:, :2])

    def render(self, tsp, K, T, W, H, gid, ids):
        return gs.render_view(tsp, gid, K, T, W, H, ids)

    def edited(self, edits: list[dict], fills: list[dict]):
        """Splat tensors and gid after the edits (dict(type, obj, R, t, dup_id)) and the fill patches."""
        b = self.base
        means, quats, opac, sh = b["means"].clone(), b["quats"].clone(), b["opacities"].clone(), b["sh"]
        scales = b["scales"].clone()
        gid = self.gid.copy()
        extra = []
        dev = means.device
        for e in edits:
            ii = torch.as_tensor(self.idx[e["obj"]], device=dev)
            if e["type"] == "remove":
                opac[ii] = 0
                continue
            R = torch.as_tensor(e["R"], dtype=torch.float32, device=dev)
            c = torch.as_tensor(self.cen[e["obj"]], dtype=torch.float32, device=dev)
            t = torch.as_tensor(e["t"], dtype=torch.float32, device=dev)
            q = torch.as_tensor(quat_wxyz_from_R(e["R"]), dtype=torch.float32, device=dev)
            if e["type"] in ("move", "rotate"):
                means[ii] = (means[ii] - c) @ R.T + c + t
                quats[ii] = qmul_wxyz_t(q.expand(len(ii), 4), quats[ii])
            elif e["type"] == "add":
                extra.append(
                    dict(
                        means=(means[ii] - c) @ R.T + c + t,
                        quats=qmul_wxyz_t(q.expand(len(ii), 4), b["quats"][ii]),
                        scales=b["scales"][ii],
                        opacities=b["opacities"][ii],
                        sh=b["sh"][ii],
                        gid=np.full(len(ii), e["dup_id"]),
                    )
                )
        for f in fills:
            opac[torch.as_tensor(f["delete"], device=dev, dtype=torch.long)] = 0
            extra.append(
                dict(**{k: torch.as_tensor(f[k], device=dev) for k in gs.KEYS}, gid=np.zeros(len(f["means"]), np.int64))
            )
        tsp = dict(means=means, quats=quats, scales=scales, opacities=opac, sh=sh)
        for x in extra:
            for k in tsp:
                tsp[k] = torch.cat([tsp[k], x[k].float()], 0)
            gid = np.concatenate([gid, x["gid"]])
        return tsp, gid

    def plane_fill(self, poly: np.ndarray, z0: float, rng, cell: float = 0.006):
        """Background Gaussians to delete inside a footprint and flat inpainted fill Gaussians that replace them."""
        lo, hi = poly.min(0) - 0.08, poly.max(0) + 0.08
        cand = self.bg[self.tree.query_ball_point((lo + hi) / 2, r=float(np.linalg.norm(hi - lo) / 2))]
        if len(cand) == 0:
            return None
        M = self.np["means"][cand]
        inside = (np.abs(M[:, 2] - z0) < 0.02) & point_in_poly(M[:, :2], poly)
        keep = cand[~inside]
        Mk = self.np["means"][keep]
        near = (np.abs(Mk[:, 2] - z0) < 0.015) & (self.np["opacities"][keep] > 0.2)
        sh = np.ceil((hi - lo) / cell).astype(int) + 1
        ij = np.floor((Mk[near, :2] - lo) / cell).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < sh[0]) & (ij[:, 1] < sh[1])
        col = np.clip(self.np["sh"][keep][near][ok, 0] * gs.C0 + 0.5, 0, 1)
        w = self.np["opacities"][keep][near][ok]
        acc = np.zeros((sh[1], sh[0], 3))
        wa = np.zeros((sh[1], sh[0]))
        np.add.at(acc, (ij[ok, 1], ij[ok, 0]), col * w[:, None])
        np.add.at(wa, (ij[ok, 1], ij[ok, 0]), w)
        have = wa > 0.3
        if have.sum() < 20:
            return None
        img = np.where(have[..., None], acc / np.maximum(wa, 1e-6)[..., None], 0)
        pm = np.zeros((sh[1], sh[0]), np.uint8)
        cv2.fillConvexPoly(pm, cv2.convexHull(np.round((poly - lo) / cell).astype(np.int32))[:, 0], 1)
        fill = (
            cv2.inpaint(
                (np.clip(img, 0, 1) * 255).astype(np.uint8), ((pm > 0) | ~have).astype(np.uint8), 6, cv2.INPAINT_TELEA
            )
            / 255.0
        )
        ring = (cv2.dilate(pm, np.ones((15, 15), np.uint8)) > 0) & (pm == 0) & have
        luma = img @ np.array([0.2126, 0.7152, 0.0722])
        std = float(np.clip(luma[ring].std(), 0.005, 0.05)) if ring.sum() > 20 else 0.02
        grain = cv2.GaussianBlur(rng.normal(0, 1, pm.shape).astype(np.float32), (0, 0), 1.2)
        grain = grain / max(float(grain.std()), 1e-6) * 0.7 * std
        ys, xs = np.nonzero(pm)
        n = len(xs)
        rgb = np.clip(fill[ys, xs] + grain[ys, xs][:, None], 0, 1)
        means = np.c_[lo[0] + (xs + 0.5) * cell, lo[1] + (ys + 0.5) * cell, np.full(n, z0 + 0.0005)].astype(np.float32)
        return dict(
            delete=cand[inside],
            means=means,
            quats=np.tile([1.0, 0, 0, 0], (n, 1)).astype(np.float32),
            scales=np.tile([0.75 * cell, 0.75 * cell, 0.0008], (n, 1)).astype(np.float32),
            opacities=np.full(n, 0.97, np.float32),
            sh=np.concatenate([((rgb - 0.5) / gs.C0)[:, None, :], np.zeros((n, 15, 3))], 1).astype(np.float32),
        )

    def footprint(self, o: int, R=None, t=None, dil: float = 0.015) -> np.ndarray:
        R = np.eye(3) if R is None else R
        t = np.zeros(3) if t is None else t
        P = (self.np["means"][self.idx[o]] - self.cen[o]) @ R.T + self.cen[o] + t
        h = cv2.convexHull(P[:, :2].astype(np.float32))[:, 0].astype(np.float64)
        c = h.mean(0)
        d = h - c
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return c + d * (n + dil) / np.maximum(n, 1e-9)


def horizontal_spots(D, K, T, inst, known, rng, n: int = 40) -> list:
    """World points on visible, known, near-horizontal background surfaces (normal z > 0.9) within 3 m."""
    H, W = D.shape
    v, u = np.mgrid[0:H, 0:W]
    X = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * D, (v + 0.5 - K[1, 2]) / K[1, 1] * D, D], -1) @ T[:3, :3].T + T[:3, 3]
    gx, gy = np.zeros_like(X), np.zeros_like(X)
    gx[:, 1:-1] = X[:, 2:] - X[:, :-2]
    gy[1:-1] = X[2:] - X[:-2]
    nrm = np.cross(gx, gy)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-9)
    ok = (np.abs(nrm[..., 2]) > 0.9) & (D > 0.3) & (D < 3.0) & (inst == 0) & known
    ok[:40], ok[-40:], ok[:, :40], ok[:, -40:] = False, False, False, False
    ys, xs = np.nonzero(ok)
    if len(xs) == 0:
        return []
    sel = rng.choice(len(xs), min(n, len(xs)), replace=False)
    return [X[ys[i], xs[i]] for i in sel]


def augment_real(img: np.ndarray, rng) -> np.ndarray:
    x = img.astype(np.float32) / 255
    x = x * rng.uniform(0.8, 1.2) * rng.normal(1.0, 0.04, 3)
    x = np.clip(x, 0, 1) ** rng.uniform(0.85, 1.15)
    grey = x.mean(-1, keepdims=True)
    x = np.clip(grey + (x - grey) * rng.uniform(0.8, 1.25), 0, 1)
    r = rng.random()
    if r < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.5))
    elif r < 0.5:
        b = cv2.GaussianBlur(x, (0, 0), 1.0)
        x = np.clip(x + rng.uniform(0.2, 0.6) * (x - b), 0, 1)
    x = np.clip(x + rng.normal(0, rng.uniform(0, 0.02), x.shape), 0, 1)
    _, enc = cv2.imencode(".jpg", (x * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def jitter(T: np.ndarray, rng) -> np.ndarray:
    R = cv2.Rodrigues(rng.normal(size=3) / np.sqrt(3) * np.radians(rng.uniform(0, 0.5)))[0]
    T2 = T.copy()
    T2[:3, :3] = R @ T[:3, :3]
    d = rng.normal(size=3)
    T2[:3, 3] += d / np.linalg.norm(d) * rng.uniform(0, 0.02)
    return T2


def plan_edit(world: World, typ: str, o: int, spots: list, n_done: int, rng):
    """One edit of object o -> (edit, fill patch or None, info), or None when it cannot be placed. A transplanted copy
    gets the id DUP_ID0 + the number of edits already planned."""
    w = world
    if typ == "remove":
        return (
            dict(type="remove", obj=o),
            w.plane_fill(w.footprint(o), w.bot[o], rng),
            dict(type="remove", obj=o, size=float(max(w.ext[o]))),
        )
    if typ == "add":
        if not spots:
            return None
        p = spots[int(rng.integers(len(spots)))]
        R = rot_z(rng.uniform(-np.pi, np.pi))
        t = np.array([p[0] - w.cen[o][0], p[1] - w.cen[o][1], p[2] - w.bot[o] + 0.003])
        dup = DUP_ID0 + n_done
        return (
            dict(type="add", obj=o, R=R, t=t, dup_id=dup),
            None,
            dict(type="add", obj=o, dup_id=dup, size=float(max(w.ext[o])), dist=float(np.linalg.norm(t[:2]))),
        )
    if typ == "move":
        if rng.random() < 0.5:
            a = rng.uniform(0, 2 * np.pi)
            d = rng.uniform(0.02, 0.10)
            t = np.array([d * math.cos(a), d * math.sin(a), 0.0])
        elif spots:
            p = spots[int(rng.integers(len(spots)))]
            t = np.array([p[0] - w.cen[o][0], p[1] - w.cen[o][1], p[2] - w.bot[o] + 0.003])
            if not (0.15 <= np.linalg.norm(t[:2]) <= 1.2):
                a = rng.uniform(0, 2 * np.pi)
                d = rng.uniform(0.15, 0.6)
                t = np.array([d * math.cos(a), d * math.sin(a), 0.0])
        else:
            return None
        R = rot_z(np.radians(rng.uniform(-30, 30))) if rng.random() < 0.4 else np.eye(3)
    else:
        t = np.zeros(3)
        R = rot_z(np.radians(rng.choice([-1, 1]) * rng.uniform(5, 90)))
    fill = w.plane_fill(w.footprint(o), w.bot[o], rng)
    info = dict(
        type=typ,
        obj=o,
        size=float(max(w.ext[o])),
        dist=float(np.linalg.norm(t[:2])),
        yaw_deg=float(np.degrees(math.atan2(R[1, 0], R[0, 0]))),
        small=bool(typ == "move" and np.linalg.norm(t[:2]) <= 0.1),
    )
    return dict(type=typ, obj=o, R=R, t=t), fill, info


def make_pair(
    w: World, f: dict, K, W: int, H: int, known, D_mono, img_real, rng, pid: str, split: str, root: Path
) -> str:
    T = np.array(f["T_wc"])
    r0 = w.render(w.base, K, T, W, H, w.gid, [0] + w.obj_ids + list(range(DUP_ID0, DUP_ID0 + 10)))
    inst0 = r0["inst"].astype(np.int64)
    vis = [o for o in w.obj_ids if ((inst0 == o) & known).sum() >= MIN_PX and max(w.ext[o]) <= MAX_EDIT_SIZE_M]
    u = rng.random()
    kind = (
        "none"
        if (u < 0.35 or not vis)
        else "remove"
        if u < 0.55
        else "add"
        if u < 0.70
        else "move"
        if u < 0.90
        else "rotate"
    )
    n_edit = 0 if kind == "none" else (1 if rng.random() < 0.7 else 2)
    spots = horizontal_spots(r0["depth"], K, T, inst0, known, rng)
    edits, fills, info, used = [], [], [], set()
    for _ in range(n_edit):
        cands = [o for o in vis if o not in used]
        if not cands:
            break
        o = int(rng.choice(cands))
        used.add(o)
        typ = kind if len(edits) == 0 else str(rng.choice(["remove", "add", "move", "rotate"]))
        planned = plan_edit(w, typ, o, spots, len(edits), rng)
        if planned is None:
            continue
        edit, fill, inf = planned
        edits.append(edit)
        if fill:
            fills.append(fill)
        info.append(inf)
    if rng.random() < (0.3 if not edits else 0.15) and spots and vis:
        o = int(rng.choice(vis))
        p = spots[int(rng.integers(len(spots)))]
        fp = w.footprint(o, rot_z(rng.uniform(-np.pi, np.pi)), np.array([p[0] - w.cen[o][0], p[1] - w.cen[o][1], 0]))
        fill = w.plane_fill(fp, float(p[2]), rng)
        if fill:
            fills.append(fill)
            info.append(dict(type="decoy_fill", shape_of=o))
    tsp, gid = w.edited(edits, fills)
    Tj = jitter(T, rng)
    ids = [0] + sorted({e["obj"] for e in edits} | {e["dup_id"] for e in edits if e["type"] == "add"})
    r1 = w.render(tsp, K, Tj, W, H, gid, ids)
    inst1 = r1["inst"].astype(np.int64)
    appeared, missing, both = np.zeros((H, W), bool), np.zeros((H, W), bool), np.zeros((H, W), bool)
    for e, inf in zip(edits, info):
        o = e["obj"]
        if e["type"] == "remove":
            m = inst0 == o
            appeared |= m
            inf["px_appeared"] = int(m.sum())
        elif e["type"] == "add":
            m = inst1 == e["dup_id"]
            missing |= m
            inf["px_missing"] = int(m.sum())
        else:
            a, b = inst0 == o, inst1 == o
            appeared |= a & ~b
            missing |= b & ~a
            both |= a & b
            inf["px_appeared"], inf["px_missing"] = int((a & ~b).sum()), int((b & ~a).sum())
    lab = np.zeros((H, W), np.uint8)
    lab[appeared] = APPEARED
    lab[missing] = MISSING
    lab[(appeared & missing) | both] = IGNORE
    lab[~known] = IGNORE
    lab[:BORDER], lab[-BORDER:], lab[:, :BORDER], lab[:, -BORDER:] = IGNORE, IGNORE, IGNORE, IGNORE
    D_render = r1["depth"]
    D_real, fit = depthlib.align_to_render(D_render, D_mono, known)
    od = root / split / pid
    od.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(od / "render.jpg"), (r1["rgb"][..., ::-1] * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(od / "real.jpg"), augment_real(img_real, rng), [cv2.IMWRITE_JPEG_QUALITY, 97])
    np.save(od / "depth_r.npy", D_render.astype(np.float16))
    np.save(od / "depth_q.npy", D_real.astype(np.float16))
    np.save(od / "known.npy", known)
    cv2.imwrite(str(od / "label.png"), lab)
    masks = {}
    for i, e in enumerate(edits):
        masks[f"e{i}_appeared"] = (
            (inst0 == e["obj"]) & (lab == APPEARED) if e["type"] != "add" else np.zeros((H, W), bool)
        )
    for i, e in enumerate(edits):
        masks[f"e{i}_missing"] = (inst1 == (e["dup_id"] if e["type"] == "add" else e["obj"])) & (lab == MISSING)
    np.savez_compressed(od / "edit_masks.npz", **masks)
    kind = kind if edits else "none"
    write_json(
        dict(
            frame=f["name"],
            t=f["t"],
            split=split,
            kind=kind,
            edits=info,
            jitter_cm=float(np.linalg.norm(Tj[:3, 3] - T[:3, 3]) * 100),
            align=fit,
            K=K.tolist(),
            T_wc=Tj.tolist(),
            label_px={
                "appeared": int((lab == APPEARED).sum()),
                "missing": int((lab == MISSING).sum()),
                "ignore": int((lab == IGNORE).sum()),
            },
        ),
        od / "meta.json",
    )
    return kind


def make_pairs(scene: Scene, world: World, per_frame: int, root: Path) -> dict:
    P = scene.poses()
    K, W, H = np.array(P["K"]), P["W"], P["H"]
    split_of = scene.walk_split()
    counts = {}
    t0 = time.time()
    for fi, f in enumerate(P["frames"]):
        split = split_of[f["name"]]
        if all((root / split / f"{f['name']}_{j}" / "meta.json").exists() for j in range(per_frame)):
            continue
        known = np.load(scene.frame_known / f"{f['name']}.npy")
        D_mono = depthlib.load_mono(scene.depth / "frames" / f"{f['name']}.npz", (W, H))
        img = cv2.imread(str(scene.frame_image(f["name"])))
        for j in range(per_frame):
            rng = np.random.default_rng(zlib.crc32(f"{f['name']}_{j}".encode()))
            k = make_pair(world, f, K, W, H, known, D_mono, img, rng, f"{f['name']}_{j}", split, root)
            counts[f"{split}:{k}"] = counts.get(f"{split}:{k}", 0) + 1
        if fi % 20 == 0:
            log(f"pairs: frame {fi}/{len(P['frames'])}, {time.time() - t0:.0f} s, {counts}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()
    cfg = load_settings(args.config).pairs
    scene = Scene(args.scene)
    inv = read_json(scene.inventory / "inventory.json")["objects"]
    gid = np.load(scene.inventory / "inventory.npz")["gid"]
    movable = {int(o["id"][1:]) for o in inv if o["cls"] != "static"}
    gid = np.where(np.isin(gid, list(movable)), gid, 0)
    with gpu_stage("pairs", need_mib=cfg.need_mib):
        world = World(gs.load(scene.splat), gid)
        log(f"pairs: {len(world.obj_ids)} editable objects")
        counts = make_pairs(scene, world, cfg.per_frame, scene.checker / "pairs")
    write_json(dict(counts=counts), scene.checker / "pairs_summary.json")


if __name__ == "__main__":
    main()
