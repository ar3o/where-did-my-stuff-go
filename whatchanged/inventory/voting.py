"""SAM 3 detections on the walk frames -> labelled 3D objects made of the splat's own Gaussians.

1. Gaussian votes per detection: each frame's detections form an exclusive label map (smaller masks win a pixel, so a
   banana is not part of its fruit bowl; pixels in no detection vote for background). The label map is rendered back
   through gsplat as a loss on one-hot per-Gaussian features, so the gradient gives every detection the Gaussians that
   produce its pixels, weighted by their blending weight. Up to 6000 Gaussians per detection are kept; elongated masks
   (principal-axis ratio >= 4) keep detections down to 10 Gaussians (others 30) so thin objects survive.
2. Association across frames: detections link when the cosine of their Gaussian-weight vectors is >= 0.2, strongest
   first, never two detections of one frame (union-find with cannot-link). Instances seen in fewer than 2 frames go.
3. Assignment: each Gaussian joins the instance with the largest accumulated weight if that is >= 50 % of all its votes
   (background included) and >= 0.02; Gaussians farther than 4 cm from the instance's dense core are dropped.
4. Region growing along thin parts: background Gaussians inside one of the instance's own detection masks, within
   1.5 cm of the instance and within 0.12 in colour of their nearest instance Gaussian, join it, until nothing joins.
5. Naming votes: each phrase's scores are z-normalised over all its detections, so a phrase that scores high everywhere
   gets no advantage; a detection votes sigmoid(z) x Ochiai overlap with the instance's rendered mask x the instance's
   visible fraction in that frame. Name = top phrase, "(uncertain)" when (top1 - top2) / top1 < 0.25.
6. Fragments with touching Gaussians (< 5 cm), the same winning phrase and mean colours within 0.15 merge; objects with
   fewer than 100 Gaussians are dropped.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from whatchanged.common import geometry as geo
from whatchanged.common.images import ochiai
from whatchanged.common.io import log
from whatchanged.splat import gaussians as gs

TOP_GAUSSIANS = 6000
BORDER_PX = 16
VOTE_CHUNK = 24


def elongation(m: np.ndarray) -> float:
    v, u = np.nonzero(m)
    if len(u) < 10:
        return 1.0
    X = np.c_[u, v].astype(np.float64)
    X -= X.mean(0)
    w = np.linalg.eigvalsh(X.T @ X / len(X))
    return float(np.sqrt(w[1] / max(w[0], 1e-6)))


def decide(votes: dict, margin: float) -> tuple[str, float]:
    if not votes:
        return "object (uncertain)", 0.0
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    top, v1 = ranked[0]
    v2 = ranked[1][1] if len(ranked) > 1 else 0.0
    m = (v1 - v2) / max(v1, 1e-9)
    return (top if m >= margin else f"{top} (uncertain)"), float(m)


def base_name(name: str) -> str:
    return name.replace(" (uncertain)", "")


def gaussian_votes(tsp: dict, N: int, frames: dict, dets_of: dict, K, W: int, H: int):
    """Per-detection Gaussian weights and the background votes, from label maps rendered back through gsplat."""
    from gsplat import rasterization

    dev = tsp["means"].device
    Kt = torch.as_tensor(K, dtype=torch.float32, device=dev)[None]
    dets, bg = [], np.zeros(N, np.float32)
    for name, d in dets_of.items():
        n = len(d)
        order = np.argsort([-m.sum() for _, _, m in d])
        labels = np.full((H, W), n, np.int64)
        for i in order:
            labels[d[i][2]] = i
        lt = torch.as_tensor(labels, device=dev)
        vm = torch.as_tensor(np.linalg.inv(frames[name]), dtype=torch.float32, device=dev)[None]
        for c0 in range(0, n + 1, VOTE_CHUNK):
            chans = list(range(c0, min(c0 + VOTE_CHUNK, n + 1)))
            onehot = torch.zeros((H, W, len(chans)), device=dev)
            for q, ch in enumerate(chans):
                onehot[..., q] = (lt == ch).float()
            b = BORDER_PX
            onehot[:b], onehot[-b:], onehot[:, :b], onehot[:, -b:] = 0, 0, 0, 0
            feat = torch.zeros((N, len(chans)), device=dev, requires_grad=True)
            img, _, _ = rasterization(
                tsp["means"],
                tsp["quats"],
                tsp["scales"],
                tsp["opacities"],
                feat,
                vm,
                Kt,
                W,
                H,
                sh_degree=None,
                packed=True,
                near_plane=gs.NEAR,
                far_plane=gs.FAR,
                rasterize_mode="antialiased",
                channel_chunk=32,
            )
            (img[0] * onehot).sum().backward()
            g = feat.grad
            for q, i in enumerate(chans):
                if i == n:
                    bg += g[:, q].cpu().numpy()
                    continue
                w = g[:, q]
                elong = elongation(d[i][2])
                k = min(TOP_GAUSSIANS, int((w > 1e-4).sum()))
                if k < (10 if elong >= 4 else 30):
                    continue
                v, ix = torch.topk(w, k)
                dets.append(
                    dict(
                        frame=name,
                        j=i,
                        phrase=d[i][0],
                        score=float(d[i][1]),
                        elong=elong,
                        gidx=ix.cpu().numpy().astype(np.int32),
                        gw=v.cpu().numpy().astype(np.float32),
                    )
                )
            del feat, img, onehot, g
    return dets, bg


def associate(dets: list[dict], N: int) -> list[list[int]]:
    from scipy import sparse

    rows = np.concatenate([np.full(len(x["gidx"]), i) for i, x in enumerate(dets)])
    cols = np.concatenate([x["gidx"] for x in dets])
    vals = np.concatenate([x["gw"] / np.linalg.norm(x["gw"]) for x in dets])
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(dets), N))
    S = (A @ A.T).tocoo()
    frame = np.array([x["frame"] for x in dets])
    edges = sorted(
        (
            (float(s), int(i), int(j))
            for i, j, s in zip(S.row, S.col, S.data)
            if i < j and s >= 0.2 and frame[i] != frame[j]
        ),
        reverse=True,
    )
    parent = list(range(len(dets)))
    fsets = [{frame[i]} for i in range(len(dets))]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, i, j in edges:
        a, b = find(i), find(j)
        if a != b and not (fsets[a] & fsets[b]):
            parent[b] = a
            fsets[a] |= fsets[b]
    groups = {}
    for i in range(len(dets)):
        groups.setdefault(find(i), []).append(i)
    return [c for c in groups.values() if len({frame[i] for i in c}) >= 2]


def assign(dets, clusters, bg, sp) -> np.ndarray:
    from scipy.spatial import cKDTree

    N = len(sp["means"])
    votes = np.zeros((len(clusters) + 1, N), np.float32)
    for ci, c in enumerate(clusters):
        for i in c:
            np.add.at(votes[ci], dets[i]["gidx"], dets[i]["gw"])
    votes[-1] = bg
    best = votes.argmax(0)
    frac = votes[best, np.arange(N)] / np.maximum(votes.sum(0), 1e-9)
    gid = np.where((best < len(clusters)) & (frac >= 0.5) & (votes[best, np.arange(N)] >= 0.02), best + 1, 0)
    gid = gid.astype(np.int32)
    del votes
    means = sp["means"]
    for ci in range(len(clusters)):
        m = gid == ci + 1
        if m.sum() < 30:
            gid[m] = 0
            continue
        Q = means[m]
        dense = Q[sp["opacities"][m] > 0.3] if (sp["opacities"][m] > 0.3).sum() >= 20 else Q
        r = np.linalg.norm(dense - np.median(dense, 0), axis=1)
        dd, _ = cKDTree(dense[r <= np.percentile(r, 90)]).query(Q, k=1)
        gid[np.flatnonzero(m)[dd > 0.04]] = 0
    return gid


def grow(dets, clusters, gid, means, rgb) -> dict[int, int]:
    from scipy.spatial import cKDTree

    grown = {}
    for ci, c in enumerate(clusters):
        if (gid == ci + 1).sum() < 30:
            continue
        inside = np.unique(np.concatenate([dets[i]["gidx"] for i in c]))
        inside = inside[gid[inside] == 0]
        if len(inside) == 0:
            continue
        added = 0
        for _ in range(30):
            obj_idx = np.flatnonzero(gid == ci + 1)
            cand = inside[gid[inside] == 0]
            if len(cand) == 0:
                break
            dd, nn = cKDTree(means[obj_idx]).query(means[cand], k=1)
            ok = (dd < 0.015) & (np.abs(rgb[cand] - rgb[obj_idx[nn]]).max(1) < 0.12)
            if not ok.any():
                break
            gid[cand[ok]] = ci + 1
            added += int(ok.sum())
        grown[ci] = added
    return grown


def phrase_stats(dets: list[dict]) -> dict[str, tuple[float, float]]:
    scores = {}
    for x in dets:
        scores.setdefault(x["phrase"], []).append(x["score"])
    return {p: (float(np.mean(v)), float(np.std(v) + 0.05)) for p, v in scores.items()}


def naming_votes(tsp, gid, n_clusters, frames, dets_of, stats, K, W, H) -> dict[int, dict]:
    ids = [0] + list(range(1, n_clusters + 1))
    vis, per = {}, {ci: {} for ci in range(n_clusters)}
    for name, d in dets_of.items():
        inst = gs.render_ids(tsp, gid, K, frames[name], W, H, ids)
        cnt = np.bincount(inst.ravel().astype(np.int64), minlength=len(ids))
        for ci in range(n_clusters):
            if cnt[ci + 1] > 0:
                vis.setdefault(ci, []).append(int(cnt[ci + 1]))
        for phrase, score, m in d:
            labs, inter = np.unique(inst[m], return_counts=True)
            for lab, n_in in zip(labs, inter):
                if lab == 0:
                    continue
                och = ochiai(n_in, m.sum(), cnt[lab])
                if och < 0.05:
                    continue
                mu, sd = stats.get(phrase, (0.5, 0.2))
                z = (float(score) - mu) / sd
                per[int(lab) - 1].setdefault(name, []).append((phrase, och / (1 + np.exp(-z)), int(cnt[lab])))
    maxvis = {ci: max(v) for ci, v in vis.items()}
    votes = {}
    for ci, frames_votes in per.items():
        vv = {}
        for rows in frames_votes.values():
            for p, w, c in rows:
                vv[p] = vv.get(p, 0.0) + w * c / max(maxvis.get(ci, 1), 1)
        votes[ci] = vv
    return votes


def merge_fragments(objs: dict, gid: np.ndarray, means: np.ndarray, rgb: np.ndarray, margin: float) -> None:
    from scipy.spatial import cKDTree

    changed = True
    while changed:
        changed = False
        keys = sorted(objs)
        for ai, a in enumerate(keys):
            for b in keys[ai + 1 :]:
                A, B = objs[a], objs[b]
                if base_name(A["name"]) != base_name(B["name"]) or np.abs(A["colour"] - B["colour"]).max() >= 0.15:
                    continue
                Pa, Pb = means[gid == a + 1], means[gid == b + 1]
                if len(Pa) > len(Pb):
                    Pa, Pb = Pb, Pa
                sub = Pa[np.random.default_rng(0).choice(len(Pa), min(len(Pa), 3000), replace=False)]
                if cKDTree(Pb).query(sub, k=1)[0].min() >= 0.05:
                    continue
                gid[gid == b + 1] = a + 1
                for p, v in B["votes"].items():
                    A["votes"][p] = A["votes"].get(p, 0.0) + v
                A["name"], A["margin"] = decide(A["votes"], margin)
                A["frames"] = sorted(set(A["frames"]) | set(B["frames"]))
                A["colour"] = rgb[gid == a + 1].mean(0)
                A["grown"] += B["grown"]
                objs.pop(b)
                changed = True
                break
            if changed:
                break


def build(
    sp: dict, frames: dict, dets_of: dict, K, W: int, H: int, phrase_class: dict, margin: float
) -> tuple[list, np.ndarray, dict]:
    """frames: name -> T_wc; dets_of: name -> merged detections. Returns (objects, gid per Gaussian, phrase stats)."""
    t0 = time.time()
    N = len(sp["means"])
    rgb = gs.rgb_of(sp)
    tsp = gs.to_torch(sp)
    dets, bg = gaussian_votes(tsp, N, frames, dets_of, K, W, H)
    log(f"inventory: {len(dets)} detections from {len(dets_of)} frames, votes {time.time() - t0:.0f} s")
    clusters = associate(dets, N)
    gid = assign(dets, clusters, bg, sp)
    grown = grow(dets, clusters, gid, sp["means"], rgb)
    stats = phrase_stats(dets)
    votes = naming_votes(tsp, gid, len(clusters), frames, dets_of, stats, K, W, H)
    objs = {}
    for ci in range(len(clusters)):
        m = gid == ci + 1
        if m.sum() < 30:
            continue
        name, mg = decide(votes.get(ci, {}), margin)
        objs[ci] = dict(
            votes=votes.get(ci, {}),
            name=name,
            margin=mg,
            colour=rgb[m].mean(0),
            grown=grown.get(ci, 0),
            frames=sorted({dets[i]["frame"] for i in clusters[ci]}),
        )
    merge_fragments(objs, gid, sp["means"], rgb, margin)
    out, remap = [], np.zeros(len(clusters) + 2, np.int32)
    for ci in sorted(objs):
        o = objs[ci]
        m = gid == ci + 1
        if m.sum() < 100:
            continue
        k = len(out) + 1
        remap[ci + 1] = k
        cc, sz, yaw = geo.obb_zup(sp["means"][m], 2, 98)
        out.append(
            dict(
                id=f"I{k}",
                name=o["name"],
                cls=phrase_class.get(base_name(o["name"]), "items"),
                name_margin=round(o["margin"], 3),
                name_votes={p: round(v, 3) for p, v in sorted(o["votes"].items(), key=lambda kv: -kv[1])[:4]},
                n_gauss=int(m.sum()),
                grown_gaussians=int(o["grown"]),
                n_views=len(o["frames"]),
                frames=o["frames"],
                box=dict(centre=cc.round(4).tolist(), size=sz.round(4).tolist(), yaw=round(float(yaw), 4)),
                colour=np.round(o["colour"], 3).tolist(),
            )
        )
    log(f"inventory: {len(out)} objects in {time.time() - t0:.0f} s")
    return out, remap[gid], stats
