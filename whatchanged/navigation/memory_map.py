"""Floor plane and 2-D walkable map of the memory scan (splat + named 3-D inventory).

    python -m whatchanged.navigation.memory_map --scene config1

The GPU is used only when the reference depth cache (splat/ref_depth.npz, whatchanged.splat.support) is missing.

1. Evidence = the splat's depth rendered from every registered memory camera (support.reference_depths, cached).
   Depth-edge pixels (relative jump > 4 % to a 4-neighbour: blended "flying" depths) are dropped.
2. Floor: RANSAC planes (geometry.ransac_plane, 1 cm) on the back-projected points, repeated on the remaining points;
   candidates are near-horizontal (normal within 15 deg of the memory frame's up), hold >= 3 % of the points and have
   >= 95 % of the cameras above them; the lowest level (largest plane within 5 cm of the lowest) is the floor. Floor
   frame F: plane -> z = 0, +z up (T_F_S). Local floor level per cell = the mode of the heights of near-floor points
   (-6..20 cm) pooled over a 45 cm region (a robust horizontal plane per region: shaggy rugs are the walking surface),
   3x3 median filtered; sigma = 1.4826 MAD of points within 6 cm of their cell's level (p98 = 98th percentile of
   |residual|). Obstacle band = [max(band_lo_min, 3 sigma, p98, step height), band_hi] above the local floor level.
3. Voxels (cell x cell x 5 cm): number of distinct memory cameras whose rendered surface falls in each voxel, carved by
   the rays that pass through. A cell is OCCUPIED if some band voxel is seen by >= min_views cameras, FLOOR-SEEN if
   its floor slab is seen by >= min_views cameras. FREE = floor-seen and not occupied; UNKNOWN = neither (never assumed
   free).
4. Inventory layer: Gaussians of every non-static inventory object (opacity >= 0.3) inside the band -> per-object cell
   opacity mass; a cell with mass >= 2 of one object belongs to that object's footprint (also marks it occupied:
   transparent furniture renders through). Cell owner = static (depth evidence or background Gaussians dominate) or the
   dominant inventory object. Oriented boxes in F from each object's Gaussians (2-98 % quantiles, z-up OBB).
5. Inflation: distance (m) from every cell to the nearest OCCUPIED or UNKNOWN cell; a robot centre may stand where it is
   more than the radius.
Outputs: <data>/<scene>/navigation/memory_map.npz + memory_map.json (objects in F) and
<runs>/navigation/qc/<scene>_memory_map.jpg.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from whatchanged.common import geometry as G
from whatchanged.common.io import log, read_json, save_npz, write_json
from whatchanged.navigation.common import RUNS, apply, cfg, inventory_sha, logit, memory_paths

EDGE_REL = 0.04
OBJ_MIN_G = 5
OBJ_MIN_MASS = 2.0  # summed Gaussian opacity in a cell for an inventory footprint (floaters: few, faint)
MIN_PTS = 2
BAND_FRAC = 0.1
FLOOR_LO, FLOOR_HI = -0.06, 0.20  # near-floor heights searched for the local walking surface (rugs, mats)
REGION_CELLS = 9  # 45 cm regions for the local floor mode
FLOOR_NEAR = 0.06  # residual window for the floor-noise statistics


def depth_edges(D, rel=EDGE_REL):
    e = np.zeros(D.shape, bool)
    for ax in (0, 1):
        d = np.abs(np.diff(D, axis=ax))
        m = np.maximum(np.take(D, range(D.shape[ax] - 1), axis=ax), 1e-3)
        j = (d / m) > rel
        if ax == 0:
            e[:-1] |= j
            e[1:] |= j
        else:
            e[:, :-1] |= j
            e[:, 1:] |= j
    return e | (D <= 0)


def backproject_cam(D, K, T, step):
    H, W = D.shape
    edge = depth_edges(D)
    v, u = np.mgrid[0:H:step, 0:W:step]
    z = D[::step, ::step]
    ok = ~edge[::step, ::step]
    u, v, z = u[ok], v[ok], z[ok].astype(np.float64)
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    return np.c_[x, y, z] @ T[:3, :3].T + T[:3, 3]


def rot_a_to_b(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def memory_reference_depths(scene: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference depths of the memory splat (cached); renders them in a GPU stage when the cache is missing."""
    from whatchanged.common.scene import Scene
    from whatchanged.splat.support import reference_depths

    sc = Scene(scene)
    if sc.ref_depth.exists():
        return reference_depths(sc, None)
    from whatchanged.common.gpu import gpu_stage
    from whatchanged.splat import gaussians as gs

    with gpu_stage("navigation_reference_depth", need_mib=3000):
        return reference_depths(sc, gs.to_torch(gs.load(sc.splat)))


def find_floor(P, cams, rng):
    rest = P.copy()
    cands = []
    for _ in range(6):
        if len(rest) < 1000:
            break
        n, d, inl = G.ransac_plane(rest, thr=0.01, iters=600, rng=rng)
        if n[2] < 0:
            n, d = -n, -d
        frac = inl.sum() / len(P)
        tilt = float(np.degrees(np.arccos(min(1.0, abs(n[2])))))
        above = float(((cams @ n + d) > 0).mean())
        cands.append(dict(n=n, d=d, frac=float(frac), tilt_deg=tilt, cams_above=above, height=float(-d)))
        rest = rest[~inl]
    ok = [c for c in cands if c["tilt_deg"] <= 15 and c["frac"] >= 0.03 and c["cams_above"] >= 0.95]
    if not ok:
        raise RuntimeError(f"no floor plane found: {cands}")
    h0 = min(c["height"] for c in ok)
    fl = max(
        [c for c in ok if c["height"] <= h0 + 0.05], key=lambda c: c["frac"]
    )  # lowest level; largest plane there (floor vs rug)
    return fl, cands


def build(scene: str) -> dict:
    C = cfg()
    g, rb = C["grid"], C["robot"]
    mem = memory_paths(scene)
    t0 = time.time()
    Kr, Tr, Dr = memory_reference_depths(scene)
    cams = Tr[:, :3, 3]
    rng = np.random.default_rng(0)
    # ---- floor
    sub = np.concatenate([backproject_cam(Dr[i], Kr, Tr[i], 8) for i in range(0, len(Tr), 2)])
    sub = sub[rng.choice(len(sub), min(len(sub), 300000), replace=False)]
    fl, cands = find_floor(sub, cams, rng)
    R = rot_a_to_b(fl["n"], np.array([0, 0, 1.0]))
    T_F_S = np.eye(4)
    T_F_S[:3, :3] = R
    zf = (sub @ R.T)[:, 2] + fl["d"]  # plane n.p + d = 0  ->  z' = n.p = -d
    T_F_S[2, 3] = fl["d"] - np.median(zf[np.abs(zf) < 0.05])
    band_hi = g["band_hi_m"]
    log(f"floor: {fl['frac']:.1%} of points, tilt {fl['tilt_deg']:.2f} deg vs memory up")
    # ---- grid extent from the evidence (robust) + margin
    PF = apply(T_F_S, sub)
    inb = (PF[:, 2] > -0.1) & (PF[:, 2] < g["z_max_m"])
    lo = np.percentile(PF[inb, :2], 0.5, axis=0) - g["margin_m"]
    hi = np.percentile(PF[inb, :2], 99.5, axis=0) + g["margin_m"]
    cell = g["cell_m"]
    nx, ny = int(np.ceil((hi[0] - lo[0]) / cell)), int(np.ceil((hi[1] - lo[1]) / cell))
    zc = 0.05
    nz = int(np.ceil((g["z_max_m"] + 0.1) / zc))
    log(f"grid {nx} x {ny} cells of {cell} m, origin {lo.round(2)}")
    TF = np.array([T_F_S @ T for T in Tr])
    # ---- local floor level (rugs / mats are the walking surface): per cell, the mode of the heights of near-floor
    # points (FLOOR_LO..FLOOR_HI) pooled over a REGION_CELLS x REGION_CELLS neighbourhood (a robust horizontal plane per
    # region)
    bw = 0.005
    nb = int(round((FLOOR_HI - FLOOR_LO) / bw))
    hist = np.zeros((nx * ny, nb), np.float32)
    for i in range(0, len(Tr), 2):
        Pi = backproject_cam(Dr[i], Kr, TF[i], g["depth_stride"] * 2)
        ix = np.floor((Pi[:, 0] - lo[0]) / cell).astype(int)
        iy = np.floor((Pi[:, 1] - lo[1]) / cell).astype(int)
        ib = np.floor((Pi[:, 2] - FLOOR_LO) / bw).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (ib >= 0) & (ib < nb)
        np.add.at(hist, (ix[ok] * ny + iy[ok], ib[ok]), 1)
    H3 = hist.reshape(nx, ny, nb)
    Hr = ndimage.uniform_filter(H3, size=(REGION_CELLS, REGION_CELLS, 3), mode="constant")
    ntot = Hr.sum(2)
    bc = (np.arange(nb) + 0.5) * bw + FLOOR_LO
    hf = np.where(ntot * REGION_CELLS * REGION_CELLS * 3 >= 60, bc[np.argmax(Hr, 2)], np.nan)
    if np.isnan(hf).all():
        hf[:] = 0.0
    idx = ndimage.distance_transform_edt(np.isnan(hf), return_distances=False, return_indices=True)
    hf = hf[tuple(idx)]  # regions without floor data: nearest region's level
    hf = ndimage.median_filter(hf, size=3).astype(np.float64)
    res = bc[None, :] - hf.reshape(-1)[:, None]
    near = np.abs(res) <= FLOOR_NEAR
    w = np.where(near, hist, 0).astype(np.float64)
    ar = np.abs(res)
    order = np.argsort(ar.ravel())
    cw = np.cumsum(w.ravel()[order])
    mad = float(ar.ravel()[order][np.searchsorted(cw, cw[-1] / 2)])
    sigma = 1.4826 * mad
    p98 = float(
        ar.ravel()[order][np.searchsorted(cw, cw[-1] * 0.98)]
    )  # heavy-tailed (rug pile, splat blur): use the 98th pct
    noise_lo = max(g["band_lo_min_m"], g["band_lo_sigma"] * sigma, p98)  # floor + splat noise
    band_lo = max(rb["step_height_m"], noise_lo)  # a legged robot steps over anything lower
    log(
        f"local floor level: p5/p50/p95 "
        f"{np.percentile(hf[hf != 0], [5, 50, 95]).round(3) if (hf != 0).any() else 0} m; "
        f"residual sigma {sigma * 100:.2f} cm, |res| p98 {p98 * 100:.2f} cm -> floor noise {noise_lo * 100:.1f} cm; "
        f"low clutter [{noise_lo * 100:.1f}, {band_lo * 100:.1f}] cm (traversable, costed); "
        f"obstacles [{band_lo * 100:.1f} cm, {band_hi:.2f} m]"
    )
    # ---- distinct-camera surface hits and ray pass-throughs per voxel (occupancy-grid inverse sensor model), z
    # relative to local floor
    cnt = np.zeros(nx * ny * nz, np.int32)
    pas = np.zeros(nx * ny * nz, np.int32)
    fcnt = np.zeros(nx * ny, np.int32)
    lcnt = np.zeros(nx * ny, np.int32)

    def vox(Pi):
        ix = np.floor((Pi[:, 0] - lo[0]) / cell).astype(int)
        iy = np.floor((Pi[:, 1] - lo[1]) / cell).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        zr = Pi[:, 2] - hf[np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)]
        iz = np.floor((zr + 0.1) / zc).astype(int)
        ok &= (iz >= 0) & (iz < nz)
        Pi[:, 2] = zr
        return ix, iy, iz, ok

    steps = np.arange(0.1, 7.0, zc * 0.8)
    for i in range(len(Tr)):
        Pi = backproject_cam(Dr[i], Kr, TF[i], g["depth_stride"])
        ix, iy, iz, ok = vox(Pi)
        floor = ok & (np.abs(Pi[:, 2]) <= noise_lo)
        lowc = ok & (Pi[:, 2] > noise_lo) & (Pi[:, 2] <= band_lo)
        band = ok & (Pi[:, 2] > band_lo) & (Pi[:, 2] < band_hi)
        # a camera sees an obstacle surface in a voxel only with >= MIN_PTS of its samples there and >= BAND_FRAC of its
        # samples in that cell's column lying in the band (floor-noise tails are ~1 % of a floor cell's samples)
        col_all = np.bincount(ix[ok] * ny + iy[ok], minlength=nx * ny)
        col_band = np.bincount(ix[band] * ny + iy[band], minlength=nx * ny)
        colok = col_band >= BAND_FRAC * np.maximum(col_all, 1)
        bsel = band & colok[np.clip(ix, 0, nx - 1) * ny + np.clip(iy, 0, ny - 1)]
        u_, c_ = np.unique((ix[bsel] * ny + iy[bsel]) * nz + iz[bsel], return_counts=True)
        cnt[u_[c_ >= MIN_PTS]] += 1
        fcnt[np.unique(ix[floor] * ny + iy[floor])] += 1
        u_, c_ = np.unique(ix[lowc] * ny + iy[lowc], return_counts=True)
        lcnt[u_[c_ >= MIN_PTS]] += 1
        # free space: samples along sparse rays from the camera up to 2 voxels before the rendered surface
        Ps = backproject_cam(Dr[i], Kr, TF[i], g["depth_stride"] * 3)
        o = TF[i][:3, 3]
        dv = Ps - o
        L = np.linalg.norm(dv, axis=1)
        dv /= L[:, None]
        t = steps[None, :]
        keep = t < (L[:, None] - 2 * zc)
        S = o + dv[:, None, :] * t[..., None]
        S = S[keep]
        jx, jy, jz, okk = vox(S)
        pas[np.unique(((jx * ny + jy) * nz + jz)[okk])] += 1
    cnt = cnt.reshape(nx, ny, nz)
    pas = pas.reshape(nx, ny, nz)
    fcnt = fcnt.reshape(nx, ny)
    lcnt = lcnt.reshape(nx, ny)
    zcent = (np.arange(nz) + 0.5) * zc - 0.1
    l_hit, l_miss = logit(C["belief"]["cell_hit"]), logit(C["belief"]["cell_miss"])
    lo_odds = cnt * l_hit + pas * l_miss
    band_vox = (cnt >= g["min_views"]) & (lo_odds > 0)
    occ_depth = band_vox.any(2)
    height = np.where(occ_depth, np.max(np.where(band_vox, zcent[None, None], -1), 2), 0.0).astype(np.float32)
    band_views = cnt.max(2)
    low_seen = lcnt >= g["min_views"]
    floor_seen = (fcnt >= g["min_views"]) | low_seen
    log(
        f"voxels: {int((cnt >= g['min_views']).sum())} seen surfaces, "
        f"{int(band_vox.sum())} occupied after free-space carving"
    )
    # ---- inventory layer (Gaussians)
    zs = np.load(mem["splat"])
    if Path(mem["inv"]).exists():
        inv = read_json(mem["inv"])["objects"]
        gid = np.load(mem["gid"])["gid"]
    else:  # inventory not built yet: geometry-only map (objects layer empty); rebuild later
        log(f"WARNING: no inventory at {mem['inv']}; building a geometry-only map")
        inv, gid = [], np.zeros(len(zs["opacities"]), np.int32)
    means = apply(T_F_S, zs["means"].astype(np.float64))
    opac = zs["opacities"].astype(np.float32)
    del zs
    good = opac >= g["gauss_min_opacity"]
    gx = np.floor((means[:, 0] - lo[0]) / cell).astype(int)
    gy = np.floor((means[:, 1] - lo[1]) / cell).astype(int)
    ing = good & (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    zrel = means[:, 2] - hf[np.clip(gx, 0, nx - 1), np.clip(gy, 0, ny - 1)]
    ing &= (zrel > band_lo) & (zrel < band_hi)
    static_ids = {int(o["id"][1:]) for o in inv if o["cls"] == "static"}
    is_static = np.isin(gid, list(static_ids)) | (gid == 0)
    stat_cnt = np.zeros((nx, ny), np.float32)
    m = ing & is_static
    np.add.at(stat_cnt, (gx[m], gy[m]), opac[m])
    objects, foot = [], {}
    best_cnt = np.zeros((nx, ny), np.float32)
    owner = np.full((nx, ny), -1, np.int16)  # -1 none, 0 static, k+1 = objects[k]
    for o in inv:
        k = int(o["id"][1:])
        sel = (gid == k) & good
        if sel.sum() < 30:
            continue
        Pk = means[sel]
        c, s, yaw = G.obb_zup(Pk, 2, 98)
        rec = dict(
            id=o["id"],
            name=o["name"],
            cls=o["cls"],
            name_conf=o.get("name_conf"),
            runner_up=o.get("runner_up"),
            box=dict(centre=np.round(c, 4).tolist(), size=np.round(s, 4).tolist(), yaw=round(float(yaw), 4)),
            n_gauss=int(sel.sum()),
            n_views=o.get("n_views"),
            colour=o.get("colour"),
            floor_frac=round(float((zrel[sel] <= band_lo).mean()), 3),
        )
        if o["cls"] == "static":  # static named parts (mantelpiece, fireplace, ...): boxes for language only, no layer
            rec["footprint_cells"] = 0
            objects.append(rec)
            continue
        mk = ing & (gid == k)
        cc = np.zeros((nx, ny), np.float32)
        np.add.at(cc, (gx[mk], gy[mk]), opac[mk])  # opacity mass: sparse / faint floaters do not make a footprint
        fp = cc >= OBJ_MIN_MASS
        rec["footprint_cells"] = int(fp.sum())
        j = len(objects)
        objects.append(rec)
        if fp.any():
            foot[j] = np.flatnonzero(fp.ravel()).astype(np.int32)
            upd = fp & (cc > best_cnt)
            best_cnt[upd] = cc[upd]
            owner[upd] = j + 1
    obj_any = owner > 0
    # static wins where background Gaussians dominate or where only depth evidence exists
    owner[(stat_cnt >= best_cnt) & occ_depth] = 0
    owner[occ_depth & (owner < 0)] = 0
    occ = occ_depth | obj_any
    free = floor_seen & ~occ
    unknown = ~free & ~occ
    blocked = occ | unknown
    dist = ndimage.distance_transform_edt(~blocked) * cell
    trav = free & (dist > rb["radius_m"])
    low = low_seen & free
    # CHECK (not an input): the memory walk itself must be traversable - standing memory cameras (> 1 m above the floor)
    # over the map
    camF = TF[:, :3, 3]
    ci = np.clip(np.floor((camF[:, 0] - lo[0]) / cell).astype(int), 0, nx - 1)
    cj = np.clip(np.floor((camF[:, 1] - lo[1]) / cell).astype(int), 0, ny - 1)
    stand_c = camF[:, 2] - hf[ci, cj] > 1.0
    walk_check = dict(
        n_standing_cams=int(stand_c.sum()),
        frac_on_free=round(float(free[ci, cj][stand_c].mean()), 3),
        frac_on_occupied=round(float(occ[ci, cj][stand_c].mean()), 3),
        frac_traversable=round(float(trav[ci, cj][stand_c].mean()), 3),
        clearance_m_p10_p50_p90=np.round(np.percentile(dist[ci, cj][stand_c], [10, 50, 90]), 3).tolist(),
        note="camera xy is about 0.2-0.4 m in front of the walker's body, "
        "so a share of cameras over furniture edges is expected",
    )
    log(f"walk check (memory cameras, not used as input): {walk_check}")
    out = mem["map_dir"]
    out.mkdir(parents=True, exist_ok=True)
    fk = np.zeros(0, np.int32) if not foot else np.concatenate([np.full(len(v), k, np.int32) for k, v in foot.items()])
    fc = np.zeros(0, np.int32) if not foot else np.concatenate(list(foot.values()))
    save_npz(
        out / "memory_map.npz",
        T_F_S=T_F_S,
        floor_level=hf.astype(np.float32),
        origin=lo,
        cell=cell,
        shape=np.array([nx, ny]),
        free=free,
        occ=occ,
        unknown=unknown,
        occ_depth=occ_depth,
        owner=owner,
        height=height,
        floor_views=fcnt.astype(np.int16),
        band_views=band_views.astype(np.int16),
        stat_cnt=stat_cnt,
        dist=dist.astype(np.float32),
        trav=trav,
        low=low,
        foot_obj=fk,
        foot_cell=fc,
        cams_F=TF[:, :3, 3],
    )
    info = dict(
        scene=scene,
        memory="base",
        inventory=str(mem["inv"]),
        inventory_sha=inventory_sha(scene) if inv else None,
        T_F_S=T_F_S.round(6).tolist(),
        floor=dict(
            frac_points=round(fl["frac"], 4),
            tilt_deg_vs_memory_up=round(fl["tilt_deg"], 3),
            offset_m=round(float(T_F_S[2, 3]), 4),
            sigma_m=round(sigma, 4),
            res_p98_m=round(p98, 4),
            candidates=[
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c_.items() if k not in ("n", "d")}
                for c_ in cands
            ],
        ),
        band_m=[round(band_lo, 4), band_hi],
        floor_noise_m=round(noise_lo, 4),
        step_height_m=rb["step_height_m"],
        walk_check=walk_check,
        area_low_clutter_m2=round(float(low.sum() * cell * cell), 2),
        grid=dict(origin=lo.round(4).tolist(), cell=cell, nx=nx, ny=ny),
        frac=dict(
            free=round(float(free.mean()), 4),
            occ=round(float(occ.mean()), 4),
            unknown=round(float(unknown.mean()), 4),
            traversable=round(float(trav.mean()), 4),
        ),
        area_m2=dict(
            free=round(float(free.sum() * cell * cell), 2), traversable=round(float(trav.sum() * cell * cell), 2)
        ),
        robot_radius_m=rb["radius_m"],
        n_cams=int(len(Tr)),
        objects=objects,
        wall_s=round(time.time() - t0, 1),
        rule=__doc__.split("\n\n")[3].strip()[:2000],
    )
    write_json(info, out / "memory_map.json")
    log(
        f"free {info['area_m2']['free']} m2, traversable {info['area_m2']['traversable']} m2, "
        f"{len(objects)} inventory objects, "
        f"{info['wall_s']} s"
    )
    qc(scene, info, out / "memory_map.npz")
    return info


# ------------------------------------------------------------------------------------------------ QC / drawing
def map_image(M, scale=8, trav=True):
    """top-down colour image of a map dict (x right, y up). Returns BGR image and a world->pixel function."""
    nx, ny = M["shape"]
    img = np.zeros((ny, nx, 3), np.uint8)
    img[:] = (60, 60, 60)
    img[M["unknown"].T] = (45, 40, 38)
    img[M["free"].T] = (205, 212, 214)
    if trav:
        img[M["trav"].T] = (232, 240, 238)
    img[M["occ"].T] = (95, 92, 90)
    if "owner" in M:
        img[(M["owner"] > 0).T] = (140, 120, 105)
    img = img[::-1]
    img = cv2.resize(img, (nx * scale, ny * scale), interpolation=cv2.INTER_NEAREST)
    o, c = np.asarray(M["origin"]), float(M["cell"])

    def w2p(xy):
        xy = np.atleast_2d(xy)
        u = (xy[:, 0] - o[0]) / c * scale
        v = (ny - (xy[:, 1] - o[1]) / c) * scale
        return np.c_[u, v]

    return img, w2p


def qc(scene, info, npz):
    M = dict(np.load(npz))
    img, w2p = map_image(M)
    cams = M["cams_F"]
    p = w2p(cams[:, :2]).astype(int)
    for a, b in zip(p[:-1], p[1:]):
        cv2.line(img, tuple(a), tuple(b), (200, 150, 60), 1, cv2.LINE_AA)
    for o in info["objects"]:
        if o["name"].startswith("object") or o["cls"] == "static":
            continue
        b = o["box"]
        cor = G.footprint(np.array(b["centre"]), np.array(b["size"]), b["yaw"])
        q = w2p(cor).astype(np.int32)
        col = (60, 140, 230) if o["cls"] == "furniture" else (80, 200, 120)
        cv2.polylines(img, [q], True, col, 1, cv2.LINE_AA)
        if o["cls"] == "furniture" or o["n_gauss"] > 5000:
            cv2.putText(
                img,
                o["name"][:22],
                tuple(q.mean(0).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
    leg = [
        ((232, 240, 238), f"traversable (centre, r={info['robot_radius_m']:.2f} m)"),
        ((205, 212, 214), "free floor (too close)"),
        ((95, 92, 90), "occupied (static)"),
        ((140, 120, 105), "occupied (inventory object)"),
        ((45, 40, 38), "unknown (not scanned)"),
        ((200, 150, 60), "memory camera path"),
    ]
    pad = np.full((img.shape[0], 260, 3), 255, np.uint8)
    for i, (c, t) in enumerate(leg):
        cv2.rectangle(pad, (10, 20 + 22 * i), (26, 34 + 22 * i), c, -1)
        cv2.putText(pad, t, (32, 32 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    s = info
    txt = [
        f"{scene} memory map",
        f"floor tilt {s['floor']['tilt_deg_vs_memory_up']} deg, sigma {s['floor']['sigma_m'] * 100:.1f} cm",
        f"band {s['band_m'][0] * 100:.1f} cm - {s['band_m'][1]} m",
        f"free {s['area_m2']['free']} m2, trav. {s['area_m2']['traversable']} m2",
        f"unknown {s['frac']['unknown']:.0%} of grid",
    ]
    for i, t in enumerate(txt):
        cv2.putText(pad, t, (10, 200 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    (RUNS / "qc").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(RUNS / "qc" / f"{scene}_memory_map.jpg"), np.concatenate([img, pad], 1), [cv2.IMWRITE_JPEG_QUALITY, 92]
    )


def load_map(scene: str, ensure_current: bool = False) -> tuple[dict, dict]:
    """The memory map (arrays, info); with ensure_current it is rebuilt first when it was made from another inventory
    (ids are re-numbered when the inventory is rebuilt, so a stale map would name the wrong objects)."""
    d = memory_paths(scene)["map_dir"]
    if ensure_current:
        cur = inventory_sha(scene)
        old = read_json(d / "memory_map.json").get("inventory_sha") if (d / "memory_map.json").exists() else None
        if old != cur:
            log(f"memory map was built from inventory {old}, current is {cur}: rebuilding")
            build(scene)
    return dict(np.load(d / "memory_map.npz")), read_json(d / "memory_map.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    a = ap.parse_args()
    build(a.scene)


if __name__ == "__main__":
    main()
