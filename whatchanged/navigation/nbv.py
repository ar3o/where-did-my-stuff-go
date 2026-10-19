"""Render-based visibility tables for next-best-view and search (memory splat; GPU, called inside the robot run's GPU
stage).

Candidates = free memory-map cells on a nbv_step grid (any clearance > planner.MIN_CLEAR_M; the planner later keeps only
those the robot can reach with its footprint). Eye = candidate (x, y) at the eye height.
  * target visibility (per target instance group): the camera looks at the group's centre; its Gaussian centres
    (<= 400 samples) are projected and counted visible when in the frame and not behind the splat's rendered depth
    (|dz| <= 3 cm + 3 %); vis = visible / samples, then divided by the best candidate's value (relative visibility: an
    object never shows its far side).
  * coverage (for the search when the target is missing): 8 headings x pitch 35 deg down; the rendered depth
    back-projected into map cells (height <= 1.6 m): which cells a look-around from that spot would see.
Render size 192 x 108 with the memory focal length scaled by 192 / W.
At run time new obstacles occlude by a 2.5-D ray test against the current obstacle cells (helpers.los_ok).
"""

from __future__ import annotations

import numpy as np

from whatchanged.navigation.common import apply, cfg, look_at

RW, RH = 192, 108


def candidates(M: dict, step: float) -> tuple[np.ndarray, np.ndarray]:
    from whatchanged.navigation.planner import MIN_CLEAR_M

    o, c = M["origin"], float(M["cell"])
    k = max(1, int(round(step / c)))
    ok = M["free"] & (M["dist"] > MIN_CLEAR_M)
    ij = np.argwhere(ok)
    ij = ij[(ij[:, 0] % k == 0) & (ij[:, 1] % k == 0)]
    return ij, o + (ij + 0.5) * c


def tables(tsp: dict, gs, M: dict, K_mem, Wm: int, groups: dict, sp_means, gid, eye_h: float | None = None) -> dict:
    """groups: {name: [inventory int ids]} -> dict(cand_ij, cand_xy, vis {name: dict(raw, rel, centre)},
    cover (n, nx * ny) bool)."""
    C = cfg()
    eye_h = C["robot"]["eye_height_m"] if eye_h is None else eye_h
    T_F_S = M["T_F_S"]
    T_S_F = np.linalg.inv(T_F_S)
    ij, xy = candidates(M, C["task"]["nbv_step_m"])
    K = K_mem.copy()
    K[:2] *= RW / Wm
    nx, ny = M["shape"]
    o, c = M["origin"], float(M["cell"])
    vis = {}
    for name, ids in groups.items():
        sel = np.flatnonzero(np.isin(gid, ids))
        if not len(sel):
            continue
        Ps = sp_means[sel[:: max(1, len(sel) // 400)]].astype(np.float64)
        ctr = apply(T_F_S, Ps).mean(0)
        v = np.zeros(len(xy))
        for n, p in enumerate(xy):
            eye = np.r_[p, eye_h]
            if np.linalg.norm(ctr[:2] - p) < 0.15:
                continue
            T = T_S_F @ look_at(eye, ctr)
            D = gs.render(tsp, K, T, RW, RH)["depth"]
            Tcw = np.linalg.inv(T)
            pc = apply(Tcw, Ps)
            z = pc[:, 2]
            u = (K[0, 0] * pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]).astype(int)
            w = (K[1, 1] * pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]).astype(int)
            m = (z > 0.2) & (u >= 0) & (w >= 0) & (u < RW) & (w < RH)
            ok = np.zeros(len(z), bool)
            j = np.flatnonzero(m)
            d = D[w[j], u[j]]
            ok[j] = (d <= 0) | (z[j] <= d + 0.03 + 0.03 * d)
            v[n] = ok.mean()
        vmax = max(float(v.max()), 1e-6)
        vis[name] = dict(raw=v, rel=v / vmax, centre=ctr)
    cover = np.zeros((len(xy), nx * ny), bool)
    fl = M["floor_level"]
    for n, p in enumerate(xy):
        eye = np.r_[p, eye_h]
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            tgt = eye + np.array([np.cos(a), np.sin(a), -np.tan(np.radians(35))])
            T = T_S_F @ look_at(eye, tgt)
            D = gs.render(tsp, K, T, RW, RH)["depth"]
            v_, u_ = np.mgrid[0:RH:2, 0:RW:2]
            z = D[::2, ::2]
            okp = z > 0
            x = (u_[okp] + 0.5 - K[0, 2]) / K[0, 0] * z[okp]
            y = (v_[okp] + 0.5 - K[1, 2]) / K[1, 1] * z[okp]
            PF = apply(T_F_S @ T, np.c_[x, y, z[okp]])
            ix = np.floor((PF[:, 0] - o[0]) / c).astype(int)
            iy = np.floor((PF[:, 1] - o[1]) / c).astype(int)
            g = (ix >= 0) & (iy >= 0) & (ix < nx) & (iy < ny)
            g &= PF[:, 2] - fl[np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)] < 1.6
            cover[n, ix[g] * ny + iy[g]] = True
    return dict(cand_ij=ij, cand_xy=xy, vis=vis, cover=cover)
