"""Path and line-of-sight helpers shared by the belief, the robot loop and the drawing code."""

from __future__ import annotations

import numpy as np

from whatchanged.navigation import planner as PL


def remaining(path, rxy) -> list:
    """The part of a path still ahead of the robot: its position, then the path from the segment nearest to it on."""
    P = np.asarray(path, float)
    if len(P) < 2:
        return P.tolist()
    best, bk = 1e9, 0
    for k in range(len(P) - 1):
        a, b = P[k], P[k + 1]
        ab = b - a
        s = float(np.clip((rxy - a) @ ab / max(ab @ ab, 1e-9), 0, 1))
        q = a + s * ab
        d = np.linalg.norm(q - rxy)
        if d < best:
            best, bk = d, k
    return [list(np.round(rxy, 3))] + P[bk + 1 :].round(3).tolist()


def nearest_blob(p, nb: list[dict]) -> dict | None:
    if p is None or not nb:
        return None
    return min(nb, key=lambda b: np.linalg.norm(np.array(b["centre"][:2]) - p))


def side(old, new, where) -> str:
    """Which side the new path passes the blocking point, relative to the direction of travel."""
    if where is None:
        return "around"
    N = np.asarray(new, float)
    if len(N) < 2:
        return "around"
    dense = np.concatenate(
        [np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / 0.05))) for a, b in zip(N[:-1], N[1:])]
    )
    k = int(np.argmin(np.linalg.norm(dense - where, axis=1)))
    tan = dense[min(k + 1, len(dense) - 1)] - dense[max(k - 1, 0)]
    off = dense[k] - where
    cr = tan[0] * off[1] - tan[1] * off[0]
    return "left" if cr > 0 else "right"


def los_ok(cand_xy, tgt, new_occ, Hd, M: dict, eye_h: float) -> np.ndarray:
    """2.5-D occlusion by new obstacles: the eye-to-target ray must pass above every new-obstacle cell it crosses.
    Returns 1 per candidate with a clear line of sight, else 0."""
    if not new_occ.any():
        return np.ones(len(cand_xy))
    o, c = M["origin"], float(M["cell"])
    out = np.ones(len(cand_xy))
    for n, p in enumerate(cand_xy):
        e = np.r_[p, eye_h]
        for s in np.linspace(0.05, 0.95, 30):
            q = e + s * (tgt - e)
            i, j = PL.to_cell(q[:2], o, c, M["shape"])
            if new_occ[i, j] and q[2] < Hd[i, j] + 0.02:
                out[n] = 0
                break
    return out
