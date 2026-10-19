"""Tight evidence mask of a small object: the pixels whose rendered (memory) 3-D point lies inside the object's own
oriented box.

The inventory's rendered id masks can include neighbours (a "banana" group that also holds the plate it lies on) and
render ghosts of moved objects; averaging the checker's P(missing) over such a mask dilutes a real change. The object's
own box (from its Gaussians, memory-map objects) is tight. Margin 3 cm. The caller keeps only known pixels inside the
checker's border.
"""

from __future__ import annotations

import numpy as np

from whatchanged.navigation.common import apply


def box_mask(
    Dp: np.ndarray, K: np.ndarray, TF: np.ndarray, box: dict, margin: float = 0.03, step: int = 1
) -> np.ndarray:
    """Dp: memory z-depth (H, W); TF: camera-to-world in F; box dict(centre, size, yaw) in F. Returns bool (H, W)."""
    H, W = Dp.shape
    c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
    ctr = np.array(box["centre"], float)
    hs = np.array(box["size"], float) / 2 + margin
    # quick reject: project the box centre and only scan a window around it
    Tcw = np.linalg.inv(TF)
    pc = Tcw[:3, :3] @ ctr + Tcw[:3, 3]
    if pc[2] < 0.2:
        return np.zeros((H, W), bool)
    rad = float(np.linalg.norm(hs)) * K[0, 0] / pc[2] * 1.3 + 4
    cu, cv = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
    x0, x1 = int(max(0, cu - rad)), int(min(W, cu + rad))
    y0, y1 = int(max(0, cv - rad)), int(min(H, cv + rad))
    out = np.zeros((H, W), bool)
    if x1 <= x0 or y1 <= y0:
        return out
    v, u = np.mgrid[y0:y1, x0:x1]
    z = Dp[y0:y1, x0:x1].astype(np.float64)
    ok = z > 0
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    P = apply(TF, np.c_[x.ravel(), y.ravel(), z.ravel()]).reshape(z.shape + (3,))
    d = P - ctr
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])  # world -> box frame (yaw about z)
    loc = d @ R.T
    inside = (np.abs(loc[..., 0]) <= hs[0]) & (np.abs(loc[..., 1]) <= hs[1]) & (np.abs(loc[..., 2]) <= hs[2]) & ok
    out[y0:y1, x0:x1] = inside
    return out
