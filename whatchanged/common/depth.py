"""Monocular depth aligned to a rendered depth map, and depth-edge masks.

MoGe-3 depth is close to metric but has a per-image scale and a smooth shape bend. It is fitted to the splat's predicted
depth in inverse depth (IRLS with Tukey weights), then a coarse tile-wise median of the remaining inverse-depth residual
is subtracted. The tiles are 96 px, too coarse to absorb a real 5-50 cm object change.
"""

from __future__ import annotations

import cv2
import numpy as np

EDGE_PX = 4
MAX_ALIGN_REL_ERR = 0.03  # a photo is compared only if the aligned depth agrees with the prediction to 3 % (median)
TUKEY_C = 4.685
TILE_PX = 96


def depth_edges(D: np.ndarray, px: int = EDGE_PX) -> np.ndarray:
    """Pixels within px of a log-depth discontinuity or of missing depth."""
    L = np.log(np.maximum(D, 1e-3))
    gx = np.abs(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)) / 8
    gy = np.abs(cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)) / 8
    e = ((gx + gy) > 0.03) | (D <= 0)
    return cv2.dilate(e.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)) > 0


def align(D_pred: np.ndarray, D_mono: np.ndarray, ok: np.ndarray, tile: int = TILE_PX) -> tuple[np.ndarray, dict]:
    """Monocular depth mapped onto the predicted depth on pixels `ok`. Returns (aligned depth, fit info)."""
    if ok.sum() < 2000:
        return np.zeros_like(D_pred), dict(
            a=None, b=None, inlier_frac=0.0, rel_err_median=1.0, note="too few known pixels"
        )
    y = 1.0 / D_pred[ok]
    x = 1.0 / D_mono[ok]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(x), min(len(x), 60000), replace=False)
    xs, ys = x[sel], y[sel]
    w = np.ones(len(xs))
    a, b = 1.0, 0.0
    for _ in range(10):
        A = np.c_[xs, np.ones(len(xs))] * np.sqrt(w)[:, None]
        a, b = np.linalg.lstsq(A, ys * np.sqrt(w), rcond=None)[0]
        r = ys - (a * xs + b)
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        u = r / (TUKEY_C * s)
        w = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0)
    inv = a / np.maximum(D_mono, 1e-3) + b
    info = dict(a=float(a), b=float(b), inlier_frac=float((w > 0).mean()))
    H, W = D_pred.shape
    res = np.where(ok, 1.0 / np.maximum(D_pred, 1e-3) - inv, np.nan)
    gh, gw = int(np.ceil(H / tile)), int(np.ceil(W / tile))
    grid = np.full((gh, gw), np.nan, np.float32)
    for i in range(gh):
        for j in range(gw):
            t = res[i * tile : (i + 1) * tile, j * tile : (j + 1) * tile]
            t = t[np.isfinite(t)]
            if len(t) > 0.2 * tile * tile:
                grid[i, j] = np.median(t)
    if np.isfinite(grid).any():
        g = np.where(np.isfinite(grid), grid, np.nanmedian(grid)).astype(np.float32)
        g = cv2.GaussianBlur(g, (3, 3), 0.8)
        inv = inv + cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        info["lowfreq_abs_median"] = float(np.median(np.abs(grid[np.isfinite(grid)])))
    D = np.where((D_mono > 0) & (inv > 1e-3), 1.0 / np.maximum(inv, 1e-3), 0).astype(np.float32)
    rel = np.abs(D[ok] - D_pred[ok]) / D_pred[ok]
    info["rel_err_median"] = float(np.median(rel)) if len(rel) else 1.0
    return D, info


def align_to_render(D_pred: np.ndarray, D_mono: np.ndarray, known: np.ndarray) -> tuple[np.ndarray, dict]:
    """align() on known pixels with valid depth away from depth edges."""
    ok = known & (D_pred > 0) & (D_mono > 0) & ~depth_edges(D_pred)
    return align(D_pred, D_mono, ok)


def load_mono(path, size: tuple[int, int]) -> np.ndarray:
    """MoGe-3 depth file resized (nearest) to (W, H)."""
    return cv2.resize(np.load(path)["depth"].astype(np.float32), size, interpolation=cv2.INTER_NEAREST)
