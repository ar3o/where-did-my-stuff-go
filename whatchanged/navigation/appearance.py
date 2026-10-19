"""Appearance change scores inside an object's tight mask. They complement the learned checker, which can miss small
colour changes such as a yellow banana gone from a pale bowl. Between the memory render and the episode render at the
same pose:
  cd = mean |memory render x gain + bias - episode render| (0-1; per-channel robust gain fitted on known pixels),
  fd = 1 - cosine of the DINOv2 PCA-128 embeddings pooled over the mask (whatchanged.common.features.pool).
Their no-change baselines (p90 / p99) come from calib_nochange.py on the memory scan only.
"""

from __future__ import annotations

import numpy as np


def gain_fit(raw: np.ndarray, real: np.ndarray, known: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel gain and bias mapping raw onto real, refitted three times on the 70 % best-fitting known pixels."""
    x = raw[known].reshape(-1, 3).astype(np.float64)
    y = real[known].reshape(-1, 3).astype(np.float64)
    if len(x) < 1000:
        return np.ones(3), np.zeros(3)
    m = np.ones(len(x), bool)
    a, b = np.ones(3), np.zeros(3)
    for _ in range(3):
        for c in range(3):
            A = np.c_[x[m, c], np.ones(m.sum())]
            a[c], b[c] = np.linalg.lstsq(A, y[m, c], rcond=None)[0]
        r = np.abs(x * a + b - y).mean(1)
        m = r < np.percentile(r, 70)
    return np.clip(a, 0.5, 2.0), np.clip(b, -0.3, 0.3)


def scores(ren_u8, rgb_u8, fr, fq, known, m, gain=None) -> tuple[float, float, tuple]:
    """ren / rgb uint8 (H, W, 3); fr / fq (128, h, w) features of the memory / episode render; m bool mask.
    Returns (cd, fd, gain)."""
    from whatchanged.common.features import pool

    a, b = (
        gain if gain is not None else gain_fit(ren_u8.astype(np.float32) / 255, rgb_u8.astype(np.float32) / 255, known)
    )
    d = np.abs(ren_u8[m].astype(np.float64) / 255 * a + b - rgb_u8[m].astype(np.float64) / 255).mean()
    e1, e2 = pool(fr, m), pool(fq, m)
    fd = 1.0 - float(e1 @ e2) if e1 is not None and e2 is not None else 0.0
    return float(d), fd, (a, b)
