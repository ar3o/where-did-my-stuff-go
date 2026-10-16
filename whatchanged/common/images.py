"""Image helpers: masked crops for image-text models, colour matching between renders and photos, overlap scores."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CROP_MARGIN = 0.25
CROP_DIM = 0.35  # brightness kept outside the mask, so the crop still shows context


def crop_masked(
    img: np.ndarray, mask: np.ndarray, out: str | Path, margin: float = CROP_MARGIN, min_px: int = 64
) -> bool:
    """Square crop around the mask with the outside dimmed; False if the mask is too small."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return False
    cx, cy = xs.mean(), ys.mean()
    s = max(np.ptp(xs), np.ptp(ys), min_px) * (1 + 2 * margin)
    x0, x1 = int(max(0, cx - s / 2)), int(min(img.shape[1], cx + s / 2))
    y0, y1 = int(max(0, cy - s / 2)), int(min(img.shape[0], cy + s / 2))
    c = img[y0:y1, x0:x1].astype(np.float32)
    inside = cv2.dilate(mask[y0:y1, x0:x1].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    c[~inside] *= CROP_DIM
    cv2.imwrite(str(out), c.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def colour_match(render: np.ndarray, photo: np.ndarray, known: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean / std match on known pixels: photo ~ a * render + b."""
    x, y = render[known].reshape(-1, 3), photo[known].reshape(-1, 3)
    a = y.std(0) / np.maximum(x.std(0), 1e-3)
    return a, y.mean(0) - a * x.mean(0)


def trimmed_gain(render: np.ndarray, photo: np.ndarray, known: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel y = a x + b on known pixels, refitted twice on the 70 % best-fitting pixels."""
    x = render[known].reshape(-1, 3).astype(np.float64)
    y = photo[known].reshape(-1, 3).astype(np.float64)
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


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def ochiai(intersection: float, area_a: float, area_b: float) -> float:
    """Intersection over the geometric mean of the areas, so a thin fragment inside a larger mask still scores high."""
    return float(intersection / np.sqrt(float(area_a) * float(area_b)))


def border_mask(H: int, W: int, px: int) -> np.ndarray:
    """True away from the image border."""
    m = np.zeros((H, W), bool)
    m[px:-px, px:-px] = True
    return m


def outline(mask: np.ndarray) -> list:
    """Simplified outer contours (pixel polygons) of a mask."""
    cn, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.approxPolyDP(c, 1.5, True)[:, 0].tolist() for c in cn if len(c) >= 3 and cv2.contourArea(c) >= 20]


def read_rgb(path: str | Path) -> np.ndarray:
    """RGB float image in [0, 1]."""
    return cv2.imread(str(path))[..., ::-1].astype(np.float32) / 255
