"""SAM 3 detections with the scene's name list, on a full work-size image plus a 2 x 2 grid of tiles cut from the
1920 px copy, so small items get enough pixels. Tile masks land directly on their window of the work grid, so they
merge with the full-image detections without resampling. Within one image, two detections of the same phrase with mask
IoU > 0.5 keep the higher score.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from whatchanged.common.io import load_detections

TILE_FRAC = 0.55  # tile side as a fraction of the image side: 2 x 2 tiles with 10 % overlap
DUPLICATE_IOU = 0.5


def tile_windows(W: int, H: int) -> list[tuple[int, int, int, int]]:
    """(x, y, w, h) windows of a W x H grid."""
    tw, th = int(round(TILE_FRAC * W)), int(round(TILE_FRAC * H))
    return [(x, y, tw, th) for y in (0, H - th) for x in (0, W - tw)]


def jobs_for_image(image: Path, hires: Path, out_stem: Path, tile_dir: Path, W: int, H: int) -> list[dict]:
    """SAM 3 job items for one image: the work-size image and the four tiles of its high-resolution copy."""
    items = [dict(image=str(image), out=f"{out_stem}.npz", mask_wh=[W, H])]
    if not hires.exists():
        return items
    im = None
    for ti, (x, y, tw, th) in enumerate(tile_windows(W, H)):
        tp = tile_dir / f"{out_stem.name}_t{ti}.jpg"
        if not tp.exists():
            if im is None:
                im = cv2.imread(str(hires))
                tile_dir.mkdir(parents=True, exist_ok=True)
            sx, sy = im.shape[1] / W, im.shape[0] / H
            crop = im[int(y * sy) : int((y + th) * sy), int(x * sx) : int((x + tw) * sx)]
            cv2.imwrite(str(tp), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        items.append(dict(image=str(tp), out=f"{out_stem}_t{ti}.npz", mask_wh=[tw, th], window=[x, y, tw, th]))
    return items


def merged(
    out_stem: Path, W: int, H: int, phrases: set[str] | None = None, min_area: int = 1, tiles: bool = True
) -> list[tuple[str, float, np.ndarray]]:
    """Full-image and tile detections of one image on the W x H grid, by descending score, duplicates removed."""
    dets = [d for d in load_detections(f"{out_stem}.npz")]
    if tiles:
        for ti, (x, y, tw, th) in enumerate(tile_windows(W, H)):
            for phrase, score, m in load_detections(f"{out_stem}_t{ti}.npz"):
                full = np.zeros((H, W), bool)
                full[y : y + th, x : x + tw] = m[:th, :tw]
                dets.append((phrase, score, full))
    if phrases is not None:
        dets = [d for d in dets if d[0] in phrases]
    keep = []
    for phrase, score, m in sorted(dets, key=lambda d: -d[1]):
        a = int(m.sum())
        if a < min_area:
            continue
        if any(q == phrase and (m & mm).sum() / (a + mm.sum() - (m & mm).sum()) > DUPLICATE_IOU for q, _, mm in keep):
            continue
        keep.append((phrase, score, m))
    return keep
