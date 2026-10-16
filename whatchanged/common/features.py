"""Mask pooling of DINOv2 PCA feature maps, and the appearance terms used to compare a region with an object."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from whatchanged.common.workers import run_worker


def pool(f: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """f (C, h, w) features, mask (H, W) bool -> unit embedding weighted by the mask's coverage of each patch."""
    C, h, w = f.shape
    wgt = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    if wgt.sum() < 0.5:
        return None
    e = (f.astype(np.float32) * wgt[None]).sum((1, 2))
    return e / max(np.linalg.norm(e), 1e-9)


def dims_term(a, b) -> float:
    """Mean |log ratio| of sorted box sides (2 cm floor): 0 for equal sizes, about 0.7 for a factor of two."""
    sa, sb = np.sort(a), np.sort(b)
    return float(np.abs(np.log(np.maximum(sa, 0.02) / np.maximum(sb, 0.02))).mean())


def encode(
    items: list[tuple[Path, Path]], pca: Path, fit_images: list[Path], job_path: Path, need_mib: int = 3000
) -> None:
    """DINOv2 PCA features for (image, output .npy) pairs; the PCA is fitted on fit_images when it does not exist."""
    queue = [dict(image=str(a), out=str(b)) for a, b in items if not Path(b).exists()]
    if not queue and pca.exists():
        return
    run_worker(
        "vision",
        "whatchanged.common.dino_worker",
        dict(items=queue, pca=str(pca), fit_images=[str(p) for p in fit_images]),
        job_path,
        need_mib,
    )
