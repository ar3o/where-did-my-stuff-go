"""Small I/O helpers: logging, atomic JSON / npz writes, packed boolean masks and SAM 3 detection files."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np


def log(*parts) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def write_json(obj, path: str | Path, indent: int | None = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=indent, default=_json_default))
    os.replace(tmp, path)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def save_npz(path: str | Path, compressed: bool = False, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")
    (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
    os.replace(tmp, path)


def pack_masks(masks) -> tuple[np.ndarray, np.ndarray]:
    """(m, H, W) bool -> (packed bits, shape [m, H, W])."""
    masks = np.asarray(masks, bool)
    return np.packbits(masks) if len(masks) else np.zeros(0, np.uint8), np.array(masks.shape)


def unpack_masks(bits: np.ndarray, shape) -> np.ndarray:
    m, h, w = (int(v) for v in shape)
    if m == 0:
        return np.zeros((0, h, w), bool)
    return np.unpackbits(bits)[: m * h * w].reshape(m, h, w).astype(bool)


def load_detections(path: str | Path) -> list[tuple[str, float, np.ndarray]]:
    """SAM 3 worker output -> [(phrase, score, mask)]; an absent file means no detections."""
    if not Path(path).exists():
        return []
    z = np.load(path)
    masks = unpack_masks(z["masks"], z["shape"])
    prompts = [str(p) for p in z["prompts"]]
    return [(prompts[int(z["prompt"][i])], float(z["score"][i]), masks[i]) for i in range(len(masks))]
