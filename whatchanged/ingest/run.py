"""Ingest a scene: reference walk video -> work frames (+ 1920 px copies), query photos or a query video -> photos,
photo groups and the object name list.

    python -m whatchanged.ingest.run --scene config1 --reference ref.MOV --queries photos/*.HEIC \\
        --groups groups.json --names names.json

A group is a set of photos of one state of the room; each group is compared with the reference on its own. Without
--groups every query forms one group. A query video becomes `query_video_frames` sharp frames in one group.
"""

from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from whatchanged import config
from whatchanged.common.io import log, read_json, write_json
from whatchanged.common.scene import Scene
from whatchanged.ingest import photos as photolib
from whatchanged.ingest import video as videolib

VIDEO_SUFFIXES = (".mov", ".mp4", ".m4v")


@dataclass
class IngestConfig:
    fps: float = 10.0
    long_side: int = 960
    hires_long_side: int = 1920
    n_keyframes: int = 64
    n_extra: int = 220
    test_every: int = 8
    blur_frac: float = 0.35
    jpeg_quality: int = 95
    query_video_frames: int = 3


def ingest_reference(scene: Scene, video: Path, cfg: IngestConfig) -> None:
    info = videolib.probe(video)
    W, H = videolib.work_size(info, cfg.long_side)
    t0 = time.time()
    buffers, scores, stds, vf = videolib.decode(video, cfg.fps, W, H, info.transfer, cfg.jpeg_quality)
    times = np.arange(len(buffers)) / cfg.fps
    rows, n_ok = videolib.select_frames(
        times, scores, stds, cfg.n_keyframes, cfg.n_extra, cfg.fps, cfg.blur_frac, cfg.test_every
    )
    scene.frames.mkdir(parents=True, exist_ok=True)
    meta = []
    for n, (i, kf, split) in enumerate(rows):
        name = f"{n:04d}"
        (scene.frames / f"{name}.jpg").write_bytes(buffers[i].tobytes())
        meta.append(
            dict(name=name, t=round(float(times[i]), 4), kf=kf, split=split, sharpness=round(float(scores[i]), 1))
        )
    Wh, Hh = videolib.work_size(info, cfg.hires_long_side)
    videolib.extract_hires(video, {m["name"]: m["t"] for m in meta}, scene.frames_hi, info.transfer, Wh, Hh)
    write_json(
        dict(
            video=str(video),
            size=[W, H],
            hires_size=[Wh, Hh],
            fps=cfg.fps,
            filter=vf,
            color_transfer=info.transfer,
            n_decoded=len(buffers),
            n_ok=n_ok,
            frames=meta,
        ),
        scene.frames_json,
    )
    n_test = sum(m["split"] == "test" for m in meta)
    log(
        f"reference: {len(meta)} frames ({cfg.n_keyframes} keyframes, {n_test} held out) from {len(buffers)} decoded "
        f"in {time.time() - t0:.0f} s"
    )


def query_video_frames(scene: Scene, video: Path, cfg: IngestConfig) -> list[str]:
    """Sharpest frames of a query video (one per equal time bin), stored like photos."""
    info = videolib.probe(video)
    W, H = videolib.work_size(info, cfg.long_side)
    buffers, scores, stds, _ = videolib.decode(video, cfg.fps, W, H, info.transfer, cfg.jpeg_quality)
    times = np.arange(len(buffers)) / cfg.fps
    ok = (scores >= cfg.blur_frac * np.median(scores)) & (stds >= videolib.MIN_GREY_STD)
    pick = videolib.sharpest_per_bin(times, scores, ok, cfg.query_video_frames)
    names = []
    scene.photos.mkdir(parents=True, exist_ok=True)
    for i in pick:
        name = f"{video.stem}_t{times[i]:.1f}"
        (scene.photos / f"{name}.jpg").write_bytes(buffers[i].tobytes())
        names.append(name)
    Wh, Hh = videolib.work_size(info, cfg.hires_long_side)
    videolib.extract_hires(video, {n: times[i] for n, i in zip(names, pick)}, scene.photos_hi, info.transfer, Wh, Hh)
    return names


def read_groups(path: Path | None, names: list[str]) -> dict[str, list[str]]:
    """{"groups": {"0": [photo, ...]}} or {"groups": [{"group": 0, "photos": [...]}]}; file suffixes are dropped."""
    if path is None:
        return {"0": names}
    raw = read_json(path)["groups"]
    if isinstance(raw, list):
        raw = {str(g["group"]): g["photos"] for g in raw}
    return {str(k): [Path(p).stem for p in v] for k, v in raw.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument(
        "--reference", required=True, type=Path, help="walk-around video of the room in its reference state"
    )
    ap.add_argument("--queries", nargs="+", required=True, type=Path, help="photos, or one video, taken later")
    ap.add_argument("--groups", type=Path, help="JSON with the photos of each room state")
    ap.add_argument("--names", required=True, type=Path, help="JSON name list: furniture, items, static")
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()
    cfg = config.load(IngestConfig, "ingest", args.config)
    scene = Scene(args.scene)
    scene.root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.names, scene.names_json)
    if not scene.frames_json.exists():
        ingest_reference(scene, args.reference, cfg)
    names, rows = [], []
    for q in args.queries:
        if q.suffix.lower() in VIDEO_SUFFIXES:
            names += query_video_frames(scene, q, cfg)
            continue
        rows.append(
            photolib.ingest_photo(
                q, scene.photos, scene.photos_hi, cfg.long_side, cfg.hires_long_side, cfg.jpeg_quality
            )
        )
        names.append(q.stem)
    write_json(rows, scene.photos_json)
    write_json(dict(groups=read_groups(args.groups, names)), scene.groups_json)
    sizes = {tuple(cv2.imread(str(scene.photo_image(n))).shape[1::-1]) for n in names}
    log(f"queries: {len(names)} photos, sizes {sorted(sizes)}")


if __name__ == "__main__":
    main()
