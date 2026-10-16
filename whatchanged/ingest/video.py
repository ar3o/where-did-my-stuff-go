"""Reference video -> sharp work frames, tone-mapped from iPhone HDR (HLG or PQ) to SDR bt709.

The clip is decoded at a fixed rate (ffmpeg applies the rotation metadata) and scaled to a long side of 960 px. Frames
are scored by Laplacian variance. Keyframes are the sharpest acceptable frame of each of `n_keyframes` equal time bins;
extra frames are the sharpest of `n_extra` bins that lie at least 0.15 s from every keyframe. A frame is acceptable when
its sharpness is at least 0.35 x the median and it is not near-uniform (grey std >= 12, e.g. a hand over the lens).
Every 8th extra frame is held out to measure the splat on views it never trained on.
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HDR_TRANSFERS = ("arib-std-b67", "smpte2084")
MIN_GREY_STD = 12


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float
    transfer: str


def probe(video: str | Path) -> VideoInfo:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json", str(video)]
    )
    j = json.loads(out)
    st = j["streams"][0]
    w, h = st["width"], st["height"]
    rot = 0
    for side in st.get("side_data_list", []) or []:
        if "rotation" in side:
            rot = int(side["rotation"])
    rot = int(st.get("tags", {}).get("rotate", rot))
    if abs(rot) % 180 == 90:
        w, h = h, w
    return VideoInfo(w, h, float(j["format"]["duration"]), st.get("color_transfer", ""))


def tone_chain(transfer: str, W: int, H: int) -> str:
    """ffmpeg filter: resize to W x H and, for HDR input, tone-map to SDR bt709 (linearise at 100 nits, bt709 primaries,
    Hable curve, limited range). Without it iPhone HLG frames come out grey next to the SDR photos of the same room."""
    if transfer in HDR_TRANSFERS:
        return (
            f"zscale=w={W}:h={H}:tin={transfer}:min=bt2020nc:pin=bt2020:t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
            f"tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv"
        )
    return f"scale={W}:{H}:flags=area"


def work_size(info: VideoInfo, long_side: int) -> tuple[int, int]:
    s = long_side / max(info.width, info.height)
    return int(round(info.width * s / 2) * 2), int(round(info.height * s / 2) * 2)


def sharpness(bgr: np.ndarray) -> tuple[float, float]:
    """(Laplacian variance, grey std) at 480 px width."""
    g = cv2.cvtColor(cv2.resize(bgr, (480, int(480 * bgr.shape[0] / bgr.shape[1]))), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var()), float(g.std())


def decode(video: str | Path, fps: float, W: int, H: int, transfer: str, jpeg_quality: int):
    """All frames at `fps` as JPEG buffers, with their sharpness and grey std."""
    vf = f"fps={fps},{tone_chain(transfer, W, H)}"
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    buffers, scores, stds = [], [], []
    nbytes = W * H * 3
    while True:
        b = p.stdout.read(nbytes)
        if len(b) < nbytes:
            break
        f = np.frombuffer(b, np.uint8).reshape(H, W, 3)
        sc, sd = sharpness(f)
        scores.append(sc)
        stds.append(sd)
        buffers.append(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1])
    p.wait()
    return buffers, np.array(scores), np.array(stds), vf


def sharpest_per_bin(times: np.ndarray, scores: np.ndarray, ok: np.ndarray, n: int) -> list[int]:
    bins = np.linspace(times[0], times[-1] + 1e-6, n + 1)
    pick = []
    for i in range(n):
        idx = np.where((times >= bins[i]) & (times < bins[i + 1]) & ok)[0]
        if len(idx):
            pick.append(int(idx[np.argmax(scores[idx])]))
    return pick


def select_frames(times, scores, stds, n_keyframes: int, n_extra: int, fps: float, blur_frac: float, test_every: int):
    """[(index, keyframe number or -1, split)] in time order."""
    ok = (scores >= blur_frac * np.median(scores)) & (stds >= MIN_GREY_STD)
    keys = sharpest_per_bin(times, scores, ok, n_keyframes)
    kt = times[keys]
    far = np.min(np.abs(times[:, None] - kt[None]), 1) >= max(0.07, 1.5 / fps)
    extras = sharpest_per_bin(times, scores, ok & far, n_extra)
    rows = [(i, k, "train") for k, i in enumerate(keys)]
    for rank, i in enumerate(extras):
        rows.append((i, -1, "test" if rank % test_every == test_every // 2 else "train"))
    return sorted(rows, key=lambda r: times[r[0]]), int(ok.sum())


def extract_hires(video: str | Path, times: dict[str, float], out_dir: Path, transfer: str, W: int, H: int) -> None:
    """One frame per name at the given time, tone-mapped at W x H (seeking per frame, 8 in parallel)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = tone_chain(transfer, W, H)

    def one(item):
        name, t = item
        out = out_dir / f"{name}.jpg"
        if not out.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    f"{t:.3f}",
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    "-vf",
                    vf,
                    "-q:v",
                    "2",
                    str(out),
                ],
                check=True,
            )

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, sorted(times.items())))
