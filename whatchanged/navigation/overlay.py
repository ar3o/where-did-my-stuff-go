"""Drawing helpers for the robot film: label pills with leader lines and greedy non-overlapping placement, a glowing
path ribbon on the floor that is depth-tested so it passes behind furniture, and pinhole projection.

Colours follow the change film: red removed, amber moved, green added, grey not observed.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
TYPE_COL = {
    "removed": (235, 64, 52),
    "moved": (255, 184, 0),
    "added": (52, 199, 89),
    "unseen": (150, 150, 150),
    "target": (80, 220, 255),
}
CYAN = (70, 225, 255)


def overlap(a, b) -> float:
    """Intersection area of two (x0, y0, x1, y1) rectangles."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0)


class Labeler:
    """Label pills placed around their anchors, avoiding each other, the anchor and reserved areas."""

    def __init__(self, size: int = 17):
        self.font = ImageFont.truetype(FONT, size)

    def size(self, text):
        d = ImageDraw.Draw(Image.new("RGB", (4, 4)))
        return d.textlength(text, font=self.font) + 40, 30

    def place(self, anchor, text, taken, W, H, avoid=()):
        bw, bh = self.size(text)
        best, bc = None, 1e18
        for r in (70, 110, 160):
            for ang in range(0, 360, 45):
                a = math.radians(ang - 90)
                cx, cy = anchor[0] + r * math.cos(a) * 1.3, anchor[1] + r * math.sin(a)
                b = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
                out = max(0, 10 - b[0]) + max(0, 10 - b[1]) + max(0, b[2] - W + 10) + max(0, b[3] - H + 10)
                cost = 60 * out * bh + 40 * overlap(b, (anchor[0] - 25, anchor[1] - 25, anchor[0] + 25, anchor[1] + 25))
                cost += sum(25 * overlap(b, t) for t in taken) + sum(60 * overlap(b, t) for t in avoid) + 2.0 * r
                if cost < bc:
                    bc, best = cost, b
        return best

    def draw(self, d, anchor, b, text, col, alpha=1.0):
        a8 = int(255 * alpha)
        cy = (b[1] + b[3]) / 2
        ex, ey = float(np.clip(anchor[0], b[0], b[2])), float(np.clip(anchor[1], b[1], b[3]))
        d.line([(ex, ey), tuple(anchor)], fill=(255, 255, 255, int(190 * alpha)), width=2)
        d.ellipse(
            (anchor[0] - 4, anchor[1] - 4, anchor[0] + 4, anchor[1] + 4), fill=col + (a8,), outline=(255, 255, 255, a8)
        )
        d.rounded_rectangle(b, radius=15, fill=(16, 18, 22, int(180 * alpha)), outline=(255, 255, 255, int(60 * alpha)))
        d.ellipse((b[0] + 11, cy - 6, b[0] + 23, cy + 6), fill=col + (a8,))
        d.text((b[0] + 30, cy), text, fill=(255, 255, 255, a8), font=self.font, anchor="lm")


def proj(P: np.ndarray, Tcw: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points -> (pixel uv (n, 2), camera z) with the world-to-camera pose Tcw."""
    pc = P @ Tcw[:3, :3].T + Tcw[:3, 3]
    z = pc[:, 2]
    zz = np.maximum(z, 1e-6)
    return np.c_[K[0, 0] * pc[:, 0] / zz + K[0, 2], K[1, 1] * pc[:, 1] / zz + K[1, 2]], z


def dense(path, step: float = 0.02) -> np.ndarray:
    """Path resampled every `step` metres."""
    P = np.asarray(path, float)
    if len(P) < 2:
        return P
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * s for s in np.linspace(0, 1, n + 1)[1:]]
    return np.array(out)


def ribbon(img_shape, path_xy, floor_z, TF, K, Dscene, t_anim: float, width: float = 0.12):
    """Path ribbon 12 cm wide, 1 cm above the local floor, drawn as 2 cm quads depth-tested against Dscene, fading
    within 0.4 m of the camera, with moving chevrons. Returns (alpha mask float H x W, chevron mask)."""
    H, W = img_shape[:2]
    P = dense(path_xy)
    if len(P) < 2:
        return np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    Tcw = np.linalg.inv(TF)
    tang = np.gradient(P, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    nrm = np.c_[-tang[:, 1], tang[:, 0]]
    z = floor_z(P) + 0.01
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    A = np.zeros((H, W), np.float32)
    Ch = np.zeros((H, W), np.float32)
    s = Dscene.shape[1] / W
    cam = TF[:3, 3]
    for k in range(len(P) - 1):
        q = np.array(
            [
                np.r_[P[k] + nrm[k] * width / 2, z[k]],
                np.r_[P[k + 1] + nrm[k + 1] * width / 2, z[k + 1]],
                np.r_[P[k + 1] - nrm[k + 1] * width / 2, z[k + 1]],
                np.r_[P[k] - nrm[k] * width / 2, z[k]],
            ]
        )
        uv, zz = proj(q, Tcw, K)
        if (zz < 0.15).any():
            continue
        d_cam = float(np.linalg.norm(q.mean(0) - cam))
        fade = float(np.clip((d_cam - 0.4) / 0.4, 0, 1)) * float(np.clip(1.0 - (L[k] - 6.0) / 2.0, 0.2, 1))
        if fade <= 0:
            continue
        x0, y0 = np.floor(uv.min(0)).astype(int)
        x1, y1 = np.ceil(uv.max(0)).astype(int)
        if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H or (x1 - x0) > W or (y1 - y0) > H:
            continue
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1 + 1, W), min(y1 + 1, H)
        roi = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(roi, np.round(uv - [x0, y0]).astype(np.int32), 1, cv2.LINE_AA)
        D = (
            cv2.resize(Dscene, (W, H), interpolation=cv2.INTER_NEAREST)[y0:y1, x0:x1]
            if s != 1
            else Dscene[y0:y1, x0:x1]
        )
        zq = float(zz.mean())
        vis = (D <= 0) | (zq <= D + 0.04 + 0.03 * D)
        m = (roi > 0) & vis
        A[y0:y1, x0:x1] = np.maximum(A[y0:y1, x0:x1], m * fade)
        ph = ((L[k] - 0.6 * t_anim) % 0.35) / 0.35
        if ph < 0.35:
            Ch[y0:y1, x0:x1] = np.maximum(Ch[y0:y1, x0:x1], m * fade)
    return A, Ch
