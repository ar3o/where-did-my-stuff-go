"""Label style for films and stills: a rounded dark pill with white text and a coloured dot for the change type, a thin
leader line to the object, placed greedily where it covers the fewest object boxes and other labels."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TYPE_COL = {"removed": (235, 64, 52), "moved": (255, 184, 0), "added": (52, 199, 89)}


def default_font() -> str:
    """DejaVu Sans Bold as shipped with matplotlib."""
    import matplotlib

    return str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf")


def label_text(e: dict) -> str:
    name = e["name"]
    unsure = "(uncertain)" in name
    return f"{name.replace(' (uncertain)', '')}{'?' if unsure else ''} · change {100 * e['change_conf']:.0f}%"


def overlap(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0)


class Labeler:
    def __init__(self, font: str = "", size: int = 20):
        font = font or default_font()
        self.font = ImageFont.truetype(font, size)
        self.font_path = font

    def box(self, text: str) -> tuple[float, float]:
        tw = ImageDraw.Draw(Image.new("RGB", (4, 4))).textlength(text, font=self.font)
        return tw + 46, 34

    def place(self, anchor, text: str, obstacles, taken, W: int, H: int):
        """Candidate centres on three rings around the anchor; cost = overlap with obstacle boxes, other labels and the
        frame edge, plus distance, with a small penalty for the left side."""
        bw, bh = self.box(text)
        best, best_cost = None, 1e18
        for r in (95, 150, 210):
            for ang in range(0, 360, 45):
                a = math.radians(ang - 90)
                cx, cy = anchor[0] + r * math.cos(a) * 1.3, anchor[1] + r * math.sin(a)
                b = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
                out = max(0, 12 - b[0]) + max(0, 12 - b[1]) + max(0, b[2] - W + 12) + max(0, b[3] - H + 12)
                cost = 50 * out * bh
                cost += 40 * overlap(b, (anchor[0] - 30, anchor[1] - 30, anchor[0] + 30, anchor[1] + 30))
                for o in obstacles:
                    cost += overlap(b, o)
                for t in taken:
                    cost += 20 * overlap(b, t)
                cost += 2.0 * r + (30 if ang in (135, 180, 225) else 0)
                if cost < best_cost:
                    best_cost, best = cost, b
        return best

    def draw(self, img: np.ndarray, anchor, b, text: str, typ: str, alpha: float = 1.0) -> np.ndarray:
        if alpha <= 0.01:
            return img
        base = Image.fromarray(img).convert("RGBA")
        ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        col = TYPE_COL[typ]
        a8 = int(255 * alpha)
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        ex, ey = float(np.clip(anchor[0], b[0], b[2])), float(np.clip(anchor[1], b[1], b[3]))
        if abs(ex - cx) < 1e-3 and abs(ey - cy) < 1e-3:
            ex, ey = cx, b[3]
        d.line([(ex, ey), (anchor[0], anchor[1])], fill=(255, 255, 255, int(200 * alpha)), width=2)
        d.ellipse(
            (anchor[0] - 4, anchor[1] - 4, anchor[0] + 4, anchor[1] + 4), fill=col + (a8,), outline=(255, 255, 255, a8)
        )
        d.rounded_rectangle(b, radius=17, fill=(16, 18, 22, int(175 * alpha)), outline=(255, 255, 255, int(60 * alpha)))
        d.ellipse((b[0] + 13, cy - 6, b[0] + 25, cy + 6), fill=col + (a8,))
        d.text((b[0] + 33, cy), text, fill=(255, 255, 255, a8), font=self.font, anchor="lm")
        return np.asarray(Image.alpha_composite(base, ov).convert("RGB"))


def tag(img: np.ndarray, text: str, font) -> np.ndarray:
    """Small corner caption."""
    base = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    tw = d.textlength(text, font=font)
    d.rounded_rectangle((12, 12, 12 + tw + 24, 40), radius=14, fill=(16, 18, 22, 150))
    d.text((24, 26), text, fill=(235, 235, 235, 230), font=font, anchor="lm")
    return np.asarray(Image.alpha_composite(base, ov).convert("RGB"))
