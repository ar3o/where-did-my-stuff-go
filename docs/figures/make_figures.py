"""README banner and badges as light and dark SVGs, text converted to paths so they render the same everywhere.

    MONTSERRAT_DIR=<dir with Montserrat-{Regular,Medium,SemiBold}.ttf> python docs/figures/make_figures.py

Colour tokens and type match the companion robot-learning repository.
"""

from __future__ import annotations

import os
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

FONT_DIR = Path(os.environ.get("MONTSERRAT_DIR", "fonts"))
OUT = Path(__file__).resolve().parents[2] / "assets"
THEMES = {
    "light": dict(
        GREY="#5f5859", LINE="#d9d4d2", REDB="#8c2a45", WHITE="#1c1819", BG="#fff", BOXF="#faf8f8", REDTXT="#8c2a45"
    ),
    "dark": dict(
        GREY="#938b8a", LINE="#2b2728", REDB="#a8324f", WHITE="#d9d3d0", BG="#0d1117", BOXF="#10151c", REDTXT="#8a3550"
    ),
}
_fonts = {}


def text_path(text: str, weight: str, size: float, tracking: float = 0.0) -> tuple[str, float]:
    path = FONT_DIR / f"Montserrat-{weight}.ttf"
    if path not in _fonts:
        _fonts[path] = TTFont(str(path))
    f = _fonts[path]
    glyphs, cmap, upm = f.getGlyphSet(), f.getBestCmap(), f["head"].unitsPerEm
    s, x, parts = size / upm, 0.0, []
    for i, ch in enumerate(text):
        g = cmap.get(ord(ch))
        if g is None:
            continue
        pen = SVGPathPen(glyphs)
        glyphs[g].draw(TransformPen(pen, (s, 0, 0, -s, x, 0)))
        parts.append(pen.getCommands())
        x += glyphs[g].width * s + (tracking * size if i < len(text) - 1 else 0)
    return " ".join(parts), x


def T(text, size, x, y, fill, tracking=0.0, weight="Medium", anchor="s") -> str:
    d, w = text_path(text, weight, size, tracking)
    if anchor == "m":
        x -= w / 2
    return f'<g transform="translate({x:.1f} {y:.1f})"><path d="{d}" fill="{fill}"/></g>'


def svg(W, H, body, bg) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
        f'<rect width="{W}" height="{H}" fill="{bg}"/>{body}</svg>'
    )


def banner(c: dict) -> str:
    """Title in tracked capitals, the last words in the accent colour."""
    size, tracking, y = 30, 0.3, 58.5
    parts = [("WHERE DID MY ", c["GREY"], "Medium"), ("STUFF GO", c["REDTXT"], "SemiBold")]
    widths = [text_path(t, w, size, tracking)[1] + tracking * size for t, _, w in parts]
    W = 1211
    x = (W - sum(widths) + tracking * size) / 2
    body = []
    for (t, fill, w), tw in zip(parts, widths):
        body.append(T(t, size, x, y, fill, tracking, w))
        x += tw
    return svg(W, 96, "".join(body), c["BG"])


BADGES = [
    ("sam3.svg", "SAM 3", "Meta", "#0466c8"),
    ("pytorch.svg", "PyTorch", "", "#EE4C2C"),
    ("gsplat.svg", "gsplat", "1.5.3", "#6e7781"),
    ("colmap.svg", "COLMAP", "SfM", "#6e7781"),
    ("dinov2.svg", "DINOv2", "Meta", "#0466c8"),
    ("siglip2.svg", "SigLIP 2", "Google", "#1a73e8"),
    ("license.svg", "License", "MIT", "#6e7781"),
]


def badge(label: str, value: str, color: str, H: int = 28, fs: float = 12.5) -> str:
    pad = 11
    dl, wl = text_path(label, "SemiBold", fs, 0.04)
    left = pad + wl + pad
    parts = [
        f'<rect width="{left:.1f}" height="{H}" fill="#24292f"/>',
        f'<g transform="translate({pad} {H / 2 + fs * 0.36:.1f})"><path d="{dl}" fill="#ffffff"/></g>',
    ]
    W = left
    if value:
        dv, wv = text_path(value, "SemiBold", fs, 0.04)
        W = left + pad + wv + pad
        parts.append(f'<rect x="{left:.1f}" width="{W - left:.1f}" height="{H}" fill="{color}"/>')
        parts.append(
            f'<g transform="translate({left + pad:.1f} {H / 2 + fs * 0.36:.1f})"><path d="{dv}" fill="#ffffff"/></g>'
        )
    elif color:
        parts[0] = f'<rect width="{left:.1f}" height="{H}" fill="{color}"/>'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H}" viewBox="0 0 {W:.1f} {H}" role="img" '
        f'aria-label="{label} {value}"><clipPath id="r"><rect width="{W:.1f}" height="{H}" rx="4"/></clipPath>'
        f'<g clip-path="url(#r)">{"".join(parts)}</g></svg>'
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "badges").mkdir(exist_ok=True)
    for theme, c in THEMES.items():
        (OUT / f"banner-{theme}.svg").write_text(banner(c))
    for name, label, value, color in BADGES:
        (OUT / "badges" / name).write_text(badge(label, value, color))
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")), len(BADGES), "badges")


if __name__ == "__main__":
    main()
