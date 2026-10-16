"""Hero strip: the same moment as the real photo, the splat rendered at the same pose, and the photo with the
detected changes (outlines, plus the checker's appeared / missing probability as a tint). Light and dark JPEGs.

    MONTSERRAT_DIR=<fonts> python docs/figures/make_hero.py --scene config1 --group 8 --photo IMG_2597
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from whatchanged.common.io import read_json
from whatchanged.common.scene import Scene

OUT = Path(__file__).resolve().parents[2] / "assets"
FONT = Path(os.environ.get("MONTSERRAT_DIR", "fonts")) / "Montserrat-Medium.ttf"
THEMES = {"light": dict(bg=(255, 255, 255), fg=(95, 88, 89)), "dark": dict(bg=(13, 17, 23), fg=(147, 139, 138))}
TYPE_COL = {"removed": (52, 64, 235), "moved": (0, 184, 255), "added": (89, 199, 52)}  # BGR
TILE = (640, 480)
GAP = 70


def change_view(scene: Scene, group: int, photo: str) -> np.ndarray:
    rec = read_json(scene.changes / f"g{group:02d}" / "changes.json")
    img = cv2.imread(str(scene.photo_image(photo)))
    z = np.load(scene.checker / "photos" / photo / "probs.npz")
    known = np.load(scene.predicted / photo / "known.npy")
    size = (img.shape[1], img.shape[0])
    heat = {k: cv2.resize(z[k].astype(np.float32), size) * known for k in ("appeared", "missing")}
    out = img.astype(np.float32)
    for key, col in (("appeared", TYPE_COL["added"]), ("missing", TYPE_COL["removed"])):
        w = np.clip((heat[key] - 0.3) / 0.7, 0, 1)[..., None] * 0.55
        out = out * (1 - w) + np.array(col, np.float32) * w
    out = out.astype(np.uint8)
    for e in rec["events"]:
        for polys in e.get("outlines", {}).get(photo, {}).values():
            for p in polys:
                if len(p) >= 3:
                    cv2.polylines(
                        out, [np.array(p, np.int32).reshape(-1, 1, 2)], True, TYPE_COL[e["kind"]], 3, cv2.LINE_AA
                    )
    return out


def tile(im: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv2.resize(im, TILE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--photo", required=True)
    args = ap.parse_args()
    scene = Scene(args.scene)
    tiles = [
        cv2.imread(str(scene.photo_image(args.photo))),
        cv2.imread(str(scene.predicted / args.photo / "rgb.jpg")),
        change_view(scene, args.group, args.photo),
    ]
    labels = ["PHOTO", "SPLAT AT THE SAME POSE", "CHANGES"]
    W, top = 2000, 20
    x0 = (W - 3 * TILE[0] - 2 * GAP) // 2
    H = top + TILE[1] + 96
    font = ImageFont.truetype(str(FONT), 34)
    for theme, c in THEMES.items():
        im = Image.new("RGB", (W, H), c["bg"])
        d = ImageDraw.Draw(im)
        for i, (t, label) in enumerate(zip(tiles, labels)):
            x = x0 + i * (TILE[0] + GAP)
            im.paste(tile(t), (x, top))
            d.text((x + TILE[0] / 2, top + TILE[1] + 66), " ".join(label), font=font, fill=c["fg"], anchor="ms")
            if i < 2:
                ax, ay = x + TILE[0] + 18, top + TILE[1] / 2
                d.line((ax, ay, ax + GAP - 36, ay), fill=c["fg"], width=3)
                d.line((ax + GAP - 36, ay, ax + GAP - 48, ay - 10), fill=c["fg"], width=3)
                d.line((ax + GAP - 36, ay, ax + GAP - 48, ay + 10), fill=c["fg"], width=3)
        im.save(OUT / f"hero_strip-{theme}.jpg", quality=88, optimize=True, progressive=True)
    print("wrote hero strips for", args.photo)


if __name__ == "__main__":
    main()
