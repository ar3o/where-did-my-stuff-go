"""Paper-style architecture figures with real thumbnails taken from a processed scene.

    MONTSERRAT_DIR=<fonts> python docs/figures/make_architecture.py --scene config1 --film-dir <runs>/config1/film \\
        --photo IMG_2597 --group 8

Writes assets/arch_pipeline.png, arch_splat.png, arch_inventory.png, arch_checker.png and arch_changes.png on a white
background. Frozen models carry a snowflake, trained parts a flame.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath

from whatchanged.common.io import read_json
from whatchanged.common.scene import Scene

OUT = Path(__file__).resolve().parents[2] / "assets"
FONT_DIR = Path(os.environ.get("MONTSERRAT_DIR", "fonts"))
for f in FONT_DIR.glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams["font.family"] = "Montserrat"

C = dict(
    input="#ffffff",
    input_e="#7a8aa8",
    frozen="#e8f0fe",
    frozen_e="#5b84d6",
    trained="#fdeee0",
    trained_e="#e0873a",
    logic="#f4f6fb",
    logic_e="#7f8fb5",
    out="#efe9fb",
    out_e="#8b6fc0",
    ink="#2b2b2b",
    mute="#6b6b6b",
)
PANEL = dict(input="#8a8a8a", frozen="#5b84d6", trained="#e0873a", logic="#8b6fc0", out="#3f9d63")
RED, GREEN, AMBER = "#eb4034", "#34c759", "#ffb800"


class Canvas:
    def __init__(self, W: float, H: float, scale: float = 1.0):
        self.W, self.H = W, H
        self.fig = plt.figure(figsize=(W / 100 * scale, H / 100 * scale), dpi=100, facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self.ax.axis("off")

    def box(self, x, y, w, h, kind="logic", title=None, size=15, lw=1.8):
        """Dashed stage panel on white with a bold title, as in the companion repository's figures."""
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=16",
                fc="white",
                ec=PANEL[kind],
                lw=lw,
                ls=(0, (4, 3)),
                zorder=1,
            )
        )
        if title:
            self.text(x + 16, y + 26, title, size + 2, weight="bold", color=C["ink"])

    def text(self, x, y, s, size=13, color=None, ha="left", weight="normal", va="center", **kw):
        return self.ax.text(
            x, y, s, fontsize=size, color=color or C["ink"], ha=ha, va=va, fontweight=weight, zorder=6, **kw
        )

    def image(self, img, x, y, w, h=None, border="#bbbbbb", caption=None, cap_size=11):
        """Image with its width w; the height follows the aspect unless h crops it."""
        ih, iw = img.shape[:2]
        if h is None:
            h = w * ih / iw
        else:
            want = w / h
            if iw / ih > want:
                nw = int(ih * want)
                img = img[:, (iw - nw) // 2 : (iw - nw) // 2 + nw]
            else:
                nh = int(iw / want)
                img = img[(ih - nh) // 2 : (ih - nh) // 2 + nh]
        self.ax.imshow(img, extent=(x, x + w, y + h, y), zorder=3, interpolation="bilinear")
        self.ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=border, lw=1.2, zorder=4))
        if caption:
            self.text(x + w / 2, y + h + 14, caption, cap_size, color=C["mute"], ha="center")
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(self.H, 0)
        return h

    def arrow(self, x0, y0, x1, y1, color="#555555", lw=2.2, rad=0.0, style="-|>", ls="-"):
        self.ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle=style,
                mutation_scale=16,
                lw=lw,
                color=color,
                connectionstyle=f"arc3,rad={rad}",
                zorder=5,
                linestyle=ls,
            )
        )

    def tag(self, x, y, s, kind="logic", size=12):
        w = 9 * len(s) * size / 12 + 16
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y - 13),
                w,
                26,
                boxstyle="round,pad=0,rounding_size=8",
                fc="white",
                ec=C[kind + "_e"],
                lw=1.6,
                zorder=5,
            )
        )
        self.text(x + w / 2, y, s, size, ha="center", color=C["ink"])
        return w

    def snowflake(self, x, y, r=11, color="#3b7fc4"):
        for a in (0, 60, 120):
            dx, dy = r * math.cos(math.radians(a)), r * math.sin(math.radians(a))
            self.ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, lw=1.8, zorder=7, solid_capstyle="round")
        for a in range(0, 360, 60):
            ux, uy = math.cos(math.radians(a)), math.sin(math.radians(a))
            bx, by = x + ux * r * 0.62, y + uy * r * 0.62
            for s in (-1, 1):
                t = math.radians(a + s * 40)
                self.ax.plot(
                    [bx, bx + math.cos(t) * r * 0.32],
                    [by, by + math.sin(t) * r * 0.32],
                    color=color,
                    lw=1.4,
                    zorder=7,
                    solid_capstyle="round",
                )

    def flame(self, x, y, r=12, color="#e0602a"):
        pts = [(0, -1.0), (0.75, -0.2), (0.45, 0.55), (0.0, 1.0), (-0.45, 0.55), (-0.75, -0.2), (0, -1.0)]
        verts = [(x + px * r * 0.8, y + py * r) for px, py in pts]
        codes = [MPath.MOVETO] + [MPath.CURVE3, MPath.CURVE3] * 3
        self.ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec="#b84a1c", lw=1.2, zorder=7))
        self.ax.add_patch(
            Polygon(
                [(x, y + r * 0.2), (x + r * 0.25, y + r * 0.7), (x - r * 0.25, y + r * 0.7)],
                fc="#ffd27a",
                ec="none",
                zorder=8,
            )
        )

    def pill(self, x, y, s, size=12):
        """Dark tag in the corner, one line."""
        w = 8.4 * len(s) * size / 12 + 30
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y - 16), w, 32, boxstyle="round,pad=0,rounding_size=16", fc="#22252b", ec="none", zorder=5
            )
        )
        self.text(x + w / 2, y, s, size, ha="center", color="white")

    def module(self, x, y, w, h, label, frozen: bool | None, kind=None, size=13, sub=None):
        kind = kind or ("frozen" if frozen else "trained" if frozen is False else "logic")
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0,rounding_size=10", fc=C[kind], ec=C[kind + "_e"], lw=2.0, zorder=3
            )
        )
        self.text(x + w / 2, y + h / 2 - (7 if sub else 0), label, size, ha="center", weight="bold")
        if sub:
            self.text(x + w / 2, y + h / 2 + 12, sub, size - 3, ha="center", color=C["mute"])
        if frozen is True:
            self.snowflake(x + w - 14, y + 14)
        elif frozen is False:
            self.flame(x + w - 14, y + 15)

    def panel_label(self, x, y, s):
        self.text(x, y, s, 17, weight="bold", color="#333333")

    def legend(self, x, y):
        self.ax.add_patch(Rectangle((x, y), 560, 38, fc="white", ec="#cccccc", lw=1.2, zorder=3))
        self.snowflake(x + 24, y + 19)
        self.text(x + 44, y + 19, "frozen model", 12)
        self.flame(x + 190, y + 20)
        self.text(x + 210, y + 19, "trained on this room", 12)
        self.ax.add_patch(Rectangle((x + 395, y + 12), 14, 14, fc=C["input"], ec=C["input_e"], lw=1.4, zorder=4))
        self.text(x + 416, y + 19, "input", 12)
        self.ax.add_patch(Rectangle((x + 468, y + 12), 14, 14, fc=C["out"], ec=C["out_e"], lw=1.4, zorder=4))
        self.text(x + 489, y + 19, "output", 12)

    def save(self, name):
        self.fig.savefig(OUT / name, dpi=100, facecolor="white")
        plt.close(self.fig)
        print("wrote", name)


def rd(p, rgb=True):
    im = cv2.imread(str(p))
    return im[..., ::-1] if rgb else im


def tint(img, mask, col, a=0.55):
    out = img.astype(np.float32)
    out[mask] = out[mask] * (1 - a) + np.array(col, np.float32) * a
    return out.astype(np.uint8)


def colour_depth(d):
    v = np.clip(d / max(np.percentile(d[d > 0], 98), 1e-3), 0, 1)
    c = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
    c[d <= 0] = 255
    return c


def sfm_top_view(scene: Scene, size=360):
    pts = np.load(scene.root / "sfm_points.npz")
    xyz, rgb = pts["xyz"], pts["rgb"]
    keep = (pts["err"] < 1.0) & (pts["track"] >= 4) & (xyz[:, 2] > -0.3) & (xyz[:, 2] < 2.5)
    xyz, rgb = xyz[keep], rgb[keep]
    lo, hi = np.percentile(xyz[:, :2], 1, 0), np.percentile(xyz[:, :2], 99, 0)
    img = np.full((size, size, 3), 255, np.uint8)
    s = (size - 20) / max(hi - lo)
    uv = ((xyz[:, :2] - lo) * s + 10).astype(int)
    ok = (uv >= 0).all(1) & (uv < size).all(1)
    order = np.argsort(xyz[ok, 2])
    img[size - 1 - uv[ok][order, 1], uv[ok][order, 0]] = rgb[ok][order]
    img = cv2.dilate(img, np.ones((2, 2), np.uint8)) if False else img
    poses = scene.poses()["frames"]
    cam = np.array([f["T_wc"][0][3] if False else np.array(f["T_wc"])[:2, 3] for f in poses])
    cu = ((cam - lo) * s + 10).astype(int)
    for u, v in cu[::6]:
        if 0 <= u < size and 0 <= v < size:
            cv2.circle(img, (int(u), size - 1 - int(v)), 3, (214, 58, 90), -1)
    return img


def load_assets(scene: Scene, photo: str, group: int, film_dir: Path) -> dict:
    A = {}
    A["frame_a"], A["frame_b"] = rd(scene.frame_image("0100")), rd(scene.frame_image("0230"))
    A["frame_c"] = rd(scene.frame_image("0030"))
    A["photo"] = rd(scene.photo_image(photo))
    pd = scene.predicted / photo
    A["pred"] = rd(pd / "rgb.jpg")
    A["pred_raw"] = rd(pd / "rgb_raw.jpg")
    A["depth"] = colour_depth(np.load(pd / "depth.npy").astype(np.float32))
    known = np.load(pd / "known.npy")
    A["known"] = tint(A["photo"], ~known, (235, 64, 52))
    A["sfm"] = sfm_top_view(scene)
    z = np.load(scene.checker / "photos" / photo / "probs.npz")
    size = (A["photo"].shape[1], A["photo"].shape[0])
    A["p_app"] = cv2.resize(z["appeared"].astype(np.float32), size)
    A["p_mis"] = cv2.resize(z["missing"].astype(np.float32), size)
    heat = A["photo"].astype(np.float32)
    for k, col in (("p_app", (52, 199, 89)), ("p_mis", (235, 64, 52))):
        w = np.clip((A[k] - 0.3) / 0.7, 0, 1)[..., None] * 0.6 * known[..., None]
        heat = heat * (1 - w) + np.array(col, np.float32) * w
    A["heat"] = heat.astype(np.uint8)
    pair = scene.checker / "pairs" / "train" / "0001_3"
    A["pair_render"], A["pair_real"] = rd(pair / "render.jpg"), rd(pair / "real.jpg")
    lab = cv2.imread(str(pair / "label.png"), -1)
    over = A["pair_real"].copy()
    over = tint(over, lab == 1, (52, 199, 89), 0.6)
    over = tint(over, lab == 2, (235, 64, 52), 0.6)
    over = tint(over, lab == 255, (0, 0, 0), 0.5)
    A["pair_label"] = over
    A["pair_depth"] = colour_depth(np.load(pair / "depth_r.npy").astype(np.float32))
    A["inventory"] = rd(OUT / "media" / "inventory.jpg")
    A["hires"] = rd(scene.frames_hi / "0132.jpg")
    rec = read_json(scene.changes / f"g{group:02d}" / "changes.json")
    out = A["photo"].copy()
    for e in rec["events"]:
        col = {"removed": (235, 64, 52), "moved": (255, 184, 0), "added": (52, 199, 89)}[e["kind"]]
        for polys in e.get("outlines", {}).get(photo, {}).values():
            for p in polys:
                if len(p) >= 3:
                    cv2.polylines(out, [np.array(p, np.int32).reshape(-1, 1, 2)], True, col, 3, cv2.LINE_AA)
    A["outlined"] = out
    A["events"] = rec["events"]
    clip = film_dir / f"g{group:02d}.mp4"
    frames = {}
    if clip.exists():
        cap = cv2.VideoCapture(str(clip))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for key, frac in (("film_before", 0.04), ("film_fly", 0.33), ("film_change", 0.5), ("film_after", 0.97)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
            ok, f = cap.read()
            if ok:
                frames[key] = f[..., ::-1]
    A.update(frames)
    return A


def fig_pipeline(A):
    c = Canvas(2600, 1180)
    c.pill(40, 40, "one walk, a few photos, a list of names", 13)
    c.legend(2000, 22)
    cols = [
        (40, "(a) Frames and photos", "input"),
        (420, "(b) Camera poses", "frozen"),
        (800, "(c) Splat of the room", "trained"),
        (1180, "(d) Objects in 3D", "frozen"),
        (1560, "(e) Change checker", "trained"),
        (1940, "(f) Changes in a group", "logic"),
    ]
    for x, title, kind in cols:
        c.box(x, 110, 350, 760, kind, title, 17)
    # 1 ingest
    c.image(A["frame_a"], 56, 160, 318, caption="walk frame, HLG tone-mapped")
    c.image(A["photo"], 56, 420, 318, 238, caption="query photo, EXIF oriented")
    c.module(56, 700, 150, 70, "sharp frames", None, size=12, sub="64 + 220")
    c.module(224, 700, 150, 70, "HDR to SDR", None, size=12, sub="bt709")
    # 2 sfm
    c.image(A["sfm"], 440, 160, 318, 318, caption="SfM points and walk cameras (top view)")
    c.module(436, 520, 160, 70, "COLMAP", True, size=13, sub="walk + photos")
    c.module(612, 520, 160, 70, "MoGe-3", True, size=13, sub="metric scale")
    c.module(436, 620, 336, 70, "world frame", None, size=13, sub="floor at z = 0, metres")
    # 3 splat
    c.image(A["pred_raw"], 820, 160, 318, 238, caption="splat render at the photo's refined pose")
    c.image(A["known"], 820, 430, 318, 238, caption="unknown pixels in red")
    c.module(816, 700, 160, 70, "3D Gaussian splat", False, size=11, sub="reference of the room")
    c.module(992, 700, 160, 70, "pose refinement", None, size=12, sub="render and compare")
    # 4 inventory
    inv = A["inventory"]
    c.image(inv[:, : inv.shape[1] // 2], 1200, 160, 318, 238, caption="named 3D objects, rendered")
    c.image(A["hires"], 1200, 430, 318, 238, caption="SAM 3 tiles of a 1920 px frame")
    c.module(1196, 700, 160, 70, "SAM 3", True, size=13, sub="name list prompts")
    c.module(1372, 700, 160, 70, "SigLIP 2", True, size=13, sub="names from crops")
    # 5 checker
    c.image(A["pair_render"], 1580, 160, 154, 116, caption="edited render", cap_size=10)
    c.image(A["pair_real"], 1750, 160, 154, 116, caption="real frame", cap_size=10)
    c.image(A["pair_label"], 1580, 320, 154, 116, caption="labels", cap_size=10)
    c.image(A["heat"], 1750, 320, 154, 116, caption="predicted change", cap_size=10)
    c.module(1580, 480, 150, 70, "DINOv2", True, size=13, sub="PCA 128 dims")
    c.module(1750, 480, 150, 70, "decoder", False, size=13, sub="3 classes, 1/4 res")
    c.module(
        1580, 580, 320, 70, "training pairs from splat edits", None, size=12, sub="remove, transplant, move, rotate"
    )
    c.module(1580, 700, 320, 70, "calibration", None, size=13, sub="Platt scaling on held-out walk")
    # 6 changes
    c.image(A["outlined"], 1960, 160, 318, 238, caption="changes found in one group")
    c.module(1960, 430, 318, 60, "re-identify, count, assign", None, size=12, sub="calibrated probabilities")
    c.module(1960, 510, 318, 60, "can't verify", None, size=12, sub="when the old place is not seen")
    c.module(1960, 590, 318, 60, "text from 3D facts", None, size=12, sub="kind, distance, landmark")
    c.module(1960, 700, 318, 70, "events", None, kind="out", size=13, sub="removed / moved / added + confidence")
    # 7 film
    c.box(2330, 110, 230, 760, "out", "(g) Film", 17)
    for i, k in enumerate(("film_before", "film_change", "film_after")):
        if k in A:
            c.image(
                A[k],
                2346,
                160 + i * 200,
                198,
                150,
                caption=["before", "fly to the change", "real photo"][i],
                cap_size=10,
            )
    # arrows
    for x in (392, 772, 1152, 1532, 1912, 2292):
        c.arrow(x, 480, x + 30, 480)
    c.arrow(1060, 880, 1650, 880, color=C["trained_e"], rad=-0.2, ls="--")
    c.text(
        1355,
        960,
        "(a) edits of the splat render give labelled training pairs without a single real change",
        13,
        ha="center",
        color=C["trained_e"],
    )
    # bottom strip: inputs
    c.box(40, 1000, 2520, 150, "input")
    c.text(60, 1030, "Inputs", 15, weight="bold", color=C["input_e"])
    c.text(60, 1075, "(i) one walk-around video of the room in its reference state", 14)
    c.text(60, 1105, "(ii) photos or a video taken later, in groups that each show one state of the room", 14)
    c.text(60, 1135, "(iii) a list of object names, the only scene-specific input", 14)
    c.text(
        1400,
        1075,
        "Outputs: JSON events with confidences and 3D positions, stills, a film that flies to every change",
        14,
    )
    c.save("arch_pipeline.png")


def fig_splat(A):
    c = Canvas(2400, 760)
    c.legend(1800, 14)
    c.box(30, 60, 1010, 660, "input", "(a) Reference splat from the walk", 17)
    c.image(A["frame_a"], 50, 110, 200, 150, caption="frames", cap_size=11)
    c.image(A["frame_b"], 50, 300, 200, 150, caption="held-out frame", cap_size=11)
    c.module(290, 130, 170, 90, "COLMAP SfM", True, sub="271 / 276 frames, 0.62 px")
    c.image(A["sfm"], 290, 260, 170, 170, caption="sparse points", cap_size=11)
    c.module(500, 130, 200, 90, "3DGS training", False, sub="30k steps, 2.5M Gaussians")
    c.module(500, 260, 200, 90, "colour affine per frame", False, size=11, sub="absorbs auto-exposure")
    c.image(A["frame_b"], 760, 110, 250, 188, caption="rendered view of the splat", cap_size=11)
    c.text(765, 330, "held-out PSNR 26.3 dB", 14, color=C["mute"])
    c.module(500, 400, 200, 80, "MoGe-3 scale", True, sub="metres, floor z = 0")
    c.arrow(250, 185, 290, 175)
    c.arrow(460, 175, 500, 175)
    c.arrow(700, 175, 760, 175)
    c.arrow(375, 220, 375, 260)
    c.arrow(600, 350, 600, 400, rad=0.0)
    c.arrow(600, 220, 600, 260)
    c.box(1070, 60, 1300, 660, "input", "(b) The predicted view of a photo and the pixels the walk never saw", 17)
    c.image(A["photo"], 1090, 120, 300, 225, caption="query photo")
    c.module(1090, 400, 300, 80, "SfM registration", True, sub="start pose, shared with the walk")
    c.module(1090, 510, 300, 90, "render and compare", False, sub="Geman-McClure, outlier mask")
    c.arrow(1240, 345, 1240, 400)
    c.arrow(1240, 480, 1240, 510)
    c.image(A["pred"], 1450, 120, 300, 225, caption="predicted view, same pose")
    c.image(A["depth"], 1780, 120, 300, 225, caption="predicted depth")
    c.image(A["known"], 2110, 120, 240, 180, caption="seen by >= 3 walk cameras", cap_size=11)
    c.module(1450, 400, 300, 80, "view support", None, sub="walk cameras that saw the point")
    c.module(1780, 400, 300, 80, "MoGe-3 depth of the photo", True, size=12, sub="aligned to the prediction")
    c.module(2110, 400, 240, 80, "usable photo?", None, sub="aligned depth within 3 %")
    c.arrow(1390, 555, 1450, 440)
    c.arrow(1750, 260, 1780, 440, rad=0.1)
    c.arrow(2080, 440, 2110, 440)
    c.text(1450, 540, "Pixels the walk barely saw are unknown and never count as evidence.", 13, color=C["mute"])
    c.text(
        1450,
        580,
        "The predicted view differs from the photo where the room changed or where the splat is wrong.",
        13,
        color=C["mute"],
    )
    c.save("arch_splat.png")


def fig_inventory(A):
    c = Canvas(2400, 760)
    c.legend(1800, 14)
    inv = A["inventory"]
    half = inv.shape[1] // 2
    c.box(30, 60, 2340, 660, "input", "Finding and naming the objects in the splat", 17)
    c.module(50, 120, 220, 90, "name list", None, kind="input", sub="furniture, items, static")
    c.image(A["hires"], 50, 250, 220, 165, caption="walk frame, 1920 px", cap_size=11)
    c.module(320, 120, 230, 90, "SAM 3", True, sub="full frame + 2x2 tiles")
    c.image(A["frame_a"], 320, 250, 230, 130, caption="masks per phrase", cap_size=11)
    c.module(600, 120, 270, 90, "(a) Gaussian voting", None, sub="masks back onto Gaussians")
    c.image(inv[:, :half], 600, 250, 270, 202, caption="Gaussians that make up each object", cap_size=11)
    c.module(920, 120, 250, 90, "(b) region growing", None, sub="thin parts, e.g. the broom")
    c.module(920, 250, 250, 90, "(c) geometric merge", None, sub="touching, alike, object-sized")
    c.module(1220, 120, 240, 90, "(d) SAM 3 votes", True, sub="z-scored x overlap x visibility")
    c.module(1220, 250, 240, 90, "(e) SigLIP 2", True, sub="4 clearest views, masked crops")
    c.module(1510, 185, 200, 90, "p = p_SAM x p_SigLIP", None, size=12, sub="name, runner-up, margin")
    c.image(inv[:, half:], 1770, 120, 280, 210, caption="96 named objects", cap_size=11)
    c.module(2080, 120, 270, 90, "fingerprints", None, sub="DINOv2 embedding, colour, size")
    c.module(2080, 250, 270, 90, "walk split", None, sub="last 20 % held out for calibration")
    for a, b in ((270, 320), (550, 600), (870, 920), (1170, 1220), (1460, 1510), (1710, 1770), (2050, 2080)):
        c.arrow(a, 165, b, 165)
    c.arrow(1045, 210, 1045, 250)
    c.arrow(1340, 210, 1340, 250, ls="--")
    c.text(
        50,
        520,
        "Detections vote for the Gaussians that produce their pixels, weighted by blending weight.",
        13,
        color=C["mute"],
    )
    c.text(
        50,
        560,
        "Detections sharing Gaussians across frames become one object; smaller masks win a pixel.",
        13,
        color=C["mute"],
    )
    c.text(
        50,
        600,
        "Geometry comes first: a looser merge once glued a milk jug to a fruit bowl.",
        13,
        color=C["mute"],
    )
    c.text(
        50,
        640,
        "Two independent name sources are multiplied, so a plain box needs both to agree to get a confident name.",
        13,
        color=C["mute"],
    )
    c.save("arch_inventory.png")


def fig_checker(A):
    c = Canvas(2400, 900)
    c.legend(1800, 14)
    c.box(30, 60, 1180, 800, "input", "(a) Training pairs made from edits of the splat", 17)
    c.image(A["frame_c"], 50, 120, 220, 165, caption="real walk frame (the 'after')", cap_size=11)
    c.module(310, 120, 230, 100, "edit the splat", None, sub="remove, copy, move, rotate")
    c.module(310, 250, 230, 70, "pose jitter 0-2 cm", None, size=12)
    c.image(A["pair_render"], 580, 120, 260, 195, caption="edited render (the 'before')", cap_size=11)
    c.image(A["pair_real"], 870, 120, 260, 195, caption="real frame, augmented", cap_size=11)
    c.image(A["pair_label"], 580, 380, 260, 195, caption="labels", cap_size=11)
    c.image(A["pair_depth"], 870, 380, 260, 195, caption="render depth", cap_size=11)
    c.arrow(270, 200, 310, 175)
    c.arrow(540, 175, 580, 215)
    c.arrow(425, 220, 425, 250, ls="--")
    c.arrow(710, 315, 710, 380)
    c.arrow(1000, 315, 1000, 380)
    c.text(
        50,
        640,
        "Green: in the photo only (appeared).  Red: in the render only (missing).  Dark: ignored,",
        13,
        color=C["mute"],
    )
    c.text(
        50, 675, "where fewer than 3 other walk cameras saw the point or an edit overlaps itself.", 13, color=C["mute"]
    )
    c.text(
        50,
        725,
        "A third of the pairs have no edit, so floaters, blur and colour differences become negatives.",
        13,
        color=C["mute"],
    )
    c.text(
        50,
        760,
        "Decoy fills sit in some unedited renders, so a filled patch is not shortcut evidence for a change.",
        13,
        color=C["mute"],
    )
    c.text(50, 810, "Training: 864 pairs; validation: 220 pairs from the last 20 % of the walk.", 13, color=C["mute"])
    c.box(1240, 60, 1130, 800, "input", "(b) The checker and its calibration", 17)
    c.module(1260, 130, 200, 80, "DINOv2 ViT-L/14", True, size=12, sub="render and real, 924 px")
    c.module(1260, 240, 200, 80, "PCA to 128 dims", None, size=12, sub="fitted on walk frames")
    c.module(1500, 130, 260, 80, "[fr, fq, |fr-fq|, fr*fq]", None, size=12, sub="feature branch at patch grid")
    c.module(1500, 240, 260, 80, "low-level branch", None, size=12, sub="RGB, depth difference")
    c.module(1800, 175, 220, 90, "decoder", False, sub="3 classes at 1/4 res")
    c.image(A["heat"], 2060, 130, 290, 218, caption="P(appeared) green, P(missing) red", cap_size=11)
    for a, b, y in ((1460, 1500, 170), (1760, 1800, 200)):
        c.arrow(a, y, b, y + (0 if a == 1460 else 0))
    c.arrow(1360, 210, 1360, 240)
    c.arrow(1630, 210, 1630, 240, ls="--")
    c.arrow(2020, 220, 2060, 220)
    c.module(1260, 420, 300, 90, "mean probability on a region", None, size=12, sub="known pixels, with log size")
    c.module(1600, 420, 250, 90, "Platt scaling", False, sub="2-fold by frame")
    c.module(1890, 420, 250, 90, "calibrated P(change)", None, kind="out", sub="ECE 0.014 to 0.026")
    c.arrow(1560, 465, 1600, 465)
    c.arrow(1850, 465, 1890, 465)
    c.arrow(2200, 350, 2020, 420)
    c.text(
        1260,
        600,
        "One photo gives a probability for each object; several photos of a group are combined by noisy-OR.",
        13,
        color=C["mute"],
    )
    c.text(
        1260,
        640,
        "No threshold is tuned on the change photos: 0.5 on a calibrated probability, or a value measured on",
        13,
        color=C["mute"],
    )
    c.text(1260, 675, "held-out walk frames.", 13, color=C["mute"])
    c.save("arch_checker.png")


def fig_changes(A):
    c = Canvas(2400, 900)
    c.legend(1800, 14)
    c.box(30, 60, 2340, 800, "input", "(a) Deciding what changed in one group of three photos", 17)
    c.image(A["photo"], 50, 130, 230, 172, caption="photo 1 of 3", cap_size=11)
    c.image(A["heat"], 50, 340, 230, 172, caption="checker maps", cap_size=11)
    c.module(330, 130, 250, 80, "object states", None, sub="present / missing / covered")
    c.module(330, 235, 250, 80, "detections", True, size=13, sub="SAM 3 on photo and on a render")
    c.module(330, 340, 250, 80, "claimed or counted?", None, size=12, sub="on an object, or already in the render")
    c.image(A["pred"], 330, 450, 250, 188, caption="render of the reference, same pose", cap_size=11)
    c.module(640, 130, 250, 80, "3D clusters", None, sub="lifted boxes, same-place radius")
    c.module(640, 235, 250, 80, "re-identify", False, size=13, sub="p_same from 5 terms")
    c.module(640, 340, 250, 80, "assign", None, sub="Hungarian, accepted at p >= 0.5")
    c.module(940, 130, 220, 100, "found elsewhere", None, kind="logic", sub="MOVED")
    c.module(940, 255, 220, 100, "not found, old place seen", None, kind="logic", size=11, sub="REMOVED")
    c.module(940, 380, 220, 100, "unclaimed cluster", None, kind="logic", sub="ADDED")
    c.module(940, 505, 220, 90, "not seen", None, kind="logic", sub="can't verify")
    c.image(A["outlined"], 1220, 130, 340, 255, caption="removed red, added green, moved amber")
    c.module(1220, 440, 340, 80, "events", None, kind="out", sub="name, kind, confidence, 3D place, text")
    c.text(1220, 560, 'e.g. "The egg carton moved 40 cm (80 % sure)."', 12, color=C["mute"])
    c.arrow(280, 215, 330, 170)
    c.arrow(280, 420, 330, 380)
    c.arrow(580, 170, 640, 170)
    c.arrow(580, 275, 640, 275)
    c.arrow(580, 380, 640, 380)
    c.arrow(890, 170, 940, 175)
    c.arrow(890, 275, 940, 300)
    c.arrow(890, 380, 940, 430)
    c.arrow(1160, 180, 1220, 200)
    c.arrow(1160, 305, 1220, 330)
    c.arrow(1160, 430, 1220, 470)
    c.arrow(1390, 385, 1390, 440)
    c.panel_label(1630, 130, "(b) The film")
    keys = [
        ("film_before", "reference splat at the photo's pose"),
        ("film_fly", "camera flies to a change"),
        ("film_change", "hologram / glide / rise"),
        ("film_after", "back to the real photo"),
    ]
    for i, (k, cap) in enumerate(keys):
        if k in A:
            c.image(A[k], 1620 + (i % 2) * 370, 160 + (i // 2) * 310, 350, 262, caption=cap, cap_size=11)
    c.text(1620, 760, "red hologram: removed   glide: moved   rise: added", 12, color=C["mute"])
    c.save("arch_changes.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--film-dir", type=Path, required=True)
    ap.add_argument("--photo", default="IMG_2597")
    ap.add_argument("--group", type=int, default=8)
    args = ap.parse_args()
    A = load_assets(Scene(args.scene), args.photo, args.group, args.film_dir)
    for f in (fig_pipeline, fig_splat, fig_inventory, fig_checker, fig_changes):
        f(A)


if __name__ == "__main__":
    main()
