"""Run the trained checker on the query photos: render = the splat at the refined pose (raw splat colour, as in
training), real = the photo, depths = predicted depth and MoGe-3 aligned to it, known = the photo's known mask.

    python -m whatchanged.checker.infer --scene config1

Output per photo: checker/photos/<photo>/probs.npz with P(appeared) and P(missing) at 1/4 resolution (float16) and
the full image size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from whatchanged.checker import model as checker_model
from whatchanged.checker.settings import load_settings
from whatchanged.common.gpu import gpu_stage
from whatchanged.common.io import log
from whatchanged.common.photoview import PhotoView
from whatchanged.common.scene import Scene


def photo_probabilities(scene: Scene, net, name: str, device: str) -> np.ndarray | None:
    feats = scene.features / "photos"
    if not (feats / f"{name}_real.npy").exists():
        return None
    pv = PhotoView(scene, name)
    if not pv.exists:
        return None
    ren = cv2.imread(str(scene.predicted / name / "rgb_raw.jpg"))[..., ::-1]
    real = cv2.imread(str(scene.photo_image(name)))[..., ::-1]
    low = checker_model.quarter_inputs(ren, real, pv.D_pred, pv.D_photo, pv.known)
    return checker_model.probabilities(
        net, np.load(feats / f"{name}_render.npy"), np.load(feats / f"{name}_real.npy"), low, device
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()
    cfg = load_settings(args.config).infer
    scene = Scene(args.scene)
    names = sorted(p.name for p in scene.predicted.iterdir() if (p / "pose.json").exists())
    with gpu_stage("checker_infer", need_mib=cfg.need_mib):
        net = checker_model.load(scene.runs / "checker" / "best.pt", "cuda")
        for name in names:
            out = scene.checker / "photos" / name / "probs.npz"
            if out.exists():
                continue
            p = photo_probabilities(scene, net, name, "cuda")
            if p is None:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            H, W = cv2.imread(str(scene.photo_image(name))).shape[:2]
            np.savez_compressed(
                out, appeared=p[1].astype(np.float16), missing=p[2].astype(np.float16), size=np.array([W, H])
            )
    log(f"checker: probabilities for {len(names)} photos")


if __name__ == "__main__":
    main()
