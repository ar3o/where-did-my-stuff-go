"""Checker stage: training pairs, DINOv2 features, training, per-photo change maps, confidence calibration.

python -m whatchanged.checker.run --scene config1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from whatchanged.checker.settings import load_settings
from whatchanged.common import features
from whatchanged.common.scene import Scene


def encode_features(scene: Scene, need_mib: int) -> None:
    """DINOv2 PCA features of every pair (render and real) and every photo (photo and predicted view)."""
    items = []
    for d in sorted((scene.checker / "pairs").glob("*/*")):
        if (d / "meta.json").exists():
            items += [(d / "render.jpg", d / "f_render.npy"), (d / "real.jpg", d / "f_real.npy")]
    for p in sorted(x for x in scene.predicted.iterdir() if (x / "pose.json").exists()):
        items += [
            (p / "rgb_raw.jpg", scene.features / "photos" / f"{p.name}_render.npy"),
            (scene.photo_image(p.name), scene.features / "photos" / f"{p.name}_real.npy"),
        ]
    frames = sorted(scene.frames.glob("*.jpg"))
    features.encode(items, scene.features / "pca.npz", frames, scene.features / "jobs_checker.json", need_mib)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()
    extra = ["--config", str(args.config)] if args.config else []
    scene = Scene(args.scene)

    def step(module: str) -> None:
        subprocess.run([sys.executable, "-m", module, "--scene", args.scene] + extra, check=True)

    step("whatchanged.checker.pairs")
    encode_features(scene, load_settings(args.config).features.need_mib)
    if not (scene.runs / "checker" / "best.pt").exists():
        step("whatchanged.checker.train")
    step("whatchanged.checker.infer")
    step("whatchanged.checker.calibrate")


if __name__ == "__main__":
    main()
