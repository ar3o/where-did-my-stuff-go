"""Frozen DINOv2 ViT-L/14-reg4 patch features reduced to 128 dims with a PCA; vision env.

    python -m whatchanged.common.dino_worker jobs.json

jobs.json = {"dino_hub": <local torch-hub checkout>, "pca": "pca.npz", "fit_images": [...], "need_mib": 3000,
             "items": [{"image": "a.jpg", "out": "a.npy"}]}
Images are resized to a long side of 924 px (66 patches; 46 patches were too coarse for 2-5 cm moves). Patch tokens are
L2-normalised, projected with the PCA (fitted once on 150 of `fit_images` when `pca` does not exist yet) and saved as
float16 (128, h, w). Items whose output exists are skipped.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from whatchanged.common.gpu import gpu_stage
from whatchanged.common.io import log, save_npz

MODEL = "dinov2_vitl14_reg"
LONG = 924
PATCH = 14
N_COMPONENTS = 128
FIT_IMAGES = 150
FIT_TOKENS = 700
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def grid_size(w: int, h: int) -> tuple[int, int]:
    s = LONG / max(w, h)
    return max(PATCH, int(round(w * s / PATCH)) * PATCH), max(PATCH, int(round(h * s / PATCH)) * PATCH)


def tokens(model, path: str):
    im = cv2.imread(str(path))[..., ::-1]
    W, H = grid_size(im.shape[1], im.shape[0])
    x = (cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 - MEAN) / STD
    x = torch.from_numpy(x).permute(2, 0, 1)[None].cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        o = model.forward_features(x)
    return torch.nn.functional.normalize(o["x_norm_patchtokens"][0].float(), dim=-1), (H // PATCH, W // PATCH)


def fit_pca(model, images: list[str], out: Path) -> None:
    torch.manual_seed(0)
    sel = np.random.default_rng(0).choice(len(images), min(FIT_IMAGES, len(images)), replace=False)
    X = []
    for i in sel:
        tk, _ = tokens(model, images[i])
        X.append(tk[torch.randperm(len(tk))[:FIT_TOKENS]])
    X = torch.cat(X)
    mu = X.mean(0)
    _, S, V = torch.pca_lowrank(X - mu, q=N_COMPONENTS, center=False)
    total = float(((X - mu) ** 2).sum(1).mean())
    explained = float((S[:N_COMPONENTS] ** 2).sum() / (len(X) - 1)) / total
    save_npz(out, mean=mu.cpu().numpy(), comp=V[:, :N_COMPONENTS].cpu().numpy(), explained=explained)
    log(f"dino: pca on {len(X)} tokens explains {explained:.3f}")


def main() -> None:
    jobs = json.loads(Path(sys.argv[1]).read_text())
    items = [it for it in jobs["items"] if not Path(it["out"]).exists()]
    pca_path = Path(jobs["pca"])
    if not items and pca_path.exists():
        return
    with gpu_stage("dino", need_mib=int(jobs.get("need_mib", 3000))):
        model = torch.hub.load(jobs["dino_hub"], MODEL, source="local").eval().cuda()
        if not pca_path.exists():
            fit_pca(model, jobs["fit_images"], pca_path)
        z = np.load(pca_path)
        mu, comp = torch.as_tensor(z["mean"]).cuda(), torch.as_tensor(z["comp"]).cuda()
        t0 = time.time()
        for it in items:
            tk, (h, w) = tokens(model, it["image"])
            f = ((tk - mu) @ comp).reshape(h, w, N_COMPONENTS).permute(2, 0, 1)
            Path(it["out"]).parent.mkdir(parents=True, exist_ok=True)
            np.save(it["out"], f.cpu().numpy().astype(np.float16))
        log(f"dino: {len(items)} images in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
