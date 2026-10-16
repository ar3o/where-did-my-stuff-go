"""MoGe-3 metric depth for a list of images; vision env.

    python -m whatchanged.common.moge_worker jobs.json

jobs.json = {"moge": "Ruicheng/moge-3-vitl", "need_mib": 3000, "items": [{"image": "a.jpg", "out": "a.npz"}]}

Per image: depth (float16 metres, 0 where MoGe's mask is off) and K (pixels). Items whose output exists are skipped.
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

RESOLUTION_LEVEL = 9


def main() -> None:
    jobs = json.loads(Path(sys.argv[1]).read_text())
    items = [it for it in jobs["items"] if not Path(it["out"]).exists()]
    if not items:
        return
    with gpu_stage("moge", need_mib=int(jobs.get("need_mib", 3000))):
        from moge.model.v3 import MoGeModel

        model = MoGeModel.from_pretrained(jobs["moge"]).cuda().eval()
        t0 = time.time()
        for it in items:
            t1 = time.time()
            im = cv2.imread(it["image"])[..., ::-1].copy()
            x = torch.tensor(im, dtype=torch.float32, device="cuda").permute(2, 0, 1) / 255
            with torch.no_grad():
                o = model.infer(x, resolution_level=RESOLUTION_LEVEL, use_fp16=True)
            d = o["depth"].float().cpu().numpy()
            ok = o["mask"].cpu().numpy().astype(bool) & np.isfinite(d)
            k = o["intrinsics"].float().cpu().numpy()
            h, w = im.shape[:2]
            K = np.array([[k[0, 0] * w, 0, k[0, 2] * w], [0, k[1, 1] * h, k[1, 2] * h], [0, 0, 1]])
            save_npz(it["out"], depth=np.where(ok, d, 0).astype(np.float16), K=K, seconds=time.time() - t1)
        log(f"moge: {len(items)} images in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
