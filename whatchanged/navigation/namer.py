"""Names newly observed obstacle blobs: SigLIP 2 zero-shot (siglip_server.py in the vision env, kept running for the
whole robot run) on a crop of the episode render around the blob's 3-D box. Vocabulary = the memory inventory's movable
names + generic obstacle words; a name is accepted when p >= 0.3, else 'object'. The crop is the sensor's own view;
nothing from the episode files is used."""

from __future__ import annotations

import json
import os
import subprocess

import cv2
import numpy as np

from whatchanged import paths
from whatchanged.common.io import log, write_json
from whatchanged.common.workers import environment
from whatchanged.navigation.common import nav_dir

EXTRA = [
    "door",
    "cardboard box",
    "moving box",
    "storage box",
    "chair",
    "table",
    "stool",
    "bag",
    "pillow",
    "plant pot",
    "toy",
]
P_MIN = 0.3


class Namer:
    def __init__(self, scene: str, info: dict, tag: str):
        names = sorted(
            {
                o["name"].replace(" (uncertain)", "")
                for o in info["objects"]
                if o["cls"] != "static" and not o["name"].startswith("object")
            }
        )
        self.vocab = names + [e for e in EXTRA if e not in names]
        d = nav_dir(scene) / "namer" / tag
        d.mkdir(parents=True, exist_ok=True)
        self.dir = d
        env_cfg = environment()
        write_json(dict(phrases=self.vocab, siglip=env_cfg.siglip), d / "vocab.json")
        env = dict(os.environ)
        env["PYTHONPATH"] = ":".join(
            p for p in (env_cfg.vision_pythonpath, str(paths.REPO), env.get("PYTHONPATH", "")) if p
        )
        self.log_file = open(d / "server.log", "w")
        self.p = subprocess.Popen(
            [env_cfg.vision_python, "-m", "whatchanged.navigation.siglip_server", str(d / "vocab.json")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log_file,
            env=env,
            cwd=str(paths.REPO),
            text=True,
            bufsize=1,
        )
        r = json.loads(self.p.stdout.readline())
        assert r.get("ready"), r
        self.k = 0
        log(f"namer ready ({r['n']} phrases, {r.get('torch_peak_mib')} MiB)")

    def name(self, rgb: np.ndarray, box: dict, TF: np.ndarray, K: np.ndarray, exclude=()) -> tuple[str, float]:
        """Name and probability of the thing inside box (F frame) seen by the camera TF in rgb."""
        H, W = rgb.shape[:2]
        c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        unit = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        P = unit * (np.array(box["size"]) / 2) @ R.T + np.array(box["centre"])
        Tcw = np.linalg.inv(TF)
        pc = P @ Tcw[:3, :3].T + Tcw[:3, 3]
        if (pc[:, 2] < 0.25).any():
            return "object", 0.0
        uv = np.c_[K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]]
        (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
        cu, cv_ = (x0 + x1) / 2, (y0 + y1) / 2
        side = max(160.0, 1.5 * max(x1 - x0, y1 - y0))
        xa, ya, xb, yb = [int(round(v)) for v in (cu - side / 2, cv_ - side / 2, cu + side / 2, cv_ + side / 2)]
        pad = max(0, -xa, -ya, xb - W, yb - H)
        img = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else rgb
        crop = cv2.resize(img[ya + pad : yb + pad, xa + pad : xb + pad], (224, 224), interpolation=cv2.INTER_AREA)
        path = self.dir / f"c{self.k % 32:02d}.jpg"
        self.k += 1
        cv2.imwrite(str(path), crop[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.p.stdin.write(json.dumps(dict(crops=[str(path)])) + "\n")
        self.p.stdin.flush()
        pr = np.array(json.loads(self.p.stdout.readline())["probs"][0])
        for e in exclude:  # names of memory objects at that spot are not 'new' things
            if e in self.vocab:
                pr[self.vocab.index(e)] = 0.0
        pr = pr / max(pr.sum(), 1e-9)
        j = int(pr.argmax())
        return (self.vocab[j], float(pr[j])) if pr[j] >= P_MIN else ("object", float(pr[j]))

    def close(self) -> None:
        try:
            self.p.stdin.write(json.dumps(dict(quit=True)) + "\n")
            self.p.stdin.flush()
            self.p.wait(timeout=20)
        except Exception:
            self.p.kill()
        self.log_file.close()
