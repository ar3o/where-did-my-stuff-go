"""One query photo with everything the comparison needs: the refined pose, the splat's predicted depth, the known mask
and the photo's MoGe-3 depth aligned to the prediction.

A photo is usable ("good") when it registered in the SfM model and its aligned depth agrees with the predicted depth to
3 % (median over known pixels). Photos that fail either test are left out of the comparison.
"""

from __future__ import annotations

import cv2
import numpy as np

from whatchanged.common import depth as depthlib
from whatchanged.common import geometry as geo
from whatchanged.common.io import read_json
from whatchanged.common.scene import Scene

VIEW_BORDER_PX = 24


class PhotoView:
    def __init__(self, scene: Scene, name: str):
        self.name = name
        pdir = scene.predicted / name
        self.good = False
        self.exists = (pdir / "pose.json").exists()
        if not self.exists:
            return
        self.pose = read_json(pdir / "pose.json")
        self.W, self.H = self.pose["size"]
        self.K, self.T = np.array(self.pose["K"]), np.array(self.pose["T_wc"])
        self.D_pred = np.load(pdir / "depth.npy").astype(np.float32)
        self.known = np.load(pdir / "known.npy").astype(bool)
        mono = depthlib.load_mono(scene.depth / "photos" / f"{name}.npz", (self.W, self.H))
        self.D_photo, self.align = depthlib.align_to_render(self.D_pred, mono, self.known)
        self.good = self.pose["start"] == "sfm" and self.align["rel_err_median"] <= depthlib.MAX_ALIGN_REL_ERR
        self.rgb = cv2.imread(str(scene.photo_image(name)))[..., ::-1].astype(np.float32) / 255

    def lift(self, mask: np.ndarray, D: np.ndarray) -> np.ndarray:
        return geo.lift(mask & self.known, D, self.K, self.T)

    def project(self, P: np.ndarray):
        return geo.project(P, self.K, self.T)

    def can_see(self, centre) -> bool:
        """The photo shows the place `centre`: it projects inside the image onto a known pixel whose surface is not
        in front of it."""
        u, v, z = self.project(np.atleast_2d(centre))
        u, v, z = int(u[0]), int(v[0]), float(z[0])
        b = VIEW_BORDER_PX
        if z < 0.1 or not (b <= u < self.W - b and b <= v < self.H - b) or not self.known[v, u]:
            return False
        tol = max(0.08, 0.06 * z)
        return bool(self.D_photo[v, u] > 0 and self.D_photo[v, u] > z - tol)
