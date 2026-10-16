"""Change checker: frozen DINOv2 PCA features of the reference render and the real image at the same pose, plus a depth
difference, through a light decoder -> 3 classes per pixel at 1/4 resolution: 0 unchanged, 1 appeared (in the photo, not
in the render), 2 missing (in the render, not in the photo).

Inputs per sample:
  fr, fq  (128, h, w) DINOv2 PCA patch features of render and real image
  low     (9, H/4, W/4): render RGB, real RGB, signed relative depth difference (real - render) / render clipped to
          +-0.5 and scaled by 4, depth-valid mask, known mask
The feature branch takes [fr, fq, |fr - fq|, fr * fq] at the patch grid (1x1 conv 192, then 2 x (3x3 conv, GroupNorm,
GELU)), is upsampled to 1/4 resolution and joined with a small low-level branch; two more conv blocks and a 1x1 conv
give the logits.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UNCHANGED, APPEARED, MISSING = 0, 1, 2


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())


class Checker(nn.Module):
    def __init__(self, c: int = 128, w: int = 192):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv2d(4 * c, w, 1), nn.GroupNorm(8, w), nn.GELU())
        self.feat = nn.Sequential(block(w, w), block(w, w))
        self.low = nn.Sequential(block(9, 32), block(32, 32))
        self.dec = nn.Sequential(block(w + 32, 128), block(128, 64), nn.Conv2d(64, 3, 1))

    def forward(self, fr, fq, low):
        x = torch.cat([fr, fq, (fr - fq).abs(), fr * fq], 1)
        x = self.feat(self.inp(x))
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.dec(torch.cat([x, self.low(low)], 1))


def make_low(ren, real, d_render, d_real, known) -> np.ndarray:
    """uint8 H x W x 3 images, depths and known mask, all at 1/4 resolution -> (9, H, W) float32."""
    valid = (d_render > 0) & (d_real > 0)
    rel = np.where(valid, (d_real - d_render) / np.maximum(d_render, 1e-3), 0)
    rel = np.clip(rel, -0.5, 0.5) * 4
    return np.concatenate(
        [ren.transpose(2, 0, 1) / 255.0, real.transpose(2, 0, 1) / 255.0, rel[None], valid[None], known[None]], 0
    ).astype(np.float32)


def quarter_inputs(render_rgb, real_rgb, d_render, d_real, known) -> np.ndarray:
    """make_low on full-resolution inputs: images area-downsampled, depths and mask nearest."""
    H, W = render_rgb.shape[:2]
    size = (W // 4, H // 4)

    def rs(x, interp=cv2.INTER_AREA):
        return cv2.resize(x, size, interpolation=interp)

    nearest = cv2.INTER_NEAREST
    return make_low(
        rs(render_rgb),
        rs(real_rgb),
        rs(d_render.astype(np.float32), nearest),
        rs(d_real.astype(np.float32), nearest),
        rs(known.astype(np.uint8), nearest).astype(bool),
    )


def probabilities(model: Checker, fr: np.ndarray, fq: np.ndarray, low: np.ndarray, device: str = "cpu") -> np.ndarray:
    """(3, H/4, W/4) class probabilities for one sample."""
    with torch.no_grad():
        logits = model(
            torch.as_tensor(fr).float()[None].to(device),
            torch.as_tensor(fq).float()[None].to(device),
            torch.as_tensor(low)[None].to(device),
        )
        return torch.softmax(logits, 1)[0].cpu().numpy()


def load(path, device: str = "cpu") -> Checker:
    model = Checker().to(device)
    model.load_state_dict(torch.load(path, map_location=device)["model"])
    return model.eval()
