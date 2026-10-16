"""Gaussian splat container and gsplat rendering: colour, expected depth, per-object id maps, and rigid motions of
Gaussian subsets with spherical-harmonics rotation.

Splat dict (numpy, world frame): means (N, 3), quats (N, 4) wxyz unit, scales (N, 3) metres, opacities (N,),
sh (N, 16, 3) in gsplat's convention (DC = (rgb - 0.5) / C0).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from whatchanged.common.geometry import quat_mul_wxyz, quat_wxyz_from_R

KEYS = ("means", "quats", "scales", "opacities", "sh")
C0 = 0.28209479177387814  # SH band-0 constant
NEAR, FAR = 0.02, 100.0
ID_CHUNK = 32
ID_MIN_WEIGHT = 0.5  # a pixel belongs to an object when it carries at least half of the pixel's opacity
ID_MIN_ALPHA = 0.3


def load(path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def core(sp: dict) -> dict:
    return {k: sp[k] for k in KEYS}


def to_torch(sp: dict, device: str = "cuda") -> dict:
    return {k: torch.as_tensor(np.ascontiguousarray(sp[k]), dtype=torch.float32, device=device) for k in KEYS}


def rgb_of(sp: dict) -> np.ndarray:
    return np.clip(sp["sh"][:, 0] * C0 + 0.5, 0, 1)


@torch.no_grad()
def render(tsp: dict, K, T_wc, W: int, H: int, extra=None, sh_degree: int = 3, depth: bool = True) -> dict:
    """rgb (H, W, 3) premultiplied in [0, 1], alpha (H, W), depth (expected z, 0 where alpha <= 0.5), and `extra`
    (H, W, C) accumulated per-Gaussian features (N, C) if given."""
    from gsplat import rasterization

    dev = tsp["means"].device
    vm = torch.as_tensor(np.linalg.inv(T_wc), dtype=torch.float32, device=dev)[None]
    Kt = torch.as_tensor(K, dtype=torch.float32, device=dev)[None]
    img, alpha, _ = rasterization(
        tsp["means"],
        tsp["quats"],
        tsp["scales"],
        tsp["opacities"],
        tsp["sh"],
        vm,
        Kt,
        W,
        H,
        sh_degree=sh_degree,
        render_mode="RGB+ED" if depth else "RGB",
        rasterize_mode="antialiased",
        near_plane=NEAR,
        far_plane=FAR,
        packed=True,
    )
    out = dict(rgb=img[0, ..., :3].clamp(0, 1).cpu().numpy(), alpha=alpha[0, ..., 0].cpu().numpy())
    if depth:
        d = img[0, ..., 3].cpu().numpy()
        out["depth"] = np.where(out["alpha"] > 0.5, d, 0.0).astype(np.float32)
    if extra is not None:
        e, _, _ = rasterization(
            tsp["means"],
            tsp["quats"],
            tsp["scales"],
            tsp["opacities"],
            extra,
            vm,
            Kt,
            W,
            H,
            sh_degree=None,
            render_mode="RGB",
            rasterize_mode="antialiased",
            near_plane=NEAR,
            far_plane=FAR,
            packed=True,
            channel_chunk=64,
        )
        out["extra"] = e[0].cpu().numpy()
    return out


def unpremultiply(r: dict, min_alpha: float = 0.05) -> np.ndarray:
    rgb = r["rgb"] / np.maximum(r["alpha"], 1e-3)[..., None]
    return np.clip(np.where(r["alpha"][..., None] > min_alpha, rgb, 0), 0, 1)


def id_features(gid: np.ndarray, ids, device: str = "cuda") -> torch.Tensor:
    """One-hot (N, len(ids)) features of object ids; column j is ids[j]."""
    lut = {int(g): j for j, g in enumerate(ids)}
    col = torch.as_tensor(np.array([lut.get(int(g), -1) for g in gid]), device=device)
    F = torch.zeros((len(gid), len(ids)), device=device)
    ok = col >= 0
    F[torch.nonzero(ok)[:, 0], col[ok]] = 1.0
    return F


def render_ids(tsp: dict, gid: np.ndarray, K, T_wc, W: int, H: int, ids) -> np.ndarray:
    """Object-id map (int64, 0 = none): the object with the largest blending weight if that weight is at least half the
    pixel's opacity. Ids are rendered 32 at a time to bound GPU memory."""
    best_w = np.zeros((H, W), np.float32)
    best_i = np.zeros((H, W), np.int64)
    alpha = np.zeros((H, W), np.float32)
    obj = [i for i in ids if i != 0]
    for c0 in range(0, len(obj), ID_CHUNK):
        sub = obj[c0 : c0 + ID_CHUNK]
        F = id_features(gid, sub, tsp["means"].device)
        r = render(tsp, K, T_wc, W, H, extra=F, depth=False)
        alpha = r["alpha"]
        w = r["extra"]
        j = w.argmax(-1)
        bw = np.take_along_axis(w, j[..., None], -1)[..., 0]
        upd = bw > best_w
        best_w[upd] = bw[upd]
        best_i[upd] = np.asarray(sub)[j[upd]]
        del F
    return np.where((best_w >= ID_MIN_WEIGHT * np.maximum(alpha, 1e-6)) & (alpha >= ID_MIN_ALPHA), best_i, 0)


def render_view(tsp: dict, gid: np.ndarray, K, T_wc, W: int, H: int, ids) -> dict:
    """One pass: unpremultiplied rgb with low-alpha pixels inpainted, depth, alpha and the id map of `ids`."""
    F = id_features(gid, ids, tsp["means"].device)
    r = render(tsp, K, T_wc, W, H, extra=F)
    alpha = r["alpha"]
    w = r["extra"][..., : len(ids)]
    j = w.argmax(-1)
    best = np.take_along_axis(w, j[..., None], -1)[..., 0]
    inst = np.where((best >= ID_MIN_WEIGHT * np.maximum(alpha, 1e-6)) & (alpha >= ID_MIN_ALPHA), np.asarray(ids)[j], 0)
    rgb = r["rgb"] / np.maximum(alpha, 1e-3)[..., None]
    low = alpha < 0.5
    rgb = np.clip(np.where(low[..., None], 0, rgb), 0, 1)
    if low.any():
        rgb8 = (rgb * 255).astype(np.uint8)
        rgb = cv2.inpaint(rgb8, low.astype(np.uint8), 5, cv2.INPAINT_TELEA).astype(np.float32) / 255
    return dict(rgb=rgb, depth=r["depth"], inst=inst.astype(np.uint16), alpha=alpha)


# ------------------------------------------------------------------ rigid motions
_SH_BASIS = {}


def _sh_basis(dirs: np.ndarray) -> np.ndarray:
    from gsplat.cuda._torch_impl import _eval_sh_bases_fast

    return _eval_sh_bases_fast(16, torch.as_tensor(dirs, dtype=torch.float64)).numpy()


def sh_rotation(R: np.ndarray) -> np.ndarray:
    """A (16, 16) with coeffs' = A @ coeffs so that colour'(d) = colour(R^T d). Fitted by least squares against gsplat's
    own basis; exact up to precision because each SH band is closed under rotation."""
    if "dirs" not in _SH_BASIS:
        d = np.random.default_rng(0).normal(size=(400, 3))
        _SH_BASIS["dirs"] = d / np.linalg.norm(d, axis=1, keepdims=True)
        _SH_BASIS["Y"] = _sh_basis(_SH_BASIS["dirs"])
    Y, Yr = _SH_BASIS["Y"], _sh_basis(_SH_BASIS["dirs"] @ np.asarray(R))
    A, *_ = np.linalg.lstsq(Y, Yr, rcond=None)
    return A


def transform(sp: dict, mask: np.ndarray, R: np.ndarray, t: np.ndarray) -> dict:
    """Copy of sp with the Gaussians in `mask` moved by x -> R x + t: positions, orientations, view-dependent colour."""
    out = {k: sp[k].copy() for k in KEYS}
    out["means"][mask] = (sp["means"][mask] @ R.T + t).astype(np.float32)
    q = quat_mul_wxyz(quat_wxyz_from_R(R)[None], sp["quats"][mask].astype(np.float64))
    out["quats"][mask] = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
    A = sh_rotation(R)
    out["sh"][mask] = np.einsum("ij,njc->nic", A, sp["sh"][mask].astype(np.float64)).astype(np.float32)
    return out
