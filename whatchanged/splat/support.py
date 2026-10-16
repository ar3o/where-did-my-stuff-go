"""Per-pixel view support: how many reference cameras actually saw the surface point under each pixel. Accumulated
opacity cannot tell an observed surface from floaters in a region the walk barely saw; this can.

    python -m whatchanged.splat.support --scene config1

1. Reference depth: the splat's expected depth from every registered frame at 480x270 (cached in splat/ref_depth.npz).
2. For a view with a predicted depth map, each pixel (2 px grid) is back-projected. A reference camera counts when the
   point is inside its frustum (5 px margin) at 0.15-6 m, lies on that camera's own visible surface
   (|z - D_ref| <= 2 cm + 3 % z, so a floater in front of the surface or a point hidden behind it does not count),
   and the two viewing rays differ by at most 60 degrees.
Outputs: predicted/<photo>/support.npy (uint8 count), known.npy (count >= 3); frame_known/<frame>.npy for every walk
frame with that frame's own camera left out (the checker's training pairs and the object fingerprints use these).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from whatchanged.common.gpu import gpu_stage
from whatchanged.common.io import log, read_json, save_npz, write_json
from whatchanged.common.scene import Scene
from whatchanged.splat import gaussians as gs
from whatchanged.splat.settings import load_settings

REF_W, REF_H = 480, 270


def reference_depths(scene: Scene, tsp: dict | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scene.ref_depth.exists():
        z = np.load(scene.ref_depth)
        return z["K"], z["T"], z["D"].astype(np.float32)
    P = scene.poses()
    K = np.array(P["K"], float)
    K[:2] *= REF_W / P["W"]
    Ts = np.array([f["T_wc"] for f in P["frames"]], float)
    D = np.zeros((len(Ts), REF_H, REF_W), np.float16)
    for i, T in enumerate(Ts):
        D[i] = gs.render(tsp, K, T, REF_W, REF_H)["depth"]
    save_npz(scene.ref_depth, K=K, T=Ts, D=D)
    return K, Ts, D.astype(np.float32)


def support_map(depth: np.ndarray, K, T, Kr, Tr, Dr, cfg, step: int = 2) -> np.ndarray:
    H, W = depth.shape
    v, u = np.mgrid[0:H:step, 0:W:step]
    z = depth[::step, ::step].astype(np.float32)
    ok = z > 0
    Pc = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1)[ok]
    Pw = Pc @ T[:3, :3].T + T[:3, 3]
    ray_q = Pw - T[:3, 3]
    ray_q /= np.linalg.norm(ray_q, axis=1, keepdims=True)
    cnt = np.zeros(len(Pw), np.int32)
    Hr, Wr = Dr.shape[1:]
    cos_max = np.cos(np.radians(cfg.max_angle_deg))
    for i in range(len(Tr)):
        T_cw = np.linalg.inv(Tr[i])
        pc = Pw @ T_cw[:3, :3].T + T_cw[:3, 3]
        zz = pc[:, 2]
        good = (zz > cfg.min_range_m) & (zz < cfg.max_range_m)
        uu = Kr[0, 0] * pc[:, 0] / np.maximum(zz, 1e-6) + Kr[0, 2]
        vv = Kr[1, 1] * pc[:, 1] / np.maximum(zz, 1e-6) + Kr[1, 2]
        good &= (uu >= 5) & (vv >= 5) & (uu < Wr - 5) & (vv < Hr - 5)
        idx = np.flatnonzero(good)
        if not len(idx):
            continue
        d = Dr[i][vv[idx].astype(int), uu[idx].astype(int)]
        idx = idx[(d > 0) & (np.abs(zz[idx] - d) <= cfg.tol_abs_m + cfg.tol_rel * zz[idx])]
        ray_r = Pw[idx] - Tr[i][:3, 3]
        ray_r /= np.linalg.norm(ray_r, axis=1, keepdims=True)
        cnt[idx[(ray_r * ray_q[idx]).sum(1) >= cos_max]] += 1
    S = np.zeros(z.shape, np.int32)
    S[ok] = cnt
    S = cv2.resize(np.clip(S, 0, 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    S[depth <= 0] = 0
    return S


def nearest_camera(T: np.ndarray, Tr: np.ndarray) -> int:
    return int(np.argmin([np.linalg.norm(x[:3, 3] - T[:3, 3]) + np.linalg.norm(x[:3, :3] - T[:3, :3]) for x in Tr]))


def photo_support(scene: Scene, Kr, Tr, Dr, cfg) -> dict:
    summary = {}
    for pdir in sorted(p for p in scene.predicted.iterdir() if (p / "pose.json").exists()):
        pose = read_json(pdir / "pose.json")
        depth = np.load(pdir / "depth.npy").astype(np.float32)
        S = support_map(depth, np.array(pose["K"]), np.array(pose["T_wc"]), Kr, Tr, Dr, cfg)
        known = S >= cfg.min_views
        np.save(pdir / "support.npy", S)
        np.save(pdir / "known.npy", known)
        cover = np.load(pdir / "cover.npy").astype(np.float32) > 0.5
        summary[pdir.name] = dict(
            known_frac=round(float(known.mean()), 4),
            cover_frac=round(float(cover.mean()), 4),
            covered_but_unknown_frac=round(float((cover & ~known).mean()), 4),
        )
    return summary


def frame_support(scene: Scene, tsp: dict, Kr, Tr, Dr, cfg) -> None:
    P = scene.poses()
    K, W, H = np.array(P["K"]), P["W"], P["H"]
    scene.frame_known.mkdir(parents=True, exist_ok=True)
    for f in P["frames"]:
        out = scene.frame_known / f"{f['name']}.npy"
        if out.exists():
            continue
        T = np.array(f["T_wc"])
        others = np.arange(len(Tr)) != nearest_camera(T, Tr)
        depth = gs.render(tsp, K, T, W, H)["depth"]
        np.save(out, support_map(depth, K, T, Kr, Tr[others], Dr[others], cfg) >= cfg.min_views)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()
    cfg = load_settings(args.config).support
    scene = Scene(args.scene)
    t0 = time.time()
    with gpu_stage("support", need_mib=cfg.need_mib):
        tsp = gs.to_torch(gs.load(scene.splat))
        Kr, Tr, Dr = reference_depths(scene, tsp)
        frame_support(scene, tsp, Kr, Tr, Dr, cfg)
    summary = photo_support(scene, Kr, Tr, Dr, cfg)
    write_json(
        dict(
            n_reference_cameras=int(len(Tr)), min_views=cfg.min_views, photos=summary, wall_s=round(time.time() - t0, 1)
        ),
        scene.predicted / "support_summary.json",
    )
    log(f"support: {len(Tr)} reference cameras, {len(summary)} photos, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
