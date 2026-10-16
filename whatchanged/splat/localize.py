"""Refine each query photo's pose against the splat and render the predicted view: what the reference says the photo
should show.

    python -m whatchanged.splat.localize --scene config1

1. Start pose: the photo's registration in the joint SfM model (photos that did not register are skipped).
2. Render-and-compare: a camera-frame increment of the pose (6 DoF) is fitted with Adam at 240x180 then 480x360
   (images blurred with sigma 2 then 1). The loss is Geman-McClure on the colour residual after a per-photo 3x4 colour
   affine (solved in closed form on the inliers each step) plus half the same on image gradients. Pixels where the
   splat has no data (alpha < 0.9) or whose residual is more than 3 sigma (median absolute deviation) above the median
   are masked out, since those may be real changes. The focal stays at the bundle-adjusted SfM value because
   gsplat 1.5.3 has no gradient for K. The refined pose is kept only if the median inlier residual improves.
3. Predicted view at the photo's resolution: rgb.jpg (colour-matched to the photo), rgb_raw.jpg (splat colour),
   depth.npy (float16 z-depth, 0 = no data), cover.npy (accumulated opacity) and pose.json.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from whatchanged.common.geometry import pose_delta
from whatchanged.common.gpu import gpu_stage
from whatchanged.common.io import log, read_json, write_json
from whatchanged.common.scene import Scene
from whatchanged.splat import gaussians as gs
from whatchanged.splat.settings import load_settings

COVER_ALPHA = 0.9


def rodrigues(w):
    th = torch.sqrt((w * w).sum() + 1e-12)
    k = w / th
    K = torch.zeros((3, 3), dtype=w.dtype, device=w.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, dtype=w.dtype, device=w.device) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


def geman_mcclure(r, s):
    r2 = r * r
    return r2 / (r2 + s * s)


def grads(x):
    return (x[:, 1:] - x[:, :-1])[:-1], (x[1:] - x[:-1])[:, :-1]


def blur(x, s: float):
    if s <= 0:
        return x
    dev = x.device
    k = int(2 * round(3 * s) + 1)
    g = torch.exp(-((torch.arange(k, device=dev) - k // 2) ** 2) / (2 * s * s))
    g = (g / g.sum()).float()
    y = x.permute(2, 0, 1)[None]
    pad = torch.nn.functional.pad
    y = torch.nn.functional.conv2d(
        pad(y, (k // 2, k // 2, 0, 0), mode="replicate"), g[None, None, None].repeat(3, 1, 1, 1), groups=3
    )
    y = torch.nn.functional.conv2d(
        pad(y, (0, 0, k // 2, k // 2), mode="replicate"), g[None, None, :, None].repeat(3, 1, 1, 1), groups=3
    )
    return y[0].permute(1, 2, 0)


def fit_affine(r, gt, mask):
    X = torch.cat([r[mask], torch.ones_like(r[mask][:, :1])], 1)
    return torch.linalg.lstsq(X, gt[mask]).solution


def apply_affine(r, A):
    return torch.cat([r, torch.ones_like(r[..., :1])], -1) @ A


def refine(tsp: dict, img: np.ndarray, K0: np.ndarray, T0: np.ndarray, cfg):
    """img: H x W x 3 float photo. Returns refined T_wc and the photometric fit before and after."""
    from gsplat import rasterization

    dev = "cuda"
    H, W = img.shape[:2]
    T0t = torch.tensor(T0, dtype=torch.float64, device=dev)
    w = torch.zeros(3, dtype=torch.float64, device=dev, requires_grad=True)
    t = torch.zeros(3, dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [w], "lr": cfg.lr_rotation}, {"params": [t], "lr": cfg.lr_translation}])

    def render(sc: float):
        Ws, Hs = int(W * sc), int(H * sc)
        top = torch.cat([rodrigues(w), t[:, None]], 1)
        Twc = T0t @ torch.cat([top, torch.tensor([[0, 0, 0, 1.0]], dtype=torch.float64, device=dev)], 0)
        Rw, tw = Twc[:3, :3], Twc[:3, 3]
        vm = torch.eye(4, dtype=torch.float64, device=dev)
        vm[:3, :3] = Rw.T
        vm[:3, 3] = -Rw.T @ tw
        Kt = torch.tensor(K0, dtype=torch.float64, device=dev) * torch.tensor(
            [[sc], [sc], [1.0]], dtype=torch.float64, device=dev
        )
        out, alpha, _ = rasterization(
            tsp["means"],
            tsp["quats"],
            tsp["scales"],
            tsp["opacities"],
            tsp["sh"],
            vm.float()[None],
            Kt.float()[None],
            Ws,
            Hs,
            sh_degree=3,
            packed=True,
            near_plane=gs.NEAR,
            far_plane=gs.FAR,
            rasterize_mode="antialiased",
        )
        return out[0], alpha[0, ..., 0], Twc

    def target(sc: float, s: float):
        im = cv2.resize(img, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
        if s > 0:
            im = cv2.GaussianBlur(im, (0, 0), s)
        return torch.tensor(im, dtype=torch.float32, device=dev)

    def inliers(res, cov):
        rr = res[cov]
        med = rr.median()
        mad = (rr - med).abs().median() * 1.4826 + 1e-4
        return cov & (res < med + 3 * mad), mad

    def evaluate(sc: float = 0.5) -> dict:
        with torch.no_grad():
            r, a, _ = render(sc)
            gt = target(sc, 0)
            cov = a > COVER_ALPHA
            rc = apply_affine(r, fit_affine(r, gt, cov))
            res = (rc - gt).abs().mean(-1)
            inl, _ = inliers(res, cov)
            mse = ((rc - gt) ** 2).mean(-1)
            return dict(
                med_abs_255=float(res[inl].median() * 255),
                inlier_frac=float(inl.float().mean()),
                cover_frac=float(cov.float().mean()),
                psnr_inliers=float(-10 * torch.log10(mse[inl].mean())),
                psnr_covered=float(-10 * torch.log10(mse[cov].mean())),
            )

    before = evaluate()
    for sc, n, s in zip(cfg.scales, cfg.steps, cfg.blur_sigmas):
        gt = target(sc, s)
        for _ in range(n):
            r, a, _ = render(sc)
            rb = blur(r, s)
            cov = a.detach() > COVER_ALPHA
            with torch.no_grad():
                A = fit_affine(rb, gt, cov)
            rc = apply_affine(rb, A)
            res = (rc - gt).abs().mean(-1)
            with torch.no_grad():
                m, mad = inliers(res, cov)
            gxr, gyr = grads(rc.mean(-1))
            gxg, gyg = grads(gt.mean(-1))
            mg = m[:-1, :-1]
            sgm = float(3 * mad)
            loss = geman_mcclure(res[m], sgm).mean() + 0.5 * (
                geman_mcclure((gxr - gxg)[mg], sgm).mean() + geman_mcclure((gyr - gyg)[mg], sgm).mean()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    after = evaluate()
    with torch.no_grad():
        _, _, Twc = render(1.0)
    return Twc.cpu().numpy(), dict(before=before, after=after)


def predicted_view(tsp: dict, K: np.ndarray, T: np.ndarray, img: np.ndarray, out: Path) -> dict:
    H, W = img.shape[:2]
    r = gs.render(tsp, K, T, W, H)
    rgb_raw = gs.unpremultiply(r)
    cov = r["alpha"] > COVER_ALPHA
    A, *_ = np.linalg.lstsq(np.c_[rgb_raw[cov], np.ones(cov.sum())], img[cov], rcond=None)
    rgb = np.clip(np.concatenate([rgb_raw, np.ones_like(rgb_raw[..., :1])], -1) @ A, 0, 1)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "rgb.jpg"), (rgb[..., ::-1] * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(
        str(out / "rgb_raw.jpg"), (rgb_raw[..., ::-1] * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    np.save(out / "depth.npy", r["depth"].astype(np.float16))
    np.save(out / "cover.npy", r["alpha"].astype(np.float16))
    return dict(colour_affine=A.round(5).tolist(), cover_frac=float((r["alpha"] > 0.5).mean()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--photos", nargs="*", help="only these photos (default: all)")
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()
    cfg = load_settings(args.config).localize
    scene = Scene(args.scene)
    registered = read_json(scene.photos_sfm_json)["photos"]
    names = args.photos or sorted(registered)
    sp = gs.load(scene.splat)
    summary = {}
    t_all = time.time()
    with gpu_stage("localize", need_mib=cfg.need_mib) as watchdog:
        tsp = gs.to_torch(sp)
        for name in names:
            t0 = time.time()
            reg = registered.get(name)
            if reg is None:
                log(f"{name}: not registered in the SfM model, skipped")
                continue
            img = cv2.imread(str(scene.photo_image(name)))[..., ::-1].astype(np.float32) / 255
            H, W = img.shape[:2]
            T0, K = np.array(reg["T_wc"]), np.array(reg["K"])
            T, fit = refine(tsp, img, K, T0, cfg)
            kept = fit["after"]["med_abs_255"] < fit["before"]["med_abs_255"]
            if not kept:
                T = T0
            view = predicted_view(tsp, K, T, img, scene.predicted / name)
            final = fit["after" if kept else "before"]
            rec = dict(
                photo=str(scene.photo_image(name)),
                size=[W, H],
                K=K.round(4).tolist(),
                T_wc=T.round(7).tolist(),
                start="sfm",
                refined_kept=bool(kept),
                T_wc_start=T0.round(7).tolist(),
                delta_vs_start_cm_deg=pose_delta(T0, T),
                sfm_reproj_px_median=reg["reproj_px_median"],
                sfm_n_points=reg["n_points"],
                photometric_before=fit["before"],
                photometric_after=fit["after"],
                **view,
            )
            write_json(rec, scene.predicted / name / "pose.json")
            summary[name] = dict(
                refined_kept=bool(kept),
                delta_vs_start_cm_deg=rec["delta_vs_start_cm_deg"],
                med_abs_255=final["med_abs_255"],
                inlier_frac=final["inlier_frac"],
                psnr_inliers=final["psnr_inliers"],
                cover_frac=view["cover_frac"],
                s=round(time.time() - t0, 1),
            )
            log(name, summary[name])
        peak = int(torch.cuda.max_memory_reserved() / 2**20)
    write_json(
        dict(
            photos=summary,
            wall_s=round(time.time() - t_all, 1),
            torch_peak_reserved_mib=peak,
            gpu_peak_used_all_mib=watchdog.peak_used,
        ),
        scene.predicted / "localize_summary.json",
    )


if __name__ == "__main__":
    main()
