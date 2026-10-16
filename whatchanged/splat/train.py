"""Train a 3D Gaussian splat (gsplat 1.5.3) of the reference walk in the world frame.

    python -m whatchanged.splat.train --scene config1

Recipe: 3DGS with gsplat's DefaultStrategy, initialised from the well-triangulated SfM points; SH degree 0 -> 3 (one
band per 1000 steps); loss 0.8 L1 + 0.2 D-SSIM; antialiased rasterisation, because photos are rendered at other
resolutions and focal lengths; packed mode for memory. A per-frame colour affine (3x4, identity start, small L2 pull)
absorbs the phone's auto-exposure and white-balance drift so the splat keeps one mean exposure; held-out frames are
scored with the affine of the temporally nearest training frame and with none. Densification stops at half the steps,
and the growth threshold is raised by 1.5x whenever the count passes max_gaussians. A checkpoint every 5000 steps lets
a rerun resume. Outputs splat/: splat.npz (+ per-frame affines), eval.json, eval_test_pairs.jpg, train_log.json.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from whatchanged.common.gpu import gpu_stage
from whatchanged.common.io import log, write_json
from whatchanged.common.scene import Scene
from whatchanged.splat import gaussians as gs
from whatchanged.splat.settings import TrainConfig, load_settings


def ssim(x, y):
    import torch
    import torch.nn.functional as F

    C1, C2 = 0.01**2, 0.03**2
    g = torch.exp(-((torch.arange(11, device=x.device) - 5.0) ** 2) / (2 * 1.5**2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)

    def f(t):
        return F.conv2d(t, w, padding=5, groups=3)

    mx, my = f(x), f(y)
    sx, sy, sxy = f(x * x) - mx**2, f(y * y) - my**2, f(x * y) - mx * my
    return (((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx**2 + my**2 + C1) * (sx + sy + C2))).mean()


def load_views(scene: Scene) -> list[dict]:
    P = scene.poses()
    views = []
    for f in P["frames"]:
        im = cv2.imread(str(scene.frame_image(f["name"])))[..., ::-1]
        views.append(
            dict(
                name=f["name"],
                K=np.array(P["K"]),
                T_wc=np.array(f["T_wc"]),
                split=f["split"],
                t=f["t"],
                img=np.ascontiguousarray(im),
            )
        )
    return views


def init_points(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    s = np.load(scene.sfm_points)
    ok = (s["err"] < 2.0) & (s["track"] >= 3)
    return s["xyz"][ok].astype(np.float32), s["rgb"][ok].astype(np.float32) / 255


def train(cfg: TrainConfig, scene: Scene, views: list[dict], watchdog) -> None:
    import torch
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    from scipy.spatial import cKDTree

    dev = "cuda"
    out = scene.splat_dir
    torch.manual_seed(0)
    tr = [v for v in views if v["split"] == "train"]
    te = [v for v in views if v["split"] == "test"]
    imgs = [torch.as_tensor(v["img"], device=dev) for v in tr]
    Ks = [torch.as_tensor(v["K"], dtype=torch.float32, device=dev) for v in tr]
    vms = [torch.as_tensor(np.linalg.inv(v["T_wc"]), dtype=torch.float32, device=dev) for v in tr]
    pts, col = init_points(scene)
    centres = np.array([v["T_wc"][:3, 3] for v in views])
    scene_scale = float(np.linalg.norm(centres - centres.mean(0), axis=1).max() * 1.1)
    dd, _ = cKDTree(pts).query(pts, k=4)
    sc0 = np.log(np.clip(dd[:, 1:].mean(1), 1e-4, 0.05))[:, None].repeat(3, 1)
    N = len(pts)
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.tensor(pts, device=dev)),
            "scales": torch.nn.Parameter(torch.tensor(sc0, dtype=torch.float32, device=dev)),
            "quats": torch.nn.Parameter(torch.tensor(np.tile([1.0, 0, 0, 0], (N, 1)), dtype=torch.float32, device=dev)),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=dev))),
            "sh0": torch.nn.Parameter(torch.tensor((col - 0.5) / gs.C0, device=dev)[:, None, :]),
            "shN": torch.nn.Parameter(torch.zeros((N, 15, 3), device=dev)),
        }
    )
    lrs = dict(means=1.6e-4 * scene_scale, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)

    def optimisers(step: int):
        decay = 0.01 ** (step / cfg.iters)
        opt = {
            k: torch.optim.Adam(
                [{"params": params[k], "lr": lr * (decay if k == "means" else 1), "name": k}], eps=1e-15
            )
            for k, lr in lrs.items()
        }
        return opt, torch.optim.lr_scheduler.ExponentialLR(opt["means"], gamma=0.01 ** (1.0 / cfg.iters))

    aff = torch.nn.Parameter(torch.zeros((len(tr), 3, 4), device=dev))
    opt_aff = torch.optim.Adam([aff], lr=1e-3)
    strat = DefaultStrategy(
        refine_stop_iter=int(cfg.iters * 0.5),
        reset_every=cfg.reset_every,
        refine_every=cfg.refine_every,
        verbose=False,
        grow_grad2d=cfg.grow_grad2d,
        absgrad=False,
    )
    ck = out / "ckpt.pt"
    step0 = 0
    if ck.exists():
        c = torch.load(ck, map_location=dev)
        for k in params:
            params[k] = torch.nn.Parameter(c["params"][k].to(dev))
        aff.data = c["aff"].to(dev)
        step0 = c["step"]
        log(f"resumed at step {step0} (densification state starts afresh)")
    opt, sched = optimisers(step0)
    strat.check_sanity(params, opt)
    state = strat.initialize_state(scene_scale=scene_scale)
    rng = np.random.default_rng(step0)
    t0 = time.time()
    tlog = []
    eye = torch.eye(3, 4, device=dev)
    for step in range(step0, cfg.iters):
        i = int(rng.integers(len(tr)))
        gt = imgs[i].float() / 255
        H, W = gt.shape[:2]
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        img, _, info = rasterization(
            params["means"],
            params["quats"],
            torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]),
            colors,
            vms[i][None],
            Ks[i][None],
            W,
            H,
            sh_degree=min(step // 1000, 3),
            packed=True,
            near_plane=gs.NEAR,
            far_plane=gs.FAR,
            rasterize_mode="antialiased",
        )
        strat.step_pre_backward(params, opt, state, step, info)
        M = eye + aff[i]
        out_c = img[0] @ M[:, :3].T + M[:, 3]
        l1 = (out_c - gt).abs().mean()
        d_ssim = 1 - ssim(out_c.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        loss = (1 - cfg.ssim_weight) * l1 + cfg.ssim_weight * d_ssim + cfg.affine_l2 * (aff[i] ** 2).mean()
        loss.backward()
        strat.step_post_backward(params, opt, state, step, info, packed=True)
        for o in opt.values():
            o.step()
            o.zero_grad(set_to_none=True)
        opt_aff.step()
        opt_aff.zero_grad(set_to_none=True)
        sched.step()
        if step < strat.refine_stop_iter and len(params["means"]) > cfg.max_gaussians and step % 500 == 0:
            strat.grow_grad2d *= 1.5
            log(f"step {step}: {len(params['means'])} Gaussians, growth threshold {strat.grow_grad2d:.2e}")
        if step % 500 == 0 or step == cfg.iters - 1:
            psnr = float(-10 * torch.log10(((out_c - gt) ** 2).mean()))
            tlog.append(
                dict(
                    step=step,
                    loss=round(float(loss), 4),
                    psnr=round(psnr, 2),
                    n=int(len(params["means"])),
                    s=round(time.time() - t0, 1),
                    peak_mib=int(torch.cuda.max_memory_reserved() / 2**20),
                )
            )
            log("train", tlog[-1])
        if (step + 1) % cfg.checkpoint_every == 0 and step + 1 < cfg.iters:
            torch.save(dict(params={k: v.detach() for k, v in params.items()}, aff=aff.detach(), step=step + 1), ck)
    train_s = time.time() - t0
    sp = dict(
        means=params["means"].detach().cpu().numpy(),
        quats=torch.nn.functional.normalize(params["quats"], dim=-1).detach().cpu().numpy(),
        scales=torch.exp(params["scales"]).detach().cpu().numpy(),
        opacities=torch.sigmoid(params["opacities"]).detach().cpu().numpy(),
        sh=torch.cat([params["sh0"], params["shN"]], 1).detach().cpu().numpy().astype(np.float32),
    )
    affine = (eye + aff).detach().cpu().numpy()
    np.savez(out / "splat.npz", **sp, affine=affine, affine_names=np.array([v["name"] for v in tr]))
    evaluate(
        sp,
        tr,
        te,
        affine,
        out,
        dict(
            n_train=len(tr),
            n_test=len(te),
            n_gaussians=int(len(sp["means"])),
            iters=cfg.iters,
            train_s=round(train_s, 1),
            scene_scale=scene_scale,
            torch_peak_reserved_mib=int(torch.cuda.max_memory_reserved() / 2**20),
            gpu_peak_used_all_mib=watchdog.peak_used,
        ),
    )
    write_json(tlog, out / "train_log.json")
    if ck.exists():
        ck.unlink()


def evaluate(sp: dict, tr: list, te: list, affine: np.ndarray, out: Path, summary: dict) -> None:
    """PSNR / SSIM on held-out frames, with the colour affine of the temporally nearest training frame and without."""
    import torch

    tsp = gs.to_torch(sp)
    t_tr = np.array([v["t"] for v in tr])
    rows, sheet = [], []
    for v in te:
        r = gs.render(tsp, v["K"], v["T_wc"], v["img"].shape[1], v["img"].shape[0], depth=False)["rgb"]
        gt = v["img"].astype(np.float32) / 255
        j = int(np.argmin(np.abs(t_tr - v["t"])))
        rc = np.clip(r @ affine[j][:, :3].T + affine[j][:, 3], 0, 1)
        with torch.no_grad():
            s = float(
                ssim(torch.tensor(rc).permute(2, 0, 1)[None].float(), torch.tensor(gt).permute(2, 0, 1)[None].float())
            )
        rows.append(
            dict(
                name=v["name"],
                psnr_identity=round(float(-10 * np.log10(((r - gt) ** 2).mean())), 2),
                psnr_neighbour_affine=round(float(-10 * np.log10(((rc - gt) ** 2).mean())), 2),
                ssim_neighbour_affine=round(s, 4),
            )
        )
        if len(sheet) < 6:
            sheet.append(np.concatenate([v["img"], (rc * 255).astype(np.uint8)], 1))
    if sheet:
        cv2.imwrite(
            str(out / "eval_test_pairs.jpg"), np.concatenate(sheet, 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
    if rows:
        summary.update(
            psnr_test_neighbour_affine_mean=float(np.mean([r["psnr_neighbour_affine"] for r in rows])),
            psnr_test_identity_mean=float(np.mean([r["psnr_identity"] for r in rows])),
            ssim_test_mean=float(np.mean([r["ssim_neighbour_affine"] for r in rows])),
        )
    write_json(dict(summary=summary, per_frame=rows), out / "eval.json")
    log("eval", summary)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--iters", type=int, help="override the configured number of steps")
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()
    cfg = load_settings(args.config).train
    if args.iters:
        cfg.iters = args.iters
    scene = Scene(args.scene)
    scene.splat_dir.mkdir(parents=True, exist_ok=True)
    views = load_views(scene)
    log(f"{len(views)} views ({sum(v['split'] == 'test' for v in views)} held out)")
    with gpu_stage("splat_train", need_mib=cfg.need_mib) as watchdog:
        train(cfg, scene, views, watchdog)


if __name__ == "__main__":
    main()
