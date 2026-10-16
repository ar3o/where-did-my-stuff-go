"""Automatic splat cleanup, no manual editing. Optional: the config1 results use the uncleaned splat
(splat.cleanup.enabled: false).

    python -m whatchanged.splat.cleanup --scene config1

A Gaussian is pruned when any of these hold:
  * opacity < 0.005;
  * largest axis > 1 m (large Gaussians up to about 0.3 m on walls are real surface);
  * outside the room box: 0.1-99.9 % of the SfM points and the cameras, grown by 0.5 m, with the floor allowed down to
    -0.3 m because a glossy floor shows its reflections below z = 0;
  * centre inside fewer than 3 training-camera frusta (in front, inside the image, within 8 m).
The raw splat is kept as splat/splat_raw.npz. Held-out PSNR before and after goes to splat/clean.json.
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
from whatchanged.splat.settings import load_settings


def prune_mask(
    sp: dict, frames: list[dict], K: np.ndarray, W: int, H: int, Q: np.ndarray, cfg
) -> tuple[np.ndarray, dict]:
    cams = np.array([np.array(f["T_wc"])[:3, 3] for f in frames])
    lo = np.minimum(np.percentile(Q, 0.1, 0), cams.min(0)) - cfg.room_margin_m
    hi = np.maximum(np.percentile(Q, 99.9, 0), cams.max(0)) + cfg.room_margin_m
    lo[2] = cfg.floor_reflection_m
    M = sp["means"]
    reasons = dict(
        low_opacity=sp["opacities"] < cfg.min_opacity,
        extreme_scale=sp["scales"].max(1) > cfg.max_scale_m,
        outside_room=~((M >= lo) & (M <= hi)).all(1),
    )
    count = np.zeros(len(M), np.int32)
    for f in frames:
        if f["split"] != "train":
            continue
        T_cw = np.linalg.inv(np.array(f["T_wc"]))
        pc = M @ T_cw[:3, :3].T + T_cw[:3, 3]
        z = pc[:, 2]
        u = K[0, 0] * pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
        v = K[1, 1] * pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
        count += ((z > 0.05) & (z < 8) & (u >= 0) & (u < W) & (v >= 0) & (v < H)).astype(np.int32)
    reasons["low_support"] = count < cfg.min_views
    drop = np.zeros(len(M), bool)
    for r in reasons.values():
        drop |= r
    return ~drop, dict(
        removed={k: int(v.sum()) for k, v in reasons.items()},
        room_lo=lo.round(3).tolist(),
        room_hi=hi.round(3).tolist(),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()
    cfg = load_settings(args.config).cleanup
    scene = Scene(args.scene)
    t0 = time.time()
    raw_path = scene.splat_dir / "splat_raw.npz"
    src = raw_path if raw_path.exists() else scene.splat
    z = dict(np.load(src))
    if src == scene.splat:
        np.savez(raw_path, **z)
    P = scene.poses()
    K, W, H = np.array(P["K"]), P["W"], P["H"]
    pts = np.load(scene.sfm_points)
    Q = pts["xyz"][(pts["track"] >= 3) & (pts["err"] < 2)]
    keep, stats = prune_mask(z, P["frames"], K, W, H, Q, cfg)
    sp = {k: z[k][keep] for k in gs.KEYS}
    np.savez(scene.splat, **sp, **{k: v for k, v in z.items() if k not in gs.KEYS})
    stats.update(n_raw=int(len(keep)), n_clean=int(keep.sum()))
    rows = []
    with gpu_stage("splat_cleanup", need_mib=cfg.need_mib):
        tensors = dict(raw=gs.to_torch(z), clean=gs.to_torch(sp))
        aff, names = z.get("affine"), [str(x) for x in z.get("affine_names", [])]
        t_of = {f["name"]: f["t"] for f in P["frames"]}
        for f in (f for f in P["frames"] if f["split"] == "test"):
            img = cv2.imread(str(scene.frame_image(f["name"])))[..., ::-1].astype(np.float32) / 255
            j = int(np.argmin([abs(t_of.get(n, 0) - f["t"]) for n in names])) if names else None
            row = dict(name=f["name"])
            for tag, t in tensors.items():
                r = gs.render(t, K, np.array(f["T_wc"]), W, H, depth=False)["rgb"]
                if j is not None:
                    r = np.clip(r @ aff[j][:, :3].T + aff[j][:, 3], 0, 1)
                row[f"psnr_{tag}"] = round(float(-10 * np.log10(((r - img) ** 2).mean())), 2)
            rows.append(row)
    if rows:
        stats.update(
            psnr_test_raw=float(np.mean([r["psnr_raw"] for r in rows])),
            psnr_test_clean=float(np.mean([r["psnr_clean"] for r in rows])),
        )
    stats["wall_s"] = round(time.time() - t0, 1)
    write_json(dict(stats, per_frame=rows), scene.splat_dir / "clean.json")
    log(f"cleanup: {stats}")


if __name__ == "__main__":
    main()
