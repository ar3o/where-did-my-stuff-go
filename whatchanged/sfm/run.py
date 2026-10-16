"""Camera poses of the reference frames and the query photos in one metric world frame.

    python -m whatchanged.sfm.run --scene config1 [--skip-colmap]

1. MoGe-3 depth of every frame and photo (vision env worker); reused later for the comparison.
2. pycolmap SfM of frames + photos (whatchanged.sfm.colmap).
3. World frame from the SfM points and the MoGe scale (whatchanged.sfm.frame).
Outputs: depth/, sfm/, poses.json {W, H, K, frames: [{name, T_wc, split, kf, t, sharpness}], world, measure},
photos_sfm.json {photos: {name: {T_wc, K, W, H, n_points, reproj_px_median} | null}}, sfm_points.npz (world xyz, rgb,
reprojection error, track length).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from whatchanged import config
from whatchanged.common import geometry as geo
from whatchanged.common.io import log, read_json, save_npz, write_json
from whatchanged.common.scene import Scene
from whatchanged.common.workers import run_worker
from whatchanged.sfm import colmap, frame


@dataclass
class SfmConfig:
    photo_focal_35mm: float = 24.0  # prior only; bundle adjustment refines it
    max_reproj_px: float = 1.5  # all frames registered and below this -> "sfm"; else "sfm_partial"
    point_max_error_px: float = 2.0
    point_min_track: int = 3
    moge_need_mib: int = 3000


def run_depth(scene: Scene, frames: list[str], photos: list[str], need_mib: int) -> None:
    items = [dict(image=str(scene.frame_image(n)), out=str(scene.depth / "frames" / f"{n}.npz")) for n in frames]
    items += [dict(image=str(scene.photo_image(n)), out=str(scene.depth / "photos" / f"{n}.npz")) for n in photos]
    run_worker("vision", "whatchanged.common.moge_worker", dict(items=items), scene.depth / "jobs_moge.json", need_mib)


def photo_cameras(scene: Scene, photos: list[str], f35: float) -> list[dict]:
    groups = {}
    for n in photos:
        h, w = cv2.imread(str(scene.photo_image(n))).shape[:2]
        f = colmap.focal_from_35mm(f35, w, h)
        groups.setdefault((w, h), dict(names=[], K=(f, f, w / 2, h / 2)))["names"].append(f"photos/{n}.jpg")
    return list(groups.values())


def keyframe_depth_pairs(scene: Scene, rec, reg: dict, kf_names: list[str]) -> list:
    pairs = []
    for n in kf_names:
        img = reg.get(f"frames/{n}.jpg")
        if img is None:
            continue
        M = colmap.cam_from_world(img)
        uv, z = [], []
        for p2 in img.points2D:
            if p2.has_point3D():
                uv.append(p2.xy)
                z.append((M[:3, :3] @ rec.points3D[p2.point3D_id].xyz + M[:3, 3])[2])
        if not uv:
            continue
        mono = np.load(scene.depth / "frames" / f"{n}.npz")["depth"].astype(np.float32)
        uv = np.asarray(uv)
        u = np.clip(uv[:, 0].astype(int), 0, mono.shape[1] - 1)
        v = np.clip(uv[:, 1].astype(int), 0, mono.shape[0] - 1)
        pairs.append((np.asarray(z), mono[v, u]))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--skip-colmap", action="store_true", help="reuse the model already in sfm/")
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()
    cfg = config.load(SfmConfig, "sfm", args.config)
    scene = Scene(args.scene)
    fr = read_json(scene.frames_json)
    W, H = fr["size"]
    frames = [f["name"] for f in fr["frames"]]
    meta = {f["name"]: f for f in fr["frames"]}
    kf = [f["name"] for f in sorted((f for f in fr["frames"] if f["kf"] >= 0), key=lambda f: f["kf"])]
    photos = sorted(p for g in scene.groups().values() for p in g)
    t0 = time.time()
    run_depth(scene, frames, photos, cfg.moge_need_mib)
    import pycolmap

    names = [f"frames/{n}.jpg" for n in frames]
    kf_names = [f"frames/{n}.jpg" for n in kf]
    if args.skip_colmap:
        rec, timing = pycolmap.Reconstruction(str(scene.sfm)), {}
    else:
        K_prior = np.median([np.load(scene.depth / "frames" / f"{n}.npz")["K"] for n in kf], 0)
        rec, timing = colmap.run_sfm(
            scene.root, scene.sfm, names, kf_names, K_prior, photo_cameras(scene, photos, cfg.photo_focal_35mm)
        )
    reg_all = {rec.images[i].name: rec.images[i] for i in rec.reg_image_ids()}
    reg = {k: v for k, v in reg_all.items() if k.startswith("frames/")}
    K = colmap.camera_K(rec.cameras[next(iter(reg.values())).camera_id])
    reproj = float(rec.compute_mean_reprojection_error())
    log(f"sfm: {len(reg)}/{len(names)} frames, {rec.num_points3D()} points, {reproj:.3f} px {timing}")
    pts = list(rec.points3D.values())
    P = np.array([p.xyz for p in pts])
    err = np.array([p.error for p in pts])
    track = np.array([p.track.length() for p in pts])
    well = (err < cfg.point_max_error_px) & (track >= cfg.point_min_track)
    reg_frames = [n for n in names if n in reg]
    poses_sfm = np.stack([np.linalg.inv(colmap.cam_from_world(reg[n])) for n in reg_frames])
    scale, ratios = frame.metric_scale(keyframe_depth_pairs(scene, rec, reg, kf))
    kf_first = [i for i, n in enumerate(reg_frames) if n in set(kf_names)]
    order = kf_first + [i for i in range(len(reg_frames)) if i not in set(kf_first)]
    T_world, world_info = frame.world_frame(P[well], poses_sfm[order], scale)
    out_frames = []
    for n, T in zip(reg_frames, poses_sfm):
        m = meta[Path(n).stem]
        out_frames.append(
            dict(
                name=m["name"],
                T_wc=geo.sim3_pose(T_world, T).round(7).tolist(),
                split=m["split"],
                kf=m["kf"],
                t=m["t"],
                sharpness=m["sharpness"],
            )
        )
    Pw = geo.apply(T_world, P)
    rgb = np.array([p.color for p in pts], np.uint8)
    save_npz(
        scene.sfm_points, xyz=Pw.astype(np.float32), rgb=rgb, err=err.astype(np.float32), track=track.astype(np.int32)
    )
    ph = {}
    for n in photos:
        img = reg_all.get(f"photos/{n}.jpg")
        if img is None:
            ph[n] = None
            continue
        cam = rec.cameras[img.camera_id]
        errs = colmap.reprojection_errors(rec, img)
        ph[n] = dict(
            T_wc=geo.sim3_pose(T_world, np.linalg.inv(colmap.cam_from_world(img))).round(7).tolist(),
            K=colmap.camera_K(cam).round(3).tolist(),
            W=int(cam.width),
            H=int(cam.height),
            n_points=len(errs),
            reproj_px_median=float(np.median(errs)) if errs else None,
        )
    n_ph = sum(v is not None for v in ph.values())
    write_json(dict(photos=ph, registered=n_ph, total=len(photos)), scene.photos_sfm_json)
    measure = dict(
        registered=len(reg),
        total=len(names),
        points=int(rec.num_points3D()),
        reproj_px=reproj,
        timing_s=timing,
        photos_registered=[n_ph, len(photos)],
        keyframe_scale_ratios=ratios,
    )
    source = "sfm" if len(reg) == len(names) and reproj < cfg.max_reproj_px else "sfm_partial"
    write_json(
        dict(
            W=W,
            H=H,
            K=K.round(4).tolist(),
            frames=out_frames,
            world=world_info,
            T_world_from_sfm=T_world.tolist(),
            measure=measure,
            pose_source=source,
            wall_s=time.time() - t0,
        ),
        scene.poses_json,
    )
    log(
        f"world frame: scale {scale:.4f}, floor tilt fix {world_info['refit']['tilt_deg']:.2f} deg; "
        f"photos {n_ph}/{len(photos)} registered"
    )


if __name__ == "__main__":
    main()
