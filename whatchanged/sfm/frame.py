"""The metric world frame of a scene, built from the SfM model alone.

* Scale: SfM is up to scale. MoGe-3 depth is metric, so the scale is the median over keyframes of the median ratio
  MoGe depth / SfM depth at the keyframe's triangulated keypoints.
* Up and origin: greedy RANSAC planes on the scaled SfM points. The support plane is the one with the most inliers
  among planes that at least 80 % of the cameras are above and whose normal agrees with the cameras' image-up
  direction. The origin is the median point where the cameras' optical axes hit that plane, and x is keyframe 0's
  image-right.
* A final refit of the plane on points within 3 cm of it (4 mm inlier threshold) puts the support surface exactly at
  z = 0. The checker's training edits work in height bands above it (a vacated footprint is filled at the object's
  bottom), so the plane has to be level.
"""

from __future__ import annotations

import numpy as np

from whatchanged.common import geometry as geo

PLANE_THR_M = 0.01
REFIT_BAND_M = 0.03
REFIT_THR_M = 0.004
MIN_ABOVE = 0.8
MIN_UP_AGREE = 0.3
MAX_PLANES = 4
POINTS_PER_PLANE_FIT = 30000


def metric_scale(depth_pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, list[float]]:
    """depth_pairs = [(SfM depths, MoGe depths at the same pixels)] per keyframe -> (scale, per-keyframe ratios)."""
    ratios = []
    for z_sfm, z_mono in depth_pairs:
        ok = (z_sfm > 0) & (z_mono > 0)
        if ok.sum() >= 50:
            ratios.append(float(np.median(z_mono[ok] / z_sfm[ok])))
    if not ratios:
        raise RuntimeError("no keyframe had enough points with MoGe depth for the metric scale")
    return float(np.median(ratios)), ratios


def support_plane(P: np.ndarray, poses: np.ndarray, rng) -> tuple[np.ndarray, float, dict]:
    C = poses[:, :3, 3]
    up_img = -poses[:, :3, 1]
    cands, rest = [], P
    for _ in range(MAX_PLANES):
        if len(rest) < 500:
            break
        n, d, inl = geo.ransac_plane(rest, PLANE_THR_M, 300, rng)
        side = C @ n + d
        if np.median(side) < 0:
            n, d, side = -n, -d, -side
        cands.append(
            dict(
                n=n,
                d=float(d),
                inliers=int(inl.sum()),
                above=float((side > 0.02).mean()),
                up=float(np.median(up_img @ n)),
            )
        )
        rest = rest[~inl]
    good = [c for c in cands if c["above"] >= MIN_ABOVE and c["up"] >= MIN_UP_AGREE]
    pick = max(good or cands, key=lambda c: c["inliers"])
    info = dict(
        candidates=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()} for c in cands],
        pick=cands.index(pick),
        plane_ok=bool(good),
    )
    return pick["n"], pick["d"], info


def frame_from_plane(n: np.ndarray, d: float, poses: np.ndarray) -> np.ndarray:
    """Rigid transform into the frame with z = n, origin at the median optical-axis hit, x along keyframe 0's
    image-right."""
    C, fwd = poses[:, :3, 3], poses[:, :3, 2]
    den = fwd @ n
    t = -(C @ n + d) / np.where(np.abs(den) < 1e-6, -1e-6, den)
    hits = C + t[:, None] * fwd
    ok = (t > 0) & (t < 10)
    c0 = np.median(hits[ok], 0) if ok.any() else C.mean(0) - (C.mean(0) @ n + d) * n
    c0 = c0 - (c0 @ n + d) * n
    x = poses[0, :3, 0] - (poses[0, :3, 0] @ n) * n
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(n, x), n])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ c0
    return T


def refit_floor(Pw: np.ndarray, rng) -> tuple[np.ndarray, dict]:
    """Small tilt and height correction that makes the support plane exactly z = 0 (the origin keeps its xy)."""
    cand = Pw[np.abs(Pw[:, 2]) < REFIT_BAND_M]
    n, d, inl = geo.ransac_plane(cand, REFIT_THR_M, 400, rng)
    if n[2] < 0:
        n, d = -n, -d
    Rc = geo.rotation_between(n, np.array([0, 0, 1.0]))
    plane_point = -d * n
    T = np.eye(4)
    T[:3, :3] = Rc
    T[:3, 3] = -Rc @ plane_point + np.array([plane_point[0], plane_point[1], 0.0])
    tilt = float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))
    return T, dict(tilt_deg=tilt, z_shift_mm=float(d * 1000), n_points=int(len(cand)), inlier_frac=float(inl.mean()))


def world_frame(P_sfm: np.ndarray, poses_sfm: np.ndarray, scale: float) -> tuple[np.ndarray, dict]:
    """Similarity SfM -> world. P_sfm: well-triangulated SfM points; poses_sfm: frame camera-to-world poses."""
    rng = np.random.default_rng(0)
    S = np.diag([scale, scale, scale, 1.0])
    P = P_sfm * scale
    poses = np.stack([geo.sim3_pose(S, T) for T in poses_sfm])
    sub = P[rng.choice(len(P), min(POINTS_PER_PLANE_FIT, len(P)), replace=False)]
    n, d, plane_info = support_plane(sub, poses, rng)
    T_frame = frame_from_plane(n, d, poses)
    T_fix, fix_info = refit_floor(geo.apply(T_frame, P), rng)
    return T_fix @ T_frame @ S, dict(scale=scale, plane=plane_info, refit=fix_info)
