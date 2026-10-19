"""Geometry helpers (numpy / cv2): planes, similarity transforms, z-up oriented boxes, pinhole lifting and projection,
camera poses and quaternions.

Conventions: world frame z-up in metres; camera-to-world poses T_wc in the OpenCV camera convention (x right, y down,
z forward); K in pixels with pixel centres at +0.5. Gaussian quaternions are wxyz (gsplat order).
"""

from __future__ import annotations

import math

import cv2
import numpy as np


# ------------------------------------------------------------------ planes and similarity transforms
def ransac_plane(P: np.ndarray, thr: float = 0.01, iters: int = 400, rng=None):
    """Plane n.p + d = 0 through P (n, 3) with a least-squares refit on the inliers. Returns (n, d, inlier mask)."""
    rng = np.random.default_rng(0) if rng is None else rng
    best = (None, None, -1)
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -n @ a
        count = int((np.abs(P @ n + d) < thr).sum())
        if count > best[2]:
            best = (n, d, count)
    n, d, _ = best
    inl = np.abs(P @ n + d) < thr
    Q = P[inl]
    c = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
    n2 = vt[2] * np.sign(vt[2] @ n)
    d2 = -n2 @ c
    return n2, d2, np.abs(P @ n2 + d2) < thr


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """dst ~ s R src + t (Umeyama 1991). Returns s, R, t."""
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    C = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = (xs**2).sum(1).mean()
    s = float(np.trace(np.diag(S) @ D) / var) if with_scale else 1.0
    return s, R, md - s * R @ ms


def trimmed_sim3(src: np.ndarray, dst: np.ndarray, iters: int = 6, keep: float = 0.8):
    """Umeyama refitted on the `keep` fraction of smallest residuals. Returns s, R, t, residuals (all), inlier mask."""
    m = np.ones(len(src), bool)
    for _ in range(iters):
        s, R, t = umeyama(src[m], dst[m])
        r = np.linalg.norm((s * src @ R.T + t) - dst, axis=1)
        m = r <= np.quantile(r, keep)
    s, R, t = umeyama(src[m], dst[m])
    r = np.linalg.norm((s * src @ R.T + t) - dst, axis=1)
    return s, R, t, r, m


def sim3_matrix(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def apply(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def sim3_pose(T_sim: np.ndarray, T_c: np.ndarray) -> np.ndarray:
    """Camera pose T_c mapped by a similarity: the scale goes into the translation, the rotation stays orthonormal."""
    A = T_sim[:3, :3]
    s = np.cbrt(np.linalg.det(A))
    out = np.eye(4)
    out[:3, :3] = (A / s) @ T_c[:3, :3]
    out[:3, 3] = A @ T_c[:3, 3] + T_sim[:3, 3]
    return out


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a to unit vector b."""
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.eye(3)
    ang = math.atan2(s, float(a @ b))
    return cv2.Rodrigues(axis / s * ang)[0]


def pose_delta(Ta: np.ndarray, Tb: np.ndarray) -> tuple[float, float]:
    """(centre distance in cm, rotation angle in degrees) between two camera poses."""
    d = float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))
    Rd = Ta[:3, :3].T @ Tb[:3, :3]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(Rd) - 1) / 2, -1, 1))))
    return round(d * 100, 2), round(ang, 3)


# ------------------------------------------------------------------ z-up oriented boxes
def rot2(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def obb_zup(P: np.ndarray, q_lo: float = 2, q_hi: float = 98):
    """Z-up oriented box of points P (n, 3): yaw from the minimum-area rectangle of the trimmed xy, extents from
    percentiles along the box axes. size[0] is along yaw. Returns centre, size, yaw in [-pi/2, pi/2)."""
    xy = P[:, :2]
    c0 = np.median(xy, 0)
    r = np.linalg.norm(xy - c0, axis=1)
    keep = r <= np.quantile(r, 0.97) + 1e-9
    rect = cv2.minAreaRect(xy[keep].astype(np.float32))
    yaw = np.deg2rad(rect[2])
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
    R = rot2(yaw)
    local = (xy - c0) @ R
    lo, hi = np.percentile(local, q_lo, 0), np.percentile(local, q_hi, 0)
    zlo, zhi = np.percentile(P[:, 2], 1), np.percentile(P[:, 2], 99)
    cxy = c0 + R @ ((lo + hi) / 2)
    size = np.array([hi[0] - lo[0], hi[1] - lo[1], zhi - zlo])
    return np.array([cxy[0], cxy[1], (zlo + zhi) / 2]), size, float(yaw)


def footprint(centre, size, yaw: float, shrink: float = 1.0) -> np.ndarray:
    """4 corners (x, y) of the footprint rectangle of a z-up box, scaled by `shrink`."""
    hx, hy = np.asarray(size[:2], float) / 2 * shrink
    return np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]) @ rot2(yaw).T + np.asarray(centre, float)[:2]


def in_footprint(xy: np.ndarray, centre, size, yaw: float, shrink: float = 1.0) -> np.ndarray:
    """Which points xy (n, 2) lie inside the footprint rectangle of a z-up box (scaled by `shrink`)."""
    loc = (xy - np.asarray(centre)[:2]) @ rot2(yaw)
    return (np.abs(loc[:, 0]) <= size[0] / 2 * shrink) & (np.abs(loc[:, 1]) <= size[1] / 2 * shrink)


def box3d(P: np.ndarray) -> dict | None:
    """Box of a lifted region (5-95 % extents, sides at least 2 cm); None below 30 points."""
    if len(P) < 30:
        return None
    c, s, yaw = obb_zup(P, 5, 95)
    return dict(centre=c.round(4).tolist(), size=np.maximum(s, 0.02).round(4).tolist(), yaw=round(float(yaw), 4))


def box_corners(centre, size, yaw: float) -> np.ndarray:
    """8 corners of a z-up box, bottom face first."""
    c, s = np.asarray(centre, float), np.asarray(size, float) / 2
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    unit = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)]) * s
    return unit @ R.T + c


# ------------------------------------------------------------------ pinhole cameras
def lift(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    """World points of the mask pixels with positive depth."""
    v, u = np.nonzero(mask & (depth > 0))
    z = depth[v, u]
    P = np.c_[(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z]
    return P @ T_wc[:3, :3].T + T_wc[:3, 3]


def project(P: np.ndarray, K: np.ndarray, T_wc: np.ndarray):
    """World points -> (u, v, z) in pixels and camera depth."""
    T_cw = np.linalg.inv(T_wc)
    pc = np.atleast_2d(P) @ T_cw[:3, :3].T + T_cw[:3, 3]
    z = pc[:, 2]
    u = K[0, 0] * pc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
    v = K[1, 1] * pc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
    return u, v, z


def look_at(p: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Camera at p looking at target with image-down opposite to `up`."""
    z = target - p
    z /= np.linalg.norm(z)
    y = -up - (-up @ z) * z
    if np.linalg.norm(y) < 1e-6:
        y = np.array([0, 0, 1.0])
    y /= np.linalg.norm(y)
    T = np.eye(4)
    T[:3, :3] = np.c_[np.cross(y, z), y, z]
    T[:3, 3] = p
    return T


def scale_K(K: np.ndarray, s: float) -> np.ndarray:
    return np.asarray(K, float) * np.array([[s], [s], [1.0]])


# ------------------------------------------------------------------ rotations
def quat_xyzw_from_R(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q = q / np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def quat_wxyz_from_R(R: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw_from_R(R)
    return np.array([w, x, y, z])


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
