import numpy as np
import pytest

from whatchanged.common import depth as depthlib
from whatchanged.common import geometry as geo
from whatchanged.sfm import frame


def test_obb_recovers_a_rotated_box():
    rng = np.random.default_rng(0)
    local = rng.uniform([-0.2, -0.05, 0.0], [0.2, 0.05, 0.3], size=(5000, 3))
    yaw = 0.5
    R = geo.rot_z(yaw)
    P = local @ R.T + [1.0, 2.0, 0.0]
    centre, size, got_yaw = geo.obb_zup(P, 0, 100)
    assert centre == pytest.approx([1.0, 2.0, 0.15], abs=0.01)
    assert sorted(size[:2]) == pytest.approx([0.1, 0.4], abs=0.02) and size[2] == pytest.approx(0.3, abs=0.02)
    long_axis = got_yaw if size[0] > size[1] else got_yaw + np.pi / 2  # size[0] lies along the returned yaw
    assert np.sin(long_axis - yaw) == pytest.approx(0, abs=0.02)


def test_trimmed_sim3_ignores_outliers():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(200, 3))
    R = geo.rot_z(0.3)
    dst = 2.0 * src @ R.T + [0.5, -1.0, 2.0]
    dst[:10] += 5.0
    s, R2, t, _, inl = geo.trimmed_sim3(src, dst, keep=0.9)
    assert s == pytest.approx(2.0, abs=1e-6)
    assert np.allclose(R2, R, atol=1e-6)
    assert not inl[:10].any()


def test_lift_and_project_round_trip():
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    T = geo.look_at(np.array([0, -2.0, 1.5]), np.zeros(3), np.array([0, 0, 1.0]))
    depth = np.full((480, 640), 2.0, np.float32)
    mask = np.zeros((480, 640), bool)
    mask[100, 200] = True
    P = geo.lift(mask, depth, K, T)
    u, v, z = geo.project(P, K, T)
    assert (u[0], v[0], z[0]) == pytest.approx((200.5, 100.5, 2.0), abs=1e-6)


def test_depth_alignment_recovers_inverse_depth_affine():
    rng = np.random.default_rng(2)
    D_pred = rng.uniform(1.0, 4.0, size=(240, 320)).astype(np.float32)
    D_mono = (1.0 / (0.8 / D_pred + 0.05)).astype(np.float32)  # 1/D_pred = (1/D_mono - 0.05) / 0.8
    ok = np.ones_like(D_pred, bool)
    D, info = depthlib.align(D_pred, D_mono, ok)
    assert info["rel_err_median"] < 1e-3
    assert np.median(np.abs(D - D_pred) / D_pred) < 1e-3


def test_world_frame_puts_the_floor_at_zero_and_scales_metric():
    rng = np.random.default_rng(3)
    floor = np.c_[rng.uniform(-2, 2, 4000), rng.uniform(-2, 2, 4000), np.zeros(4000)]
    wall = np.c_[rng.uniform(-2, 2, 1000), np.full(1000, 2.0), rng.uniform(0, 2, 1000)]
    P = np.vstack([floor, wall])
    poses = []
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        c = np.array([1.5 * np.cos(a), 1.5 * np.sin(a), 1.4])
        poses.append(geo.look_at(c, np.zeros(3), np.array([0, 0, 1.0])))
    poses = np.stack(poses)
    T_sfm = geo.sim3_matrix(
        0.5,
        geo.rot_z(0.7) @ geo.rotation_between(np.array([0, 0, 1.0]), np.array([0, 0.6, 0.8])),
        np.array([3.0, 1.0, -2.0]),
    )
    P_sfm = geo.apply(T_sfm, P)
    poses_sfm = np.stack([geo.sim3_pose(T_sfm, T) for T in poses])
    T_world, info = frame.world_frame(P_sfm, poses_sfm, scale=2.0)
    Pw = geo.apply(T_world, P_sfm)
    assert np.abs(Pw[:4000, 2]).max() < 1e-3
    heights = [geo.sim3_pose(T_world, T)[2, 3] for T in poses_sfm]
    assert heights == pytest.approx([1.4] * 12, abs=1e-3)
    assert frame.metric_scale([(np.ones(100), np.full(100, 2.0))])[0] == pytest.approx(2.0)
