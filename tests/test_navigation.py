from types import SimpleNamespace

import numpy as np
import pytest

from whatchanged.navigation import language, planner, score
from whatchanged.navigation.belief import Belief

CELL = 0.05


def room(nx: int = 80, ny: int = 60, gap_m: float | None = None):
    """Free 4 x 3 m room with a wall across x = 2 m; the wall has an opening of gap_m metres centred at y = 1.5 m."""
    free = np.ones((nx, ny), bool)
    occ = np.zeros((nx, ny), bool)
    occ[:, 0] = occ[:, -1] = occ[0, :] = occ[-1, :] = True
    occ[39:41, :] = True
    if gap_m is not None:
        half = int(round(gap_m / CELL / 2))
        occ[39:41, 30 - half : 30 + half] = False
    free &= ~occ
    return free, occ, np.zeros((nx, ny), bool)


def near(goal, tol: float = 0.15):
    return lambda xy: np.linalg.norm(xy - np.asarray(goal), axis=1) <= tol


def test_astar_goes_through_the_gap():
    free, occ, unk = room(gap_m=1.0)
    res = planner.plan(free, occ, unk, np.zeros(2), CELL, [1.0, 1.5], near([3.0, 1.5]), radius=0.25)
    assert res["status"] == "ok"
    P = np.array(res["path_xy"])
    assert np.allclose(P[0], [1.0, 1.5]) and np.linalg.norm(P[-1] - [3.0, 1.5]) <= 0.15
    crossing = P[np.argmin(np.abs(P[:, 0] - 2.0))]
    assert abs(crossing[1] - 1.5) < 0.3  # through the opening
    assert res["length"] == pytest.approx(2.0, abs=0.2)  # string pulling leaves an almost straight line
    assert not planner.path_blocked(res["path_xy"], free, occ, unk, np.zeros(2), CELL, 0.2)[0]


def test_no_path_reports_partial_plan_and_squeeze():
    free, occ, unk = room(gap_m=0.4)
    res = planner.plan(free, occ, unk, np.zeros(2), CELL, [1.0, 1.5], near([3.0, 1.5]), radius=0.25)
    assert res["status"] == "no_path_partial"
    assert res["path_xy"][-1][0] < 2.0  # stops on the near side of the wall
    assert res["squeeze"]["min_clear_m"] == pytest.approx(0.2, abs=0.05)  # half of the 0.4 m opening


def test_astar_on_a_toy_grid():
    stand = np.ones((5, 5), bool)
    stand[2, :4] = False
    path, cost = planner.astar(stand, np.ones((5, 5)), (0, 0), np.array([[4, 0]]))
    assert path[0] == (0, 0) and path[-1] == (4, 0)
    assert all(stand[p] for p in path)
    assert any(p[1] == 4 for p in path)  # the only way round the wall is through its open end
    assert cost == pytest.approx(4 + 4 * np.sqrt(2), abs=1e-6)


def shrink_belief(free, occ, unk):
    events = []
    b = SimpleNamespace(
        ladder=[0.25, 0.2, 0.15],
        radius=0.25,
        radius_used=0.25,
        free_now=free,
        occ_now=occ,
        unk_now=unk,
        o=np.zeros(2),
        cell=CELL,
        M={},
        t_prev=0.0,
        emit=lambda t, kind, **f: events.append((kind, f)),
    )
    return b, events


def test_shrink_ladder_takes_the_first_radius_that_fits():
    free, occ, unk = room(gap_m=0.5)
    b, events = shrink_belief(free, occ, unk)
    robot = SimpleNamespace(xy=np.array([1.0, 1.5]))
    res = Belief._plan(b, robot, near([3.0, 1.5]), "reach")
    assert res["status"] == "ok"
    assert b.radius_used == pytest.approx(0.2)  # 0.25 m does not pass a 0.5 m opening, 0.2 m does
    assert events == [("shrink", dict(r0=0.25, r1=0.2))]


def test_shrink_ladder_keeps_the_default_radius_when_it_fits():
    free, occ, unk = room(gap_m=1.0)
    b, events = shrink_belief(free, occ, unk)
    res = Belief._plan(b, SimpleNamespace(xy=np.array([1.0, 1.5])), near([3.0, 1.5]), "reach")
    assert res["status"] == "ok" and b.radius_used == 0.25 and not events


def obj(oid, name, cls, centre, size, n_gauss, floor_frac=0.0, yaw=0.0):
    return dict(
        id=oid, name=name, cls=cls, n_gauss=n_gauss, floor_frac=floor_frac, box=dict(centre=centre, size=size, yaw=yaw)
    )


def test_relation_on_a_side_table():
    objects = [
        obj("I1", "side table", "furniture", [1.0, 1.0, 0.25], [0.4, 0.4, 0.5], 800),  # too sparse to be a landmark
        obj("I2", "sofa", "furniture", [3.0, 1.0, 0.4], [2.0, 0.9, 0.8], 6000),
        obj("I3", "mug", "items", [1.05, 0.95, 0.55], [0.1, 0.1, 0.1], 400),
    ]
    assert language.relation(objects[2]["box"], objects, name="mug") == "on the side table"


def test_relation_on_a_pillow_on_a_sofa():
    objects = [
        obj("I1", "sofa", "furniture", [2.0, 1.0, 0.4], [2.0, 0.9, 0.8], 6000),
        obj("I2", "pillow", "items", [2.0, 1.0, 0.525], [0.45, 0.45, 0.15], 1500),
        obj("I3", "remote control", "items", [2.05, 1.0, 0.62], [0.18, 0.05, 0.04], 300),
    ]
    assert language.relation(objects[2]["box"], objects, name="remote control") == "on the pillow on the sofa"


def test_relation_on_the_floor_next_to_furniture():
    objects = [obj("I1", "sofa", "furniture", [2.0, 1.0, 0.4], [2.0, 0.9, 0.8], 6000)]
    box = dict(centre=[2.0, 0.4, 0.05], size=[0.2, 0.2, 0.1], yaw=0.0)
    assert language.relation(box, objects) == "on the floor next to the sofa"
    assert language.relation(box, objects, floor_cells_free=True) == "in the walkway"


@pytest.mark.parametrize(
    "verdict, claim",
    [
        (dict(verdict="confirmed"), "here"),
        (dict(verdict="reached"), "here"),
        (dict(verdict="not_here"), "gone"),
        (dict(kind="moved"), "gone"),
        (dict(verdict="reached_moved"), "gone"),
        (
            dict(
                verdict="unverified",
                text="Could not verify the hammer: something new is standing in front of its place.",
            ),
            "covered",
        ),
        (dict(verdict="unverified", text="Could not see the hammer's place from any reachable view."), "abstain"),
        (dict(verdict="blocked", text="No path to the hammer."), "abstain"),
    ],
)
def test_claim_mapping(verdict, claim):
    assert score.claim(verdict) == claim


def test_score_counts_false_here():
    truth = {"_note": "ignored", "hammer": {"1": "here", "2": "gone"}, "banana": {"1": "covered"}}
    rows = {"ep1_hammer": "here", "ep2_hammer": "here", "ep1_banana": "abstain"}
    s = score.score(rows, truth)
    assert (s["correct"], s["wrong"], s["abstain"], s["false_here"], s["n"]) == (1, 1, 1, 1, 3)
    assert s["per_run"]["ep2_hammer"] == dict(truth="gone", claim="here", result="wrong")
