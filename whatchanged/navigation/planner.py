"""Grid path planning for a round humanoid footprint (numpy / heapq; no GPU).

plan(free, occ, unknown, origin, cell, start_xy, goal_fn, radius):
  * blocked = occupied or unknown in the current belief (unknown is never assumed free); clearance = distance (m) to the
    nearest blocked cell (Euclidean distance transform);
  * a robot centre may stand on a free cell with clearance > radius;
  * A* (8-connected, octile heuristic) from the start to any goal cell, step cost = length x (1 + CLEAR_W x
    max(0, 1 - (clearance - radius) / radius)), which prefers the middle of corridors like a ROS inflation layer, then
    any-angle shortening (string pulling: a segment is kept only if every 2 cm sample stays on standable cells);
  * a start that is not standable is snapped to the nearest standable cell within SNAP_M (reported); when no goal is
    reachable the result is "no path" with a plan to the reachable cell nearest to the goal; the 'squeeze' path with
    the minimum clearance MIN_CLEAR_M is computed as well, so a report can say how wide the narrowest gap is.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable

import numpy as np
from scipy import ndimage

CLEAR_W = 2.0
LOW_W = 1.0  # extra cost per metre on low-clutter cells (between the floor noise and the step height)
SNAP_M = 0.5
MIN_CLEAR_M = 0.10
NB = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def clearance(blocked: np.ndarray, cell: float) -> np.ndarray:
    return ndimage.distance_transform_edt(~blocked) * cell


def to_cell(xy, origin, cell: float, shape) -> tuple[int, int]:
    ij = np.floor((np.asarray(xy, float) - origin) / cell).astype(int)
    return int(np.clip(ij[0], 0, shape[0] - 1)), int(np.clip(ij[1], 0, shape[1] - 1))


def to_xy(ij, origin, cell: float) -> np.ndarray:
    return np.asarray(origin) + (np.asarray(ij, float) + 0.5) * cell


def astar(stand: np.ndarray, cost_field: np.ndarray, start: tuple, goals) -> tuple[list, float] | None:
    nx, ny = stand.shape
    goal_set = set(map(tuple, goals))
    if not goal_set:
        return None
    G = np.array(list(goal_set))

    def h(i, j):
        d = np.abs(G - (i, j))
        return float(np.min(np.max(d, 1) + (np.sqrt(2) - 1) * np.min(d, 1)))

    g = {start: 0.0}
    par = {start: None}
    pq = [(h(*start), 0.0, start)]
    seen = set()
    while pq:
        f, gc, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u in goal_set:
            path = []
            while u is not None:
                path.append(u)
                u = par[u]
            return path[::-1], gc
        for di, dj in NB:
            v = (u[0] + di, u[1] + dj)
            if not (0 <= v[0] < nx and 0 <= v[1] < ny) or not stand[v] or v in seen:
                continue
            step = np.hypot(di, dj) * 0.5 * (cost_field[u] + cost_field[v])
            ng = gc + step
            if ng < g.get(v, 1e18):
                g[v] = ng
                par[v] = u
                heapq.heappush(pq, (ng + h(*v), ng, v))
    return None


def reachable(stand: np.ndarray, start: tuple) -> np.ndarray:
    lab, _ = ndimage.label(stand, structure=np.ones((3, 3)))
    if lab[start] == 0:
        return np.zeros_like(stand)
    return lab == lab[start]


def los(stand: np.ndarray, a, b, origin, cell: float) -> bool:
    pa, pb = to_xy(a, origin, cell), to_xy(b, origin, cell)
    n = max(2, int(np.ceil(np.linalg.norm(pb - pa) / 0.02)))
    for s in np.linspace(0, 1, n):
        i, j = to_cell(pa + s * (pb - pa), origin, cell, stand.shape)
        if not stand[i, j]:
            return False
    return True


def shorten(path: list, stand: np.ndarray, origin, cell: float) -> list:
    if len(path) <= 2:
        return path
    out = [path[0]]
    k = 0
    while k < len(path) - 1:
        j = len(path) - 1
        while j > k + 1 and not los(stand, path[k], path[j], origin, cell):
            j -= 1
        out.append(path[j])
        k = j
    return out


def length_m(pts) -> float:
    pts = np.asarray(pts, float)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0


def plan(
    free: np.ndarray,
    occ: np.ndarray,
    unknown: np.ndarray,
    origin,
    cell: float,
    start_xy,
    goal_xy_fn: Callable[[np.ndarray], np.ndarray],
    radius: float,
    extra_block: np.ndarray | None = None,
    low: np.ndarray | None = None,
) -> dict:
    """free / occ / unknown: (nx, ny) bool current belief. goal_xy_fn(xy (n, 2)) -> bool mask of acceptable standing
    positions. Returns dict(status, path_xy, length, min_clear_m, ...); status is ok, start_blocked, no_path or
    no_path_partial."""
    blocked = occ | unknown | ~free
    if extra_block is not None:
        blocked = blocked | extra_block
    clear = clearance(blocked, cell)
    shape = free.shape
    stand = ~blocked & (clear > radius)
    cost = 1.0 + CLEAR_W * np.clip(1.0 - (clear - radius) / max(radius, 1e-3), 0, 1)
    if low is not None:
        cost = cost + LOW_W * low  # low clutter (< step height): traversable but avoided when possible
    s = to_cell(start_xy, origin, cell, shape)
    res = dict(radius=radius, start_cell=s, snapped_m=0.0)
    allc = np.argwhere(np.ones(shape, bool))
    goal_ok = goal_xy_fn(to_xy(allc, origin, cell)).reshape(shape)
    if not stand[s]:
        cand = np.argwhere(stand)
        if len(cand):
            d = np.linalg.norm((cand - s) * cell, axis=1)
            k = int(np.argmin(d))
            if d[k] <= SNAP_M:
                res["snapped_m"] = round(float(d[k]), 3)
                s = tuple(cand[k])
    res["start_standable"] = bool(stand[s])
    # squeeze path (minimum clearance) for the report
    sq_stand = ~blocked & (clear > MIN_CLEAR_M)
    s_sq = s if sq_stand[s] else None
    if s_sq is None:
        c2 = np.argwhere(sq_stand)
        if len(c2):
            d = np.linalg.norm((c2 - to_cell(start_xy, origin, cell, shape)) * cell, axis=1)
            if d.min() <= SNAP_M:
                s_sq = tuple(c2[int(np.argmin(d))])
    if s_sq is not None:
        sq_cost = 1.0 + CLEAR_W * np.clip(1.0 - (clear - MIN_CLEAR_M) / radius, 0, 1)
        r = astar(sq_stand, sq_cost, s_sq, np.argwhere(goal_ok & sq_stand))
        if r:
            pth = r[0]
            res["squeeze"] = dict(
                path_xy=to_xy(np.array(pth), origin, cell).round(3).tolist(),
                length=round(len(pth) * cell, 2),
                min_clear_m=round(float(min(clear[p] for p in pth)), 3),
                narrowest_xy=to_xy(np.array(min(pth, key=lambda p: clear[p])), origin, cell).round(3).tolist(),
            )
    if not stand[s]:
        res.update(status="start_blocked", path_xy=[list(np.round(start_xy, 3))], length=0.0)
        return res
    goals = np.argwhere(goal_ok & stand)
    r = astar(stand, cost, s, goals)
    if r is not None:
        pth = shorten(r[0], stand, origin, cell)
        xy = to_xy(np.array(pth), origin, cell)
        xy[0] = start_xy if res["snapped_m"] == 0 else xy[0]
        res.update(
            status="ok",
            path_xy=xy.round(3).tolist(),
            length=round(length_m(xy), 3),
            min_clear_m=round(float(min(clear[p] for p in r[0])), 3),
            goal_cell=list(map(int, r[0][-1])),
        )
        return res
    # no path: plan to the reachable cell nearest to the goal region
    R = reachable(stand, s)
    gxy = to_xy(np.argwhere(goal_ok), origin, cell) if goal_ok.any() else None
    cand = np.argwhere(R)
    if gxy is None or not len(cand):
        res.update(status="no_path", path_xy=[list(np.round(start_xy, 3))], length=0.0)
        return res
    cxy = to_xy(cand, origin, cell)
    from scipy.spatial import cKDTree

    d, _ = cKDTree(gxy).query(cxy)
    best = tuple(cand[int(np.argmin(d))])
    r = astar(stand, cost, s, np.array([best]))
    pth = shorten(r[0], stand, origin, cell) if r else [s]
    xy = to_xy(np.array(pth), origin, cell)
    res.update(
        status="no_path_partial",
        path_xy=xy.round(3).tolist(),
        length=round(length_m(xy), 3),
        remaining_m=round(float(d.min()), 3),
        goal_cell=list(map(int, best)),
    )
    return res


def path_blocked(
    path_xy, free, occ, unknown, origin, cell: float, radius: float, extra_block=None
) -> tuple[bool, np.ndarray | None]:
    """Does a previously planned path now cross blocked or too narrow cells? -> (bool, first offending point)."""
    blocked = occ | unknown | ~free
    if extra_block is not None:
        blocked = blocked | extra_block
    clear = clearance(blocked, cell)
    pts = np.asarray(path_xy, float)
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / 0.02)))
        for s in np.linspace(0, 1, n):
            p = a + s * (b - a)
            i, j = to_cell(p, origin, cell, free.shape)
            if clear[i, j] <= radius:
                return True, p
    return False, None
