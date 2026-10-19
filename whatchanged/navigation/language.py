"""Template sentences from structured facts only (no free-text generation).

Facts: object name, change type, distance (m), direction relative to the robot (ahead / ahead-left / left / behind ...),
geometric relation to nearby furniture computed from 3-D boxes (on / next to / in the walkway / behind), plan events
(detour side and added length, no path, look / search). Furniture = inventory objects of class furniture or static
whose box is at least 0.25 m tall, with at least 2000 Gaussians and fewer than 50 % of them at floor level (floors and
rugs are 'the floor').
"""

from __future__ import annotations

import numpy as np

from whatchanged.common import geometry as G

SUPPORT_MARGIN = 0.08


def short(name: str) -> str:
    return name.replace(" (uncertain)", "")


def direction(robot_xy, heading: float, p) -> str:
    v = np.asarray(p[:2], float) - np.asarray(robot_xy, float)
    a = np.degrees(np.arctan2(v[1], v[0]) - heading)
    a = (a + 180) % 360 - 180
    if abs(a) <= 25:
        return "ahead"
    if abs(a) >= 135:
        return "behind you"
    side = "left" if a > 0 else "right"
    return f"ahead-{side}" if abs(a) < 70 else f"to the {side}"


def furniture(objects: list[dict]) -> list[dict]:
    return [
        o
        for o in objects
        if o["cls"] in ("furniture", "static")
        and o["box"]["size"][2] >= 0.25
        and o["n_gauss"] >= 2000
        and o.get("floor_frac", 0) < 0.5
    ]


def fp_dist(c_xy, f: dict) -> float:
    """Distance (m) from a point to the footprint rectangle of furniture f."""
    loc = (np.asarray(c_xy, float) - np.array(f["box"]["centre"][:2])) @ G.rot2(f["box"]["yaw"])
    dx = max(abs(loc[0]) - f["box"]["size"][0] / 2, 0)
    dy = max(abs(loc[1]) - f["box"]["size"][1] / 2, 0)
    return float(np.hypot(dx, dy))


def top_of(o: dict) -> float:
    return o["box"]["centre"][2] + o["box"]["size"][2] / 2


def relation(
    box: dict, objects: list[dict], robot_xy=None, eye_h: float = 1.5, floor_cells_free=None, name=None
) -> str:
    """box: dict(centre, size, yaw) in F. Returns a phrase like 'on the coffee table' or 'on the floor next to the
    sofa'; empty when nothing is near."""
    c = np.array(box["centre"], float)
    bot = c[2] - box["size"][2] / 2
    # never relate a thing to itself
    F = [f for f in furniture(objects) if name is None or short(f["name"]) != short(name)]
    # supports: furniture, or a larger item (a pillow, a box) whose top is at the object's bottom and whose footprint
    # (+ 8 cm) contains the object's centre; the highest such surface wins.
    # Small furniture (a side table, a stool) is too sparse in the splat to be a landmark but is still a surface things
    # sit on.
    small_f = [
        o
        for o in objects
        if o["cls"] == "furniture"
        and o not in F
        and o["n_gauss"] >= 500
        and o.get("floor_frac", 0) < 0.5
        and max(o["box"]["size"][:2]) >= 0.2
        and (name is None or short(o["name"]) != short(name))
    ]
    items = [
        o
        for o in objects
        if o["cls"] == "items"
        and max(o["box"]["size"][:2]) >= 0.2
        and o["n_gauss"] >= 1000
        and np.linalg.norm(np.array(o["box"]["centre"]) - c) > 0.01
    ]
    sup = [f for f in F + small_f + items if fp_dist(c[:2], f) <= SUPPORT_MARGIN and -0.08 <= bot - top_of(f) <= 0.06]
    if sup:
        f = max(sup, key=lambda f: top_of(f))
        under = (
            [
                g
                for g in F
                if g is not f
                and fp_dist(np.array(f["box"]["centre"][:2]), g) <= SUPPORT_MARGIN
                and top_of(g) > top_of(f) - 0.4
            ]
            if f["cls"] == "items"
            else []
        )
        return f"on the {short(f['name'])}" + (f" on the {short(under[0]['name'])}" if under else "")
    if bot < 0.08:
        if floor_cells_free:
            return "in the walkway"
        near = [(fp_dist(c[:2], f) - 0.5 * min(box["size"][:2]), f) for f in F]
        near = [x for x in near if x[0] < 0.35]
        if near:
            return f"on the floor next to the {short(min(near, key=lambda x: x[0])[1]['name'])}"
        return "on the floor"
    near = sorted(((fp_dist(c[:2], f) + (0.15 if f["cls"] == "static" else 0.0), f) for f in F), key=lambda x: x[0])
    if near and near[0][0] < 0.35:
        return f"next to the {short(near[0][1]['name'])}"
    return ""


def behind(box: dict, objects: list[dict], robot_xy, eye_h: float = 1.5) -> str | None:
    """Furniture that hides the box from the robot (2.5-D: the eye-to-box ray passes through the furniture's footprint
    below its top)."""
    c = np.array(box["centre"], float)
    e = np.r_[robot_xy, eye_h]
    for f in furniture(objects):
        fc = np.array(f["box"]["centre"])
        fs = np.array(f["box"]["size"])
        for s in np.linspace(0.1, 0.9, 17):
            p = e + s * (c - e)
            if (
                G.in_footprint(p[None, :2], fc, fs, f["box"]["yaw"], 1.0)[0]
                and p[2] < top_of(f)
                and not G.in_footprint(c[None, :2], fc, fs, f["box"]["yaw"], 1.0)[0]
            ):
                return short(f["name"])
    return None


def m(x: float) -> str:
    return f"{x:.1f} m"


def sentence(kind: str, **f) -> str:
    n = short(f.get("name", "object"))
    N = n[:1].upper() + n[1:]
    if kind == "task":
        return f"Task: find the {n} ({f['n_inst']} in memory)." if f.get("n_inst", 1) > 1 else f"Task: find the {n}."
    if kind == "target_confirmed":
        return f"{N} confirmed {f['rel']}, {m(f['dist'])} {f['dir']}.".replace("  ", " ")
    if kind == "target_moved":
        return f"{N} is not {f['rel_old']} any more; found it {f['rel_new']}, {m(f['dist'])} {f['dir']}.".replace(
            "  ", " "
        )
    if kind == "target_elsewhere":
        return f"Saw a {n} {f['rel_new']}, {m(f['dist'])} {f['dir']}; its remembered place is not checked yet.".replace(
            "  ", " "
        )
    if kind == "target_missing":
        return f"{N} not found: its place {f['rel_old']} is empty; last seen there. Searching."
    if kind == "target_unseen":
        why = f" (hidden behind the {f['behind']})" if f.get("behind") else ""
        return f"{N} not visible yet{why}; last seen {f['rel_old']}; moving to look."
    if kind == "target_occluded":
        return f"{N} is covered from here{(' by the ' + f['by']) if f.get('by') else ''}; moving to look."
    if kind == "new_obstacle":
        what = ("a " + n) if n != "object" else "an unknown object"
        return f"New obstacle: {what} {f['rel']}, {m(f['dist'])} {f['dir']}.".replace("  ", " ")
    if kind == "freed":
        return f"Space is free where the {n} was."
    if kind == "plan":
        return f"Path to the {n}: {m(f['length'])}."
    if kind == "detour":
        if f["extra"] >= 0.05:
            return f"Detouring {f['side']} around the {n} (adds {m(f['extra'])})."
        return f"Re-planned around the {n}."
    if kind == "no_path":
        return (
            f"No path for my {f['radius']:.2f} m footprint; going to the closest reachable spot, "
            f"{m(f['remaining'])} short of the {n}."
        )
    if kind == "start_blocked":
        return f"Too tight to move here for my {f['radius']:.2f} m footprint (narrowest gap {m(2 * f['gap'])})."
    if kind == "arrived":
        return f"Within reach of the {n}."
    if kind == "look":
        return f"Going to a spot {m(f['dist'])} {f['dir']} to look for the {n}."
    return kind
