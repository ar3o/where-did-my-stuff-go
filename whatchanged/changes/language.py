"""Text from 3D facts only: every word comes from an event's kind, name, distance, height change, confidence and the
nearest piece of furniture in the inventory. No language model looks at the images; one that did confused a change of
viewpoint with a change of the room.

    summary_line: "egg carton moved 40 cm (80%)"
    sentence:     "The egg carton moved 40 cm, to near the rattan cabinet (80 % sure)."
"""

from __future__ import annotations

import numpy as np

NEAR_M = 1.0
HEIGHT_CHANGE_M = 0.15


def clean_name(name: str) -> str:
    return name.replace(" (uncertain)", "?")


def summary_line(e: dict) -> str:
    dist = f" {e['dist_m'] * 100:.0f} cm" if e["kind"] == "moved" else ""
    return f"{e['name']} {e['kind']}{dist} ({100 * e['change_conf']:.0f}%)"


def nearest_landmark(p, furniture: list[dict], exclude: str) -> str | None:
    best, name = NEAR_M, None
    for o in furniture:
        if o["id"] == exclude:
            continue
        c, s = np.asarray(o["box"]["centre"]), np.asarray(o["box"]["size"])
        d = float(np.linalg.norm(np.maximum(np.abs(np.asarray(p) - c) - s / 2, 0)))
        if d < best:
            best, name = d, clean_name(o["name"])
    return name


def sentence(e: dict, furniture: list[dict]) -> str:
    name = clean_name(e["name"])
    sure = f"({100 * e['change_conf']:.0f} % sure)"
    if e["kind"] == "removed":
        where = nearest_landmark(e["centre_before"], furniture, e["object"])
        return f"The {name} is gone{f' from near the {where}' if where else ''} {sure}."
    if e["kind"] == "added":
        where = nearest_landmark(e["centre_after"], furniture, e["object"])
        return f"There is a {name} that was not there before{f', near the {where}' if where else ''} {sure}."
    a, b = np.asarray(e["centre_before"]), np.asarray(e["centre_after"])
    parts = [f"The {name} moved {e['dist_m'] * 100:.0f} cm"]
    dz = float(b[2] - a[2])
    if abs(dz) >= HEIGHT_CHANGE_M:
        parts.append(f"{'up' if dz > 0 else 'down'} {abs(dz) * 100:.0f} cm")
    src, dst = nearest_landmark(a, furniture, e["object"]), nearest_landmark(b, furniture, e["object"])
    if dst and dst != src:
        parts.append(f"{f'from near the {src} ' if src else ''}to near the {dst}")
    return ", ".join(parts) + f" {sure}."
