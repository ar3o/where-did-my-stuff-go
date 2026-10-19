"""Evaluation of one robot run (CPU). Ground truth is read only here: prepared worlds -> world.py's
<data>/<scene>/navigation/worlds/<world>/edit.json; real episodes -> <data>/<scene>/navigation/truth/<world>.json, a
hand-written note of what changed (evaluation only).

    python -m whatchanged.navigation.evaluate --scene config1 --run r01

Metrics: verdict vs the expected verdict for the target (confirmed if the target is unchanged; moved (with displacement
error) / not_here if it was moved / removed); target location error; false changes (memory objects reported gone that
the truth leaves in place, among those observed); new-obstacle detection of added objects; collisions (robot trace
within its radius of an added object's footprint); path length, sim time, wall time, radius used, share of reachable
views checked.
"""

from __future__ import annotations

import argparse

import numpy as np

from whatchanged.common import geometry as G
from whatchanged.common.io import log, read_json, write_json
from whatchanged.navigation.common import RUNS, nav_dir


def box_dist(P, b: dict) -> np.ndarray:
    """Distance (m) from points to the footprint rectangle of a box."""
    loc = (np.asarray(P, float)[:, :2] - np.array(b["centre"][:2])) @ G.rot2(b["yaw"])
    dx = np.maximum(np.abs(loc[:, 0]) - b["size"][0] / 2, 0)
    dy = np.maximum(np.abs(loc[:, 1]) - b["size"][1] / 2, 0)
    return np.hypot(dx, dy)


def evaluate(scene: str, run_name: str) -> dict:
    rd = nav_dir(scene) / "runs" / run_name
    L = read_json(rd / "log.json")
    from whatchanged.navigation.world import world_dir

    gt = None
    p = world_dir(scene, L["world"]) / "edit.json"
    if p.exists():
        gt = read_json(p)["gt"]
    elif L["world"] == "memory":
        gt = []
    tp = nav_dir(scene) / "truth" / f"{L['world']}.json"
    if tp.exists():
        gt = read_json(tp)["gt"]
    tr = np.array(L["trace"])
    out = dict(
        scene=scene,
        run=run_name,
        world=L["world"],
        prompt=L["prompt"],
        verdict=L["verdict"],
        radius_m=L["radius_m"],
        sim_s=L["sim_s"],
        wall_s=L["wall_s"],
        path_m=round(float(np.linalg.norm(np.diff(tr, axis=0), axis=1).sum()), 2) if len(tr) > 1 else 0,
        checked=L["final_belief"].get("checked"),
        has_truth=gt is not None,
    )
    if gt is not None:
        main = max(L["task"]["groups"], key=lambda g: g["weight"])
        ed = [e for e in gt if set(e["objects"]) & set(main["ids"])]
        if not ed:
            exp = "confirmed"
        elif ed[0]["op"] == "move":
            exp = "moved"
        else:
            exp = "not_here"
        if L["task"]["intent"] == "go" and exp == "confirmed":
            exp = "reached"
        out["expected"] = exp
        out["correct"] = L["verdict"]["kind"] == exp
        if ed and ed[0]["op"] == "move":
            fs = L["final_belief"]["target"]
            out["true_displacement_m"] = ed[0]["displacement_m"]
            out["located_error_m"] = round(
                float(np.linalg.norm(np.array(fs["loc"][:2]) - np.array(ed[0]["to_F"][:2]))), 3
            )
        changed = {i for e in gt if e["op"] in ("move", "remove") for i in e["objects"]}
        rep_gone = [i for i, o in L["final_belief"]["objects"].items() if o["state"] == "gone"]
        out["false_gone"] = sorted(set(rep_gone) - changed)
        out["true_gone_found"] = sorted(set(rep_gone) & changed)
        adds = [e for e in gt if e["op"] == "add_copy"]
        rows = []
        for e in adds:
            b = e["box_F"]
            det = any(
                np.linalg.norm(np.array(n["centre"][:2]) - np.array(b["centre"][:2])) < 0.5
                for n in L["final_belief"]["new_obstacles"]
            )
            coll = bool(len(tr) and box_dist(tr, b).min() < L["radius_m"] - 0.02)
            rows.append(
                dict(
                    name=e["name"],
                    detected=det,
                    collision=coll,
                    min_clearance_m=round(float(box_dist(tr, b).min()), 3) if len(tr) else None,
                )
            )
        out["added_obstacles"] = rows
        out["false_new_obstacles"] = sum(
            1
            for n in L["final_belief"]["new_obstacles"]
            if not any(
                np.linalg.norm(np.array(n["centre"][:2]) - np.array(e["box_F"]["centre"][:2])) < 0.5 for e in adds
            )
        )
    out["events"] = [e["text"] for e in L["events"]]
    write_json(out, rd / "eval.json")
    write_json(out, RUNS / f"eval_sim_{scene}_{run_name}.json")
    log("eval", {k: v for k, v in out.items() if k != "events"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    evaluate(a.scene, a.run)


if __name__ == "__main__":
    main()
