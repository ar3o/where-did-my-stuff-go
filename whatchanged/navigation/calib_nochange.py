"""Memory-only no-change baseline for the per-object change score: what an unchanged object looks like to the checker.

    python -m whatchanged.navigation.calib_nochange --scene config1

Sensor proxy = the memory splat itself (no change at all) rendered at the memory-walk poses jittered by 2 cm / 0.5 deg,
with colour gain 0.9-1.1, blur sigma 1.2 and noise (the look of a different splat of the same room). For every small
inventory item (<= 0.6 m) visible on >= min_box_px known pixels of its tight box mask inside the 16 px border, the
checker's mean P(missing) and mean P(appeared) over the mask and the appearance scores (appearance.py) are collected.
The change score is c = max(mean P(missing), mean P(appeared)).
tau_chg = the 99th percentile of c over all (item, pose) samples; tau_present = the 90th; the same percentiles of the
appearance scores. Output <runs>/<scene>/navigation/nochange_tau.json. Never uses episodes. A real episode splat has
extra reconstruction differences, so a real unchanged object can exceed this tau more often than 1 %; the belief
therefore needs >= 2 novel views before it acts on an exceeding score.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from whatchanged.common.io import log, write_json
from whatchanged.common.scene import Scene
from whatchanged.navigation import appearance
from whatchanged.navigation.boxmask import box_mask
from whatchanged.navigation.common import cfg, memory_paths, scene_spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()
    C = cfg()
    from whatchanged.common.gpu import gpu_stage
    from whatchanged.navigation.memory_map import load_map
    from whatchanged.navigation.sim import ChangeNet, Sensor

    mem = memory_paths(a.scene)
    M, info = load_map(a.scene, ensure_current=True)
    P = Scene(a.scene).poses()
    spec = scene_spec(a.scene)
    W, H = C["sim"]["render_w"], C["sim"]["render_h"]
    K = np.array(P["K"], float)
    K[0] *= W / P["W"]
    K[1] *= H / P["H"]
    objs = {int(o["id"][1:]): o for o in info["objects"] if o["cls"] != "static"}
    rng = np.random.default_rng(0)
    rows = []
    with gpu_stage("nav_calib_nc", need_mib=3500):
        sensor = Sensor(a.scene, mem["splat"], mem, K, W, H)
        net = ChangeNet(spec)
        for fi, f in enumerate(P["frames"][:: a.stride]):
            T = np.array(f["T_wc"])
            th = np.radians(0.5)
            w = rng.normal(size=3)
            w /= np.linalg.norm(w)
            Kx = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
            Tj = T.copy()
            Tj[:3, :3] = T[:3, :3] @ (np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx)
            v = rng.normal(size=3)
            Tj[:3, 3] = T[:3, 3] + 0.02 * v / np.linalg.norm(v)
            ren, Dp, inst, S, full = sensor.memory(T)
            rgb, Dep = sensor.view(Tj)
            rgb = np.clip(rgb.astype(np.float32) / 255 * rng.uniform(0.9, 1.1, 3)[None, None], 0, 1)
            rgb = cv2.GaussianBlur(rgb, (0, 0), 1.2) + rng.normal(0, 0.01, rgb.shape)
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            known = S >= 3
            Pn, Pg, fq, fr = net(ren, rgb, Dp, Dep, known)
            gn = appearance.gain_fit(ren.astype(np.float32) / 255, rgb.astype(np.float32) / 255, known)
            inner = np.zeros_like(known)
            inner[16:-16, 16:-16] = True
            ki = known & inner
            TFc = M["T_F_S"] @ T
            for gk in np.unique(inst[ki]).tolist():
                gk = int(gk)
                o = objs.get(gk)
                if o is None or o["cls"] != "items" or max(o["box"]["size"]) > 0.6:
                    continue
                m = box_mask(Dp, K, TFc, o["box"]) & ki
                n = int(m.sum())
                if n < C["belief"]["min_box_px"]:
                    continue
                cd, fd, _ = appearance.scores(ren, rgb, fr, fq, ki, m, gain=gn)
                rows.append(
                    dict(
                        obj=o["id"],
                        name=o["name"],
                        size=max(o["box"]["size"]),
                        px=n,
                        pg=float(Pg[m].mean()),
                        pn=float(Pn[m].mean()),
                        cd=cd,
                        fd=fd,
                        cls=o["cls"],
                    )
                )
            if fi % 10 == 0:
                log(f"pose {fi} samples {len(rows)}")
    rows_i = [r for r in rows if r["cls"] == "items" and r["size"] <= 0.6]  # the targets are small items
    c = np.array([max(r["pg"], r["pn"]) for r in rows_i])
    pg = np.array([r["pg"] for r in rows_i])
    cd_ = np.array([r["cd"] for r in rows_i])
    fd_ = np.array([r["fd"] for r in rows_i])
    out = dict(
        scene=a.scene,
        tau_cd_99=float(np.percentile(cd_, 99)),
        tau_cd_90=float(np.percentile(cd_, 90)),
        tau_fd_99=float(np.percentile(fd_, 99)),
        tau_fd_90=float(np.percentile(fd_, 90)),
        n_samples=len(rows_i),
        n_objects=len({r["obj"] for r in rows_i}),
        statistic="max(mean P(gone), mean P(new)) over the object's tight box mask (boxmask.py)",
        tau_chg=float(np.percentile(c, 99)),
        tau_present=float(np.percentile(c, 90)),
        tau_gone_only=float(np.percentile(pg, 99)),
        c_percentiles={str(q): round(float(np.percentile(c, q)), 4) for q in (50, 90, 95, 99, 99.9)},
        rule="memory splat vs jittered / blurred memory render (no change); tau_chg = p99 of "
        "max(mean P(missing), mean P(appeared)) over each small item's box mask",
        samples=rows,
    )
    write_json(out, spec["nochange"])
    log({k: v for k, v in out.items() if k != "samples"})


if __name__ == "__main__":
    main()
