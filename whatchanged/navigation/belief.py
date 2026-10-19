"""Incremental belief of the simulated robot (sim.py).

* Object evidence kinds per view, with a dead band; thresholds from the checker's calibration and the memory-only
  no-change baseline (calib_nochange.py): gone (calibrated q_gone >= 0.5), changed (max(mean P(missing), mean
  P(appeared)) over the mask >= tau_chg = p99 of the no-change baseline; weak: q = 0.7), present (q_gone < 0.1 and score
  < tau_present = p90), else ambiguous (no update). Views are weighted by novelty (15 cm or 15 deg of pose change since
  that object's last used view = a full new view).
* Appearance scores (appearance.py: colour difference cd and DINOv2 embedding distance fd inside the small item's tight
  box mask, memory render vs episode render) count as 'changed' above their no-change p99 and block 'present' above
  their p90.
* Objects: memory-visible pixels (rendered id, known, >= 16 px from the border) >= min_vis_px -> q_gone / q_cov
  (checker confidence calibration) -> gone / covered / present evidence; L += w x logit(q_gone), w = min(1, dt /
  decorrelate_s) x min(1, (visible / full silhouette) / clear_vis), full silhouette = the object rendered alone on a
  canvas with twice the field of view; state present (L <= -l_conf) / gone (L >= l_conf) / unverified / not observed.
* Walkable map = observed occupancy from the sensor depth of every tick (the checker is not used for obstacles); the
  memory map is the prior for unseen cells. Hit / miss log-odds per cell with the standard inverse sensor model, at
  most 0.75 l_conf per view. New obstacle = L >= l_conf on a cell the memory saw free, away from memory-occupied cells,
  whose obstacle points are mostly unexplained by the memory render; memory-unknown cells seen occupied are occupied,
  not 'new'; freed = memory-occupied cells with L <= -l_conf; unknown floor seen -> free; the robot's own footprint is
  free.
* Seen cells = map cells hit by the sensor's depth (any height), for the minimap and the search coverage.
* Target re-identification: connected 'new here' regions (P(appeared) >= 0.5, >= 300 px) lifted with the sensor depth,
  fingerprinted against the target's reference (whatchanged.inventory.fingerprint.score; references pooled on
  memory-splat renders, the sensor's domain, sim.memory_refs; memory box size) and accepted at the held-out calibrated
  threshold (precision 0.95); a region within r_ref of the remembered place is not 'elsewhere'; clusters (0.2 m)
  accumulate w x logit(0.95).
* Target state: confirmed / missing (place gone, nothing found) / moved (place gone + strong cluster) / seen elsewhere /
  unverified / not yet seen. p_moved = P(place gone) x P(cluster).
Search coverage: U = surface cells (memory free floor + occupied cells up to 1.9 m) in the look-around cover of every
reachable search candidate (nbv.py tables); checked = |U seen| / |U|.
"""

from __future__ import annotations

import json
import math

import cv2
import numpy as np
from scipy import ndimage

from whatchanged.common.io import log, read_json, save_npz
from whatchanged.navigation import appearance
from whatchanged.navigation import language as LG
from whatchanged.navigation import planner as PL
from whatchanged.navigation.boxmask import box_mask
from whatchanged.navigation.common import apply, cfg, inventory_sha, logit, nav_dir, scene_spec, sigmoid
from whatchanged.navigation.helpers import los_ok, nearest_blob, remaining, side

P_MATCH = 0.95
BORDER = 16
NEW_RUN = 3  # consecutive perception ticks a new obstacle must persist before it is reported
MIN_REPORT_CELLS = (
    8  # a new obstacle is named in the log / room description from 8 cells (0.02 m2); smaller ones only block
)
Q_PRESENT = 0.1  # presence evidence only from a confident reading (calibrated q_gone < 0.1 and score < tau_present)
Q_CHANGED = 0.7  # weak change evidence: score above the memory-only no-change p99
STEP_MAX = 0.75  # one view moves an object's log-odds by at most 0.75 x l_conf: every decision needs >= 2 views
CALIB_TYPES = {"gone": ("gone", "missing"), "added": ("added", "appeared")}  # older calibrations use gone / added


class Belief:
    def __init__(self, scene, M, info, task, radius, K, W, H):
        C = cfg()
        self.C, self.Cb = C, C["belief"]
        self.M, self.info, self.task = M, info, task
        self.K, self.W, self.H = K, W, H
        self.radius = radius
        self.radius_used = radius
        self.ladder = C["robots"]["shrink_ladder_m"]
        self.reach = C["robot"]["reach_m"]
        self.eye_h = C["robot"]["eye_height_m"]
        self.o, self.cell = M["origin"], float(M["cell"])
        self.nx, self.ny = M["shape"]
        self.band_lo, self.band_hi = info["band_m"]
        self.noise_lo = float(info.get("floor_noise_m", 0.067))
        self.objs = {x["id"]: x for x in info["objects"]}
        self.movable = [x["id"] for x in info["objects"] if x["cls"] != "static"]
        self.gid_of = {int(i[1:]): i for i in self.movable}
        self.L = {i: 0.0 for i in self.movable}
        self.nobs = {i: 0 for i in self.movable}
        self.tlast = {i: -1e9 for i in self.movable}
        self.Ld = np.zeros((self.nx, self.ny), np.float32)
        self.Hd = np.zeros((self.nx, self.ny), np.float32)
        self.Nhit = np.zeros((self.nx, self.ny), np.int32)  # ticks a cell was seen occupied ...
        self.Nunexp = np.zeros(
            (self.nx, self.ny), np.int32
        )  # ... and of those, ticks its mass was not explained by the memory
        self.seen = np.zeros((self.nx, self.ny), bool)
        spec = scene_spec(scene)
        self.cal = read_json(spec["calib"])
        self.refs = {}
        if not spec["move_match"].exists():
            raise FileNotFoundError(
                f"{spec['move_match']}: fingerprint match threshold missing; set move_match or move_match_from for "
                f"scene {scene} in configs/navigation.json"
            )
        self.thr = float(read_json(spec["move_match"])["threshold"])
        self.groups = task["groups"]
        self.main = max(self.groups, key=lambda g: g["weight"]) if self.groups else None
        self.clusters = {g["key"]: [] for g in self.groups}
        self.r_ref = {g["key"]: max(0.15, 0.5 * float(np.linalg.norm(g["size"][:2])) + 0.05) for g in self.groups}
        self.events = []
        self.trace = []
        self.last_pose = {}
        self._gain = None
        self.namer = None
        self.blob_run, self.reported_xy = {}, []
        self.blob_names, self.blob_probs = {}, {}
        nc = read_json(spec["nochange"])
        self.tau_chg, self.tau_present = float(nc["tau_chg"]), float(nc["tau_present"])
        self.tau = {k: (float(nc[f"tau_{k}_99"]), float(nc[f"tau_{k}_90"])) for k in ("cd", "fd")}
        self.covered = {}
        self.gstate = {}
        self.t_prev = None
        self.target_visible_px = 0
        self.free_now, self.occ_now, self.unk_now = M["free"].copy(), M["occ"].copy(), M["unknown"].copy()
        self.surface = M["free"] | (M["occ"] & (M["height"] < 1.9))
        self.emit(0.0, "task", name=task["name"], n_inst=len(self.groups))

    # ------------------------------------------------------------------------------------------ events
    def emit(self, t, kind, **f):
        if kind == "verdict":
            s = f"{f['text']}{self.evidence_text(f.get('verdict'))}"
        elif kind == "room":
            s = f["text"]
        else:
            s = (
                LG.sentence(kind, **f)
                if kind
                in ("task", "target_confirmed", "new_obstacle", "freed", "plan", "detour", "no_path", "arrived", "look")
                else f.get("text", kind)
            )
        if kind == "search_start":
            s = f"The {f['name']} is not at its place; searching the unseen parts of the room."
        if kind == "target_gone":
            s = f"The {f['name']} is not {f['rel']}."
        if kind == "target_moved":
            s = f"Found the {f['name']} {f['rel_new']}, {100 * f['moved_m']:.0f} cm from where I remember it."
        if kind == "shrink":
            s = f"No path for my {f['r0']:.2f} m footprint; trying {f['r1']:.2f} m."
        ev = dict(
            t=round(t, 2), kind=kind, text=s, facts={k: v for k, v in f.items() if isinstance(v, (int, float, str))}
        )
        self.events.append(ev)
        log(f"[{t:5.1f}s] {s}")

    def view_counts(self):
        """independent views of the target's place (novelty-weighted w >= 0.1) by evidence kind."""
        n = dict(present=0, gone=0, changed=0, ambiguous=0)
        g = self.main
        seen_ids = [i for i in g["ids"] if self.nobs[i] > 0] if g else []
        dec = (
            max(seen_ids, key=lambda i: self.L[i]) if seen_ids else None
        )  # the fragment that decides the group (target_state)
        for r in self.trace:
            if r["obj"] == dec and r["w"] >= 0.1:
                n[r["kind"]] += 1
        return n

    def evidence_text(self, verdict):
        """The evidence behind a verdict in plain counts. The log-odds posterior saturates at 99 % after a few views and
        was calibrated on synthetic edits only, so the film states how many separate views agreed instead of a
        percentage."""
        if verdict in (None, "blocked"):
            return ""
        n = self.view_counts()
        tot = sum(n.values())
        if tot == 0:
            return " (no clear view of its place)"
        if verdict in ("confirmed", "reached"):
            agree = n["present"]
        elif verdict in ("not_here", "moved", "reached_moved"):
            agree = n["gone"] + n["changed"]
        else:
            if sum(self.covered.get(i, 0) for i in self.main["ids"]):
                return ""
            return f" ({tot} view{'s' if tot != 1 else ''} of its place, none decisive)"
        return f" ({agree} of {tot} separate views agree)"

    # ------------------------------------------------------------------------------------------ nbv tables
    def load_nbv(self, sensor, P, scene="scene"):
        """visibility / coverage tables from the memory splat (cached per scene + target instance ids + map grid)."""
        import hashlib

        from whatchanged.navigation import nbv

        key = hashlib.md5(
            json_dumps(
                [
                    scene,
                    inventory_sha(scene),
                    [g["gids"] for g in self.groups],
                    list(map(int, self.M["shape"])),
                    float(self.M["origin"][0]),
                    self.eye_h,
                    hashlib.md5(np.packbits(self.M["free"]).tobytes()).hexdigest(),
                ]
            ).encode()
        ).hexdigest()[:12]
        cp = nav_dir(scene) / "nbv_cache" / f"{key}.npz"
        if cp.exists():
            z = np.load(cp)
            self.cand_xy, self.cand_ij = z["cand_xy"], z["cand_ij"]
            self.vis = {g["key"]: z[f"vis_{g['key']}"] for g in self.groups if f"vis_{g['key']}" in z.files}
            cover = np.unpackbits(z["cover"], axis=1)[:, : int(z["n"])].astype(bool)
        else:
            means = sensor.mem["means"].cpu().numpy()
            tab = nbv.tables(
                sensor.mem,
                sensor.gs,
                self.M,
                np.array(P["K"], float),
                P["W"],
                {g["key"]: g["gids"] for g in self.groups},
                means,
                sensor.gid,
                eye_h=self.eye_h,
            )
            self.cand_xy, self.cand_ij = tab["cand_xy"], tab["cand_ij"]
            self.vis = {k: v["rel"] for k, v in tab["vis"].items()}
            cover = tab["cover"]
            save_npz(
                cp,
                cand_xy=self.cand_xy,
                cand_ij=self.cand_ij,
                cover=np.packbits(cover, axis=1),
                n=np.array(cover.shape[1]),
                **{f"vis_{k}": v for k, v in self.vis.items()},
            )
        self.cover = cover.reshape(len(self.cand_xy), self.nx, self.ny) & self.surface[None]
        self.visited_search = np.zeros(len(self.cand_xy), bool)
        log(f"nbv: {len(self.cand_xy)} candidates")

    # ------------------------------------------------------------------------------------------ update
    def update(self, t, TF, rgb, Dq, Dp, inst, S, Pn, Pg, fq, robot, full=None, ren=None, fr=None):
        Cb = self.Cb
        lconf, lmax = Cb["l_conf"], Cb["l_max"]
        w_t = 1.0 if self.t_prev is None else min(1.0, (t - self.t_prev) / Cb["decorrelate_s"])
        self.t_prev = t
        known = S >= 3
        K, W, H = self.K, self.W, self.H
        full = full or {}
        inner = np.zeros_like(known)
        inner[BORDER:-BORDER, BORDER:-BORDER] = True  # the checker's 16 px border is not evidence (as its training)
        known_in = known & inner
        o, c, nx, ny = self.o, self.cell, self.nx, self.ny
        fl = self.M["floor_level"]
        rxy = TF[:2, 3]
        heading = float(math.atan2(TF[1, 2], TF[0, 2]))
        # seen cells (sensor depth)
        P_all, _ = bp(Dq, K, TF, 4)
        ix = np.floor((P_all[:, 0] - o[0]) / c).astype(int)
        iy = np.floor((P_all[:, 1] - o[1]) / c).astype(int)
        g_ = (ix >= 0) & (iy >= 0) & (ix < nx) & (iy < ny) & (np.linalg.norm(P_all[:, :2] - rxy, axis=1) < 4.5)
        self.seen[ix[g_], iy[g_]] = True
        # objects
        ids_here, cnt = np.unique(inst[known_in], return_counts=True)
        tgt_ids = set(self.main["gids"]) if self.main else set()
        self.target_visible_px = 0
        for gk, n in zip(ids_here.tolist(), cnt.tolist()):
            oid = self.gid_of.get(gk)
            if oid is None:
                continue
            if gk in tgt_ids:
                self.target_visible_px += n
            if n < min(Cb["min_vis_px"], Cb["min_box_px"]):
                continue
            if (
                not (self.objs[oid]["cls"] == "items" and max(self.objs[oid]["box"]["size"]) <= 0.6)
                and n < Cb["min_vis_px"]
            ):
                continue
            o_ = self.objs[oid]
            small = o_["cls"] == "items" and max(o_["box"]["size"]) <= 0.6
            if small:  # tight evidence mask: pixels whose memory 3-D point lies in the object's own box
                m = box_mask(Dp, K, TF, o_["box"]) & known_in
                n = int(m.sum())
                if n < Cb["min_box_px"]:
                    continue
            else:
                m = (inst == gk) & known_in
            size = max(self.objs[oid]["box"]["size"])
            qg = calib(self.cal, "gone", float(Pg[m].mean()), size)
            qc = calib(self.cal, "added", float(Pn[m].mean()), size)
            # covered: something now stands in front of the remembered object (the sensor's depth is clearly nearer than
            # the memory's over most of its mask), or the checker reads the mask as 'new here' rather than 'gone'.
            # Either way the object may still be behind it, so this view is no evidence of absence.
            dq_m, dp_m = Dq[m], Dp[m]
            front = float(((dq_m > 0) & (dp_m > 0) & (dq_m < dp_m - (0.05 + 0.03 * dp_m))).mean()) if m.any() else 0.0
            if front >= 0.5 or (qg < 0.5 and qc >= 0.5):
                self.covered[oid] = self.covered.get(oid, 0) + 1
                continue  # covered by something new: can't verify from here
            frac = n / max(
                full.get(gk, n), 1
            )  # visible share of the object's full (untruncated, unoccluded) silhouette
            pg_m, pn_m = float(Pg[m].mean()), float(Pn[m].mean())
            score = max(pg_m, pn_m)  # the no-change baseline's statistic (calib_nochange.py)
            cd = fd = 0.0
            if small and ren is not None:
                if self._gain is None or self._gain[0] != t:
                    self._gain = (
                        t,
                        appearance.gain_fit(ren.astype(np.float32) / 255, rgb.astype(np.float32) / 255, known_in),
                    )
                cd, fd, _ = appearance.scores(ren, rgb, fr, fq, known_in, m, gain=self._gain[1])
            app_chg = cd >= self.tau["cd"][0] or fd >= self.tau["fd"][0]
            app_low = cd < self.tau["cd"][1] and fd < self.tau["fd"][1]
            # independent-view weight: repeated looks from (almost) the same pose are one view (novelty = pose change
            # since this object's last used view: 15 cm or 15 deg = a full new view); plus the time decorrelation and
            # the visible share
            heading = math.atan2(TF[1, 2], TF[0, 2])
            lastp = self.last_pose.get(oid)
            nov = (
                1.0
                if lastp is None
                else min(
                    1.0, np.linalg.norm(rxy - lastp[0]) / 0.15 + abs(ang_wrap(heading - lastp[1])) / math.radians(15)
                )
            )
            w = min(1.0, (t - self.tlast[oid]) / Cb["decorrelate_s"]) * min(1.0, frac / Cb["clear_vis"]) * nov
            # evidence kinds (dead band: a weak, ambiguous reading is not evidence of presence)
            if qg >= 0.5:
                kind, lg = "gone", logit(qg)
            elif score >= self.tau_chg or app_chg:
                kind, lg = (
                    "changed",
                    logit(Q_CHANGED),
                )  # clearly above the memory-only no-change baseline (checker or appearance)
            elif qg < Q_PRESENT and score < self.tau_present and app_low:
                kind, lg = "present", logit(qg)
            else:
                kind, lg = "ambiguous", 0.0
            if w > 1e-3:
                self.tlast[oid] = t
                self.last_pose[oid] = (rxy.copy(), heading)
            step = float(np.clip(w * lg, -STEP_MAX * lconf, STEP_MAX * lconf))  # no decision from a single view
            self.L[oid] = float(np.clip(self.L[oid] + step, -lmax, lmax))
            self.nobs[oid] += 1 if w > 1e-3 else 0
            if gk in tgt_ids:
                self.trace.append(
                    dict(
                        t=round(t, 2),
                        obj=oid,
                        px=int(n),
                        frac=round(frac, 3),
                        mean_pg=round(pg_m, 3),
                        mean_pn=round(pn_m, 3),
                        score=round(score, 3),
                        cd=round(cd, 3),
                        fd=round(fd, 3),
                        q_gone=round(qg, 3),
                        q_cov=round(qc, 3),
                        kind=kind,
                        nov=round(float(nov), 2),
                        w=round(w, 3),
                        L=round(self.L[oid], 2),
                    )
                )
        # ---- observed occupancy (a real robot's map): every rendered depth pixel is back-projected into the 5 cm grid;
        # obstacle = points 10 cm - 1.6 m above the local floor (>= 2 points and >= 10 % of the column's points in a
        # view), free = floor-level points seen with no obstacle points; log-odds with the standard inverse sensor
        # model; the memory map is only the prior for unseen cells.
        st = 4
        v_, u_ = np.mgrid[0:H:st, 0:W:st]
        sub = (Dq[::st, ::st] > 0.3) & (Dq[::st, ::st] < 4.5)
        z = Dq[::st, ::st][sub].astype(np.float64)
        PF = apply(TF, np.c_[(u_[sub] + 0.5 - K[0, 2]) / K[0, 0] * z, (v_[sub] + 0.5 - K[1, 2]) / K[1, 1] * z, z])
        ix = np.floor((PF[:, 0] - o[0]) / c).astype(int)
        iy = np.floor((PF[:, 1] - o[1]) / c).astype(int)
        g_ = (ix >= 0) & (iy >= 0) & (ix < nx) & (iy < ny)
        ix, iy, PF = ix[g_], iy[g_], PF[g_]
        zr = PF[:, 2] - fl[ix, iy]
        band = (zr > self.band_lo) & (zr < self.band_hi)
        flo = np.abs(zr) <= self.noise_lo
        lin = ix * ny + iy
        n_all = np.bincount(lin[band | flo], minlength=nx * ny)
        n_band = np.bincount(lin[band], minlength=nx * ny)
        n_floor = np.bincount(lin[flo], minlength=nx * ny)
        hmax = np.zeros(nx * ny)
        np.maximum.at(hmax, lin[band], zr[band])
        hit = (n_band >= 2) & (n_band >= 0.1 * np.maximum(n_all, 1))
        # is the obstacle mass explained by the memory? A band point is unexplained when the memory render at the same
        # pixel has no surface or a surface clearly behind it (something now stands in front of what was remembered).
        # Cells whose band points are mostly explained are occupied as remembered (the memory map's footprints are only
        # a few cm off), so they never become 'new'.
        dp = Dp[::st, ::st][sub].astype(np.float64)[g_]
        unexp = (dp <= 0) | (z[g_] < dp - (0.05 + 0.03 * z[g_]))
        n_unexp = np.bincount(lin[band & unexp], minlength=nx * ny)
        Nh, Nu = self.Nhit.reshape(-1), self.Nunexp.reshape(-1)
        Nh[hit] += 1
        Nu[hit & (n_unexp >= 0.5 * n_band)] += 1
        miss = (n_floor >= 3) & ~hit
        cap = STEP_MAX * lconf  # no cell decision from a single view
        Ldf, Hf = self.Ld.reshape(-1), self.Hd.reshape(-1)
        Ldf[hit] = np.clip(Ldf[hit] + min(cap, w_t * logit(Cb["cell_hit"])), -lmax, lmax)
        Ldf[miss] = np.clip(Ldf[miss] + max(-cap, w_t * logit(Cb["cell_miss"])), -lmax, lmax)
        Hf[hit] = np.maximum(Hf[hit], hmax[hit])
        self._grid(rxy)
        # new obstacles / freed space events
        # new obstacles are reported only when persistent (>= NEW_RUN consecutive
        # perception ticks), once, and not within 0.6 m of an already reported one (duplicates merged); named from the
        # best view with the names of memory objects at that spot excluded
        nbs = self.new_blobs()
        keys_now = {b["key"] for b in nbs}
        self.blob_run = {k: self.blob_run.get(k, 0) + 1 for k in keys_now}
        for b in nbs:
            if self.blob_run[b["key"]] < NEW_RUN:
                continue
            cxy = np.array(b["centre"][:2])
            if any(np.linalg.norm(cxy - r) < 0.6 for r in self.reported_xy):
                continue
            self.reported_xy.append(cxy)
            ijc = np.array(b["ij"])
            walk = bool(self.M["free"][ijc[:, 0], ijc[:, 1]].mean() >= 0.5)
            excl = {
                LG.short(o["name"])
                for o in self.info["objects"]
                if o["cls"] != "static" and o["n_gauss"] >= 300 and LG.fp_dist(cxy, o) <= 0.15
            }
            nm = self.blob_name(b, TF, rgb, excl)
            self.emit(
                t,
                "new_obstacle",
                name=nm,
                rel=LG.relation(b, self.info["objects"], floor_cells_free=walk, name=None),
                dist=float(np.linalg.norm(cxy - rxy)),
                dir=LG.direction(rxy, heading, b["centre"]),
            )
        for b in blobs(self.M["occ"] & (self.Ld <= -lconf), self.M, None):
            key = "F{}_{}".format(*np.round(np.array(b["centre"][:2]) / 0.3).astype(int))
            if key not in self.gstate.setdefault("_freed", set()) and b["cells"] >= 8:
                self.gstate["_freed"].add(key)
                ijc = np.array(b["ij"])
                own = self.M["owner"][ijc[:, 0], ijc[:, 1]]
                own = own[own > 0]
                if not len(own):
                    continue  # freed static / unlabelled space: on the map, not in the log
                nm = self.info["objects"][int(np.bincount(own).argmax()) - 1]["name"]
                if nm in self.gstate.setdefault("_freed_names", set()):
                    continue
                self.gstate["_freed_names"].add(nm)
                self.emit(t, "freed", name=nm)
        # target re-identification on 'new here' regions
        if self.groups:
            self._reid(t, TF, rgb, Dq, Pn, fq, w_t)
        # target state change events
        g, s = self.target_state()
        if g is not None and s["state"] != self.gstate.get("target"):
            if s["state"] == "confirmed":
                self.emit(
                    t,
                    "target_confirmed",
                    name=g["name"],
                    rel=LG.relation(g_box(g), self.info["objects"], name=g["name"]),
                    dist=float(np.linalg.norm(np.array(g["centre"][:2]) - rxy)),
                    dir=LG.direction(rxy, heading, g["centre"]),
                )
            elif s["state"] in ("missing", "moved") and self.gstate.get("target") not in ("missing", "moved"):
                self.emit(
                    t,
                    "target_gone",
                    name=g["name"],
                    rel=LG.relation(g_box(g), self.info["objects"], name=g["name"]) or "where I remember it",
                    p=s["p_gone"],
                )
            if s["state"] == "moved":
                self.emit(t, "target_moved", name=g["name"], rel_new=s["rel_new"], moved_m=s["moved_m"])
            self.gstate["target"] = s["state"]

    def _grid(self, rxy):
        lconf = self.Cb["l_conf"]
        M = self.M
        # New obstacle (reported, red): observed occupied on floor the memory saw free and not within 10 cm of
        # memory-occupied cells (the memory's furniture footprints are a few cm smaller than what the sensor sees, so
        # their rims are not 'new'). Planning uses every observed-occupied cell regardless (below: occ includes (Ld >=
        # lconf)).
        near_mem = ndimage.binary_dilation(M["occ"], iterations=2)
        new_occ = (self.Ld >= lconf) & M["free"] & ~near_mem & (self.Nunexp >= 0.5 * np.maximum(self.Nhit, 1))
        seen_occ = (self.Ld >= lconf) & M["unknown"]  # memory-unknown space now seen occupied: occupied, not "new"
        freed = M["occ"] & (self.Ld <= -lconf)
        seen_floor = M["unknown"] & (self.Ld <= -lconf)
        free = (M["free"] | freed | seen_floor) & ~((self.Ld >= lconf) & M["free"])
        occ = (
            (M["occ"] & ~freed) | ((self.Ld >= lconf) & M["free"]) | seen_occ
        )  # planning: every observed obstacle blocks
        unk = M["unknown"] & ~seen_floor & ~seen_occ
        disc = disc_mask(M, rxy, self.radius_used)
        free |= disc & unk
        unk &= ~disc
        self.free_now, self.occ_now, self.unk_now, self.new_occ = free, occ, unk, new_occ

    def _reid(self, t, TF, rgb, Dq, Pn, fq, w_t):
        from whatchanged.common import geometry as G
        from whatchanged.common.features import pool
        from whatchanged.inventory.fingerprint import score as fp_score

        K = self.K
        m_all = (Pn >= 0.5) & (Dq > 0)
        lab, n = ndimage.label(cv2.morphologyEx(m_all.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)))
        for k in range(1, n + 1):
            m = lab == k
            if m.sum() < 300:
                continue
            vv, uu = np.nonzero(m)
            zz = Dq[vv, uu].astype(np.float64)
            zm = np.median(zz)
            keep = np.abs(zz - zm) < 0.15 * zm
            P3 = apply(
                TF,
                np.c_[
                    (uu[keep] + 0.5 - K[0, 2]) / K[0, 0] * zz[keep],
                    (vv[keep] + 0.5 - K[1, 2]) / K[1, 1] * zz[keep],
                    zz[keep],
                ],
            )
            if len(P3) < 30:
                continue
            cB, sB, _ = G.obb_zup(P3, 5, 95)
            emb = pool(fq, m)
            if emb is None:
                continue
            col = rgb[m].astype(np.float32).mean(0) / 255
            for g in self.groups:
                rid = next((i for i in g["ids"] if i in self.refs), None)
                if rid is None:
                    continue
                s = fp_score(
                    emb, col, sB, dict(emb=np.array(self.refs[rid]["emb"]), col=self.refs[rid]["col"], size=g["size"])
                )
                if s < self.thr:
                    continue
                if any(
                    np.linalg.norm(np.array(g2["centre"][:2]) - cB[:2]) < self.r_ref[g2["key"]] for g2 in self.groups
                ):
                    continue
                # best-match assignment: a region at or near another remembered object (within 0.5 m) that fingerprints
                # at least as well as that object is that object, slightly moved, not the target (a pipe wrench nudged
                # on its cabinet is not the hammer)
                tgt_names = {self.objs[i]["name"] for i in g["ids"]}
                rival = False
                for oid_, ref_ in self.refs.items():
                    o_ = self.objs.get(oid_)
                    if o_ is None or oid_ in g["ids"] or o_["name"] in tgt_names or o_["cls"] == "static":
                        continue
                    if np.linalg.norm(np.array(o_["box"]["centre"][:2]) - cB[:2]) > 0.5:
                        continue
                    s2 = fp_score(
                        emb, col, sB, dict(emb=np.array(ref_["emb"]), col=ref_["col"], size=o_["box"]["size"])
                    )
                    if s2 >= s:
                        rival = True
                        break
                if rival:
                    continue
                # a region inside another remembered object's footprint is that object until shown otherwise (thin objects
                # such as a pipe wrench rarely get a fingerprint reference, so the best-match test above cannot always speak for them)
                if any(
                    LG.fp_dist(cB[:2], o_) <= 0.05
                    and o_["name"] not in tgt_names
                    and o_["cls"] != "static"
                    and o_["n_gauss"] >= 300
                    and abs(cB[2] - o_["box"]["centre"][2]) <= 0.5 * o_["box"]["size"][2] + 0.1
                    for o_ in self.info["objects"]
                ):
                    continue
                cl = next(
                    (x for x in self.clusters[g["key"]] if np.linalg.norm(np.array(x["centre"][:2]) - cB[:2]) < 0.2),
                    None,
                )
                if cl is None:
                    cl = dict(centre=cB.tolist(), size=sB.tolist(), L=0.0, n=0, t_last=-1e9, score=0.0)
                    self.clusters[g["key"]].append(cl)
                wc = min(1.0, (t - cl["t_last"]) / self.Cb["decorrelate_s"])
                cl["L"] = float(min(self.Cb["l_max"], cl["L"] + wc * logit(P_MATCH)))
                cl["n"] += 1
                cl["t_last"] = t
                cl["score"] = max(cl["score"], s)
                a = 1.0 / cl["n"]
                cl["centre"] = list((1 - a) * np.array(cl["centre"]) + a * cB)

    # ------------------------------------------------------------------------------------------ queries
    def obj_state(self, oid):
        if self.nobs.get(oid, 0) == 0:
            return "not observed"
        L = self.L[oid]
        lc = self.Cb["l_conf"]
        return "present" if L <= -lc else ("gone" if L >= lc else "unverified")

    def target_state(self):
        g = self.main
        if g is None:
            return None, {}
        lc = self.Cb["l_conf"]
        fr = (
            [i for i in g["ids"] if self.objs[i]["n_gauss"] >= 100 and self.nobs[i] > 0]
            or [i for i in g["ids"] if self.nobs[i] > 0]
            or g["ids"]
        )
        Lg = max(
            self.L[i] for i in fr
        )  # a group is as changed as its most changed fragment (a plate is not the banana)
        ng = sum(self.nobs[i] for i in g["ids"])
        strong = [x for x in self.clusters[g["key"]] if x["L"] >= lc]
        cl = max(strong, key=lambda x: x["L"]) if strong else None
        if Lg <= -lc:
            s = "confirmed"
        elif Lg >= lc:
            s = "moved" if cl else "missing"
        elif cl:
            s = "seen elsewhere"
        elif ng == 0:
            s = "not yet seen"
        else:
            s = "unverified"
        loc = cl["centre"] if cl is not None and s in ("moved", "seen elsewhere") else g["centre"]
        out = dict(
            state=s,
            L=round(Lg, 2),
            p_present=round(float(sigmoid(-Lg)), 3),
            p_gone=round(float(sigmoid(Lg)), 3),
            n_obs=ng,
            loc=np.round(loc, 3).tolist(),
            n_elsewhere=cl["n"] if cl else 0,
        )
        if cl is not None:
            out["moved_m"] = round(float(np.linalg.norm(np.array(cl["centre"][:2]) - np.array(g["centre"][:2]))), 3)
            out["rel_new"] = (
                LG.relation(dict(centre=cl["centre"], size=g["size"], yaw=g["yaw"]), self.info["objects"]) or "nearby"
            )
            out["p_moved"] = round(float(sigmoid(Lg) * sigmoid(cl["L"])), 3)
        return g, out

    def new_blobs(self):
        out = blobs(self.new_occ, self.M, self.Hd) if hasattr(self, "new_occ") else []
        out = [b for b in out if min(b["size"][:2]) >= 0.10 and b["cells"] >= MIN_REPORT_CELLS]
        for b in out:
            b["key"] = "N{}_{}".format(*np.round(np.array(b["centre"][:2]) / 0.3).astype(int))
            b["name"] = self.blob_names.get(b["key"], "object")
        return out

    def reported_blobs(self):
        return [
            b
            for b in self.new_blobs()
            if any(np.linalg.norm(np.array(b["centre"][:2]) - r) < 0.6 for r in self.reported_xy)
        ]

    def blob_name(self, b, TF, rgb, exclude=()):
        """SigLIP 2 name of a newly observed obstacle from the episode render (sensor view); 'object' without a namer or
        below p 0.3."""
        if self.namer is None:
            return "object"
        box = dict(
            centre=b["centre"],
            size=[max(b["size"][0], 0.1), max(b["size"][1], 0.1), max(b["size"][2], 0.1)],
            yaw=b["yaw"],
        )
        nm, p = self.namer.name(rgb, box, TF, self.K, exclude)
        self.blob_names[b["key"]] = nm
        self.blob_probs[b["key"]] = round(p, 3)
        return nm

    def standable(self, xy):
        blocked = self.occ_now | self.unk_now | ~self.free_now
        i, j = PL.to_cell(xy, self.o, self.cell, self.M["shape"])
        cl = PL.clearance(blocked, self.cell)
        return bool(cl[i, j] > min(self.radius_used, 0.1) or disc_mask(self.M, xy, 0.05)[i, j])

    def dist_body_loc(self, robot, loc):
        return max(0.0, float(np.linalg.norm(np.asarray(loc, float)[:2] - robot.xy)) - self.radius_used)

    def dist_body(self, robot, g):
        return max(0.0, float(np.linalg.norm(np.array(g["centre"][:2]) - robot.xy)) - self.radius_used)

    def relation_text(self, g):
        return LG.relation(g_box(g), self.info["objects"], name=g["name"]) or "at its place"

    # ------------------------------------------------------------------------------------------ planning
    def _plan(self, robot, goal_fn, label):
        """plan with the shrink ladder (reported)."""
        res = None
        for r in [x for x in self.ladder if x <= self.radius + 1e-9]:
            res = PL.plan(
                self.free_now,
                self.occ_now,
                self.unk_now,
                self.o,
                self.cell,
                robot.xy,
                goal_fn,
                r,
                low=self.M["low"] if "low" in self.M else None,
            )
            if res["status"] == "ok":
                if r < self.radius_used - 1e-9:
                    self.emit(self.t_prev or 0.0, "shrink", r0=self.radius_used, r1=r)
                    self.radius_used = r
                return res
        return res

    def plan_to(self, robot, mode, g=None, loc=None, exclude=()):
        reach_r = self.radius_used + self.reach
        lo_d, hi_d = self.C["sim"]["look_dist_m"]
        if mode == "reach":
            tgt = np.array(loc if loc is not None else g["centre"], float)
            c = tgt[:2]
            res = self._plan(robot, lambda xy: np.linalg.norm(xy - c, axis=1) <= reach_r, "reach")
            return res, tgt
        if mode in ("look", "look_at_loc"):
            tgt = np.array(loc if loc is not None else g["centre"], float)
            vis = self.vis.get(g["key"], np.ones(len(self.cand_xy))) if mode == "look" else np.ones(len(self.cand_xy))
            d = np.linalg.norm(self.cand_xy - tgt[:2], axis=1)
            ok = (d >= lo_d) & (d <= hi_d) & (vis >= self.C["task"]["nbv_min_vis"])
            ok &= self._los(tgt) & self.reachable_cands(robot)
            for e in exclude or []:
                ok &= np.linalg.norm(self.cand_xy - np.asarray(e, float)[:2], axis=1) > 0.3
            if not exclude and self._here_ok(robot, tgt, vis):
                return dict(
                    status="ok", path_xy=[robot.xy.round(3).tolist(), robot.xy.round(3).tolist()], length=0.0
                ), tgt
            order = np.argsort(-(vis - 0.15 * np.linalg.norm(self.cand_xy - robot.xy, axis=1)))
            last = None
            for j in order[:12]:
                if not ok[j]:
                    continue
                p = self.cand_xy[j]
                res = self._plan(robot, lambda xy, p=p: np.linalg.norm(xy - p, axis=1) <= 0.1, "look")
                last = res
                if res["status"] == "ok":
                    return res, tgt
            if last is None:
                return None, tgt
            return last, tgt
        if mode == "search":
            R = self.reachable_cands(robot)
            gain = (self.cover & ~self.seen[None]).reshape(len(self.cand_xy), -1).sum(1).astype(float)
            sc = gain / max(gain.max(), 1) - 0.1 * np.linalg.norm(self.cand_xy - robot.xy, axis=1)
            sc[self.visited_search | ~R] = -1e9
            for j in np.argsort(-sc)[:10]:
                if gain[j] <= 0 or sc[j] < -1e8:
                    break
                p = self.cand_xy[j]
                res = self._plan(robot, lambda xy, p=p: np.linalg.norm(xy - p, axis=1) <= 0.1, "search")
                if res["status"] == "ok":
                    self.visited_search[j] = True
                    return res, np.r_[p + 0.8 * np.array([math.cos(robot.yaw), math.sin(robot.yaw)]), 0.5]
                self.visited_search[j] = True
            return None, None
        raise ValueError(mode)

    def _here_ok(self, robot, tgt, vis):
        d = float(np.linalg.norm(robot.xy - tgt[:2]))
        lo_d, hi_d = self.C["sim"]["look_dist_m"]
        if not (lo_d <= d <= hi_d) or not len(self.cand_xy):
            return False
        j = int(np.argmin(np.linalg.norm(self.cand_xy - robot.xy, axis=1)))
        return bool(
            np.linalg.norm(self.cand_xy[j] - robot.xy) < 0.2
            and vis[j] >= self.C["task"]["nbv_min_vis"]
            and self._los(tgt)[j]
        )

    def _los(self, tgt):
        if not hasattr(self, "new_occ"):
            return np.ones(len(self.cand_xy), bool)
        return los_ok(self.cand_xy, np.asarray(tgt, float), self.new_occ, self.Hd, self.M, self.eye_h) > 0

    def replan_if_needed(self, t, robot, path):
        rest = remaining(path, robot.xy)
        if len(rest) < 2:
            return path
        bad, where = PL.path_blocked(
            rest[1:] if len(rest) > 2 else rest,
            self.free_now,
            self.occ_now,
            self.unk_now,
            self.o,
            self.cell,
            self.radius_used - 0.02,
        )
        if not bad:
            return path
        goal = np.array(rest[-1])
        res = self._plan(robot, lambda xy: np.linalg.norm(xy - goal, axis=1) <= 0.12, "replan")
        if res is None or res["status"] != "ok":
            key = tuple(np.round(goal, 1))
            if getattr(self, "_nopath_goal", None) != key:  # say it once per goal, not every tick
                self._nopath_goal = key
                self.emit(
                    t,
                    "text",
                    text=f"The way ahead looks narrower than my {self.radius_used:.2f} m footprint; "
                    f"keeping to the last plan ({float(np.linalg.norm(goal - robot.xy)):.1f} m to go).",
                )
            return rest
        ob = nearest_blob(where, self.new_blobs())
        self.emit(
            t,
            "detour",
            name="new obstacle" if ob else "obstacle",
            side=side(rest, res["path_xy"], where),
            extra=max(0.0, res["length"] - PL.length_m(rest)),
        )
        return res["path_xy"]

    def blocked_text(self, g, res=None):
        sq = (res or {}).get("squeeze")
        what = None
        if sq:
            p = np.array(sq["narrowest_xy"])
            bl = self.reported_blobs()
            if bl:
                b = min(bl, key=lambda b: np.linalg.norm(np.array(b["centre"][:2]) - p))
                if np.linalg.norm(np.array(b["centre"][:2]) - p) < 0.6:
                    what = "a new obstacle " + (LG.relation(b, self.info["objects"], floor_cells_free=True) or "")
            if what is None:
                F = LG.furniture(self.info["objects"])
                if F:
                    f = min(F, key=lambda f: LG.fp_dist(p, f))
                    what = f"the {LG.short(f['name'])} (gap {2 * sq['min_clear_m']:.2f} m wide)"
        # no squeeze path: name the new obstacle standing closest to the target, if any
        if what is None and g is not None:
            bl = self.reported_blobs()
            c = np.array(g["centre"][:2], float)
            near = [b for b in bl if np.linalg.norm(np.array(b["centre"][:2]) - c) < 1.0]
            if near:
                b = min(near, key=lambda b: np.linalg.norm(np.array(b["centre"][:2]) - c))
                nm = self.blob_names.get(b["key"], "object")
                what = f"a new {'obstacle' if nm == 'object' else nm} next to it"
        what = what or "space I could not see or that is occupied"
        return f"No path to the {g['name']} for a {self.ladder[-1]:.2f} m footprint, blocked by {what.strip()}."

    def reachable_cands(self, robot):
        """search candidates reachable now, with the smallest footprint of the shrink ladder (cached per map state)."""
        blocked = self.occ_now | self.unk_now | ~self.free_now
        cl = PL.clearance(blocked, self.cell)
        r = min(self.ladder)
        stand = ~blocked & (cl > r)
        s0 = PL.to_cell(robot.xy, self.o, self.cell, self.M["shape"])
        if not stand[s0]:
            cand = np.argwhere(stand)
            if not len(cand):
                return np.zeros(len(self.cand_xy), bool)
            d = np.linalg.norm((cand - s0) * self.cell, axis=1)
            if d.min() > PL.SNAP_M:
                return np.zeros(len(self.cand_xy), bool)
            s0 = tuple(cand[int(np.argmin(d))])
        R = PL.reachable(stand, s0)
        self._reach_U = (
            self.cover[R[self.cand_ij[:, 0], self.cand_ij[:, 1]]].any(0) if R.any() else np.zeros_like(self.seen)
        )
        return R[self.cand_ij[:, 0], self.cand_ij[:, 1]]

    def _U(self):
        return (
            getattr(self, "_reach_U", None)
            if getattr(self, "_reach_U", None) is not None and self._reach_U.any()
            else self.cover.any(0)
        )

    def hidden_text(self, g):
        c = sum(self.covered.get(i, 0) for i in g["ids"])
        rel = self.relation_text(g)
        st = self.target_state()[1]
        if st.get("p_gone", 0) > 0.55 and not c:
            return (
                f"Not confirmed: the {g['name']} looks changed at its place ({rel}) - "
                f"{100 * st['p_gone']:.0f} % sure it is not there, but I did not find it elsewhere."
            )
        if c:
            return f"Could not verify the {g['name']} ({rel}): something new is standing in front of its place."
        return f"Could not see the {g['name']}'s place ({rel}) from any reachable view."

    def search_gain_frac(self):
        U = self._U()
        if not U.any():
            return 0.0
        return float((U & ~self.seen).sum() / U.sum())

    def checked_frac(self):
        """share of the surface cells that any REACHABLE viewpoint can see which the robot has actually seen."""
        U = self._U()
        return float((U & self.seen).sum() / max(U.sum(), 1))

    # ------------------------------------------------------------------------------------------ language
    def describe(self, t, when):
        pres = [i for i in self.movable if self.obj_state(i) == "present"]
        gone = [i for i in self.movable if self.obj_state(i) == "gone" and self.objs[i]["n_gauss"] >= 1500]
        unseen = [i for i in self.movable if self.obj_state(i) == "not observed"]
        nb = self.reported_blobs()
        parts = []
        for i in sorted(gone, key=lambda i: -self.objs[i]["n_gauss"])[:3]:
            rel = LG.relation(self.objs[i]["box"], self.info["objects"], name=self.objs[i]["name"])
            parts.append(f"the {LG.short(self.objs[i]['name'])} is not {rel or 'at its place'}")
        for b in nb[:2]:
            ijc = np.array(b["ij"])
            walk = bool(self.M["free"][ijc[:, 0], ijc[:, 1]].mean() >= 0.5)
            nmb = self.blob_names.get(b["key"], "object")
            rel = LG.relation(b, self.info["objects"], floor_cells_free=walk)
            parts.append(
                f"a new {'obstacle' if nmb == 'object' else nmb} {rel or 'on the floor'} "
                f"({b['size'][0]:.1f} x {b['size'][1]:.1f} m)"
            )
        g, s = self.target_state()
        if s.get("state") == "moved":
            parts.append(f"the {g['name']} moved {100 * s['moved_m']:.0f} cm")
        head = "From the doorway" if when == "start" else "At the end"
        ch = ("; ".join(parts) + ".") if parts else "no changes seen."
        txt = (
            f"{head}: {ch[:1].upper() + ch[1:]} {len(pres)} objects look as remembered, {len(unseen)} not seen"
            f"{' yet' if when == 'start' else ''}; "
            f"{100 * self.seen.sum() / max(self.surface.sum(), 1):.0f} % of the room seen."
        )
        self.emit(t, "room", text=txt, when=when, n_changes=len(parts), n_present=len(pres), n_unseen=len(unseen))

    # ------------------------------------------------------------------------------------------ drawing
    def minimap_img(self, robot, path, scale=3, trace=None, final=False, memory_only=False):
        """Top-down map: grey = free floor not yet seen, light = seen free, dark = memory-occupied (lighter blue-grey
        once seen), solid red with white outline = newly observed obstacles, green = freed space (memory-occupied, now
        seen empty), yellow = path / trace, robot + field-of-view cone."""
        M = self.M
        nx, ny = self.nx, self.ny
        lc = self.Cb["l_conf"]
        img = np.zeros((ny, nx, 3), np.uint8)
        img[:] = (30, 32, 36)
        if memory_only:
            img[M["unknown"].T] = (86, 88, 94)
            img[M["free"].T] = (150, 150, 150)
            img[M["occ"].T] = (72, 74, 78)
            freed = np.zeros_like(M["occ"])
            new_occ = np.zeros_like(M["occ"])
        else:
            freed = M["occ"] & (self.Ld <= -lc)
            new_occ = self.new_occ if hasattr(self, "new_occ") else np.zeros_like(M["occ"])
            fr = self.free_now
            img[M["unknown"].T] = (86, 88, 94)
            img[(fr & ~self.seen).T] = (150, 150, 150)
            img[(fr & self.seen).T] = (214, 228, 246)
            img[(self.occ_now & ~new_occ).T] = (72, 74, 78)
            img[(self.occ_now & ~new_occ & self.seen).T] = (104, 118, 142)
            img[freed.T] = (80, 200, 110)
            img[new_occ.T] = (232, 48, 40)
        img = cv2.resize(img[::-1], (nx * scale, ny * scale), interpolation=cv2.INTER_NEAREST)
        o, c = self.o, self.cell
        for msk, col in ((new_occ, (255, 255, 255)), (freed, (20, 120, 50))):
            if msk.any():
                m8 = cv2.resize(msk.T[::-1].astype(np.uint8), (nx * scale, ny * scale), interpolation=cv2.INTER_NEAREST)
                cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, col, 2, cv2.LINE_AA)

        def w2p(xy):
            xy = np.atleast_2d(xy)
            return np.c_[(xy[:, 0] - o[0]) / c * scale, (ny - (xy[:, 1] - o[1]) / c) * scale]

        lw = max(1, scale // 3)
        if trace is not None and len(trace) > 1:
            cv2.polylines(img, [w2p(trace).astype(np.int32)], False, (255, 225, 70), lw + 1, cv2.LINE_AA)
        if path is not None and len(path) > 1:
            cv2.polylines(img, [w2p(np.asarray(path)).astype(np.int32)], False, (255, 225, 70), lw + 1, cv2.LINE_AA)
        g, s = self.target_state()
        if g is not None:
            colr = {"confirmed": (52, 199, 89), "missing": (235, 64, 52), "moved": (235, 64, 52)}.get(
                s["state"], (170, 170, 170)
            )
            p = w2p(np.array(g["centre"][:2]))[0].astype(int)
            cv2.circle(img, tuple(p), 3 * lw + 3, colr, lw + 1, cv2.LINE_AA)
            if s["state"] in ("moved", "seen elsewhere"):
                q = w2p(np.array(s["loc"][:2]))[0].astype(int)
                cv2.circle(img, tuple(q), 3 * lw + 3, (255, 184, 0), lw + 1, cv2.LINE_AA)
                cv2.arrowedLine(img, tuple(p), tuple(q), (255, 184, 0), lw, cv2.LINE_AA, tipLength=0.2)
        if final:
            for i in self.movable:
                stt = self.obj_state(i)
                if self.objs[i]["n_gauss"] < 1500 or stt == "unverified":
                    continue
                colr = {"present": (52, 199, 89), "gone": (235, 64, 52)}.get(stt, (150, 150, 150))
                p = w2p(np.array(self.objs[i]["box"]["centre"][:2]))[0].astype(int)
                cv2.circle(img, tuple(p), lw + 2, colr, -1 if stt != "not observed" else lw, cv2.LINE_AA)
        if robot is not None and not memory_only:
            p = w2p(robot.xy)[0]
            h = robot.yaw + robot.pan
            fov = math.atan2(self.W / 2, self.K[0, 0])
            rad = 1.6 / c * scale  # 1.6 m FOV cone
            cone = np.array(
                [
                    p,
                    p + rad * np.array([math.cos(h + fov), -math.sin(h + fov)]),
                    p + rad * np.array([math.cos(h - fov), -math.sin(h - fov)]),
                ]
            )
            ov = img.copy()
            cv2.fillConvexPoly(ov, np.round(cone).astype(np.int32), (255, 255, 160), cv2.LINE_AA)
            img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
            sz = 4 * lw + 6
            tip = p + sz * np.array([math.cos(h), -math.sin(h)])
            l = p + 0.65 * sz * np.array([math.cos(h + 2.5), -math.sin(h + 2.5)])
            r = p + 0.65 * sz * np.array([math.cos(h - 2.5), -math.sin(h - 2.5)])
            cv2.fillConvexPoly(img, np.round([tip, l, p, r]).astype(np.int32), (255, 255, 255), cv2.LINE_AA)
            cv2.circle(
                img, tuple(p.astype(int)), max(2, int(self.radius_used / c * scale)), (255, 255, 255), 1, cv2.LINE_AA
            )
        return img

    def snapshot(self):
        g, s = self.target_state()
        return dict(
            target=s,
            objects={
                i: dict(name=self.objs[i]["name"], state=self.obj_state(i), L=round(self.L[i], 2), n_obs=self.nobs[i])
                for i in self.movable
                if self.nobs[i] > 0
            },
            new_obstacles=[{k: v for k, v in b.items() if k != "ij"} for b in self.new_blobs()],
            clusters=self.clusters,
            target_obs=self.trace,
            seen_frac=round(float(self.seen[self.surface].mean()), 4),
            checked=round(self.checked_frac(), 4),
            radius_used=self.radius_used,
        )


# ------------------------------------------------------------------------------------------------ helpers
def json_dumps(x):
    return json.dumps(x, default=float)


def ang_wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def g_box(g):
    return dict(centre=g["centre"], size=g["size"], yaw=g["yaw"])


def calib(cal: dict, typ: str, mean_p: float, size: float) -> float:
    """Calibrated probability of a change type from the mean class probability over a mask and the object size."""
    key = next(k for k in CALIB_TYPES[typ] if k in cal["types"])
    w = cal["types"][key]["weights"]
    return float(sigmoid(w[0] * mean_p + w[1] * np.log(max(size, 0.02)) + w[2]))


def bp(D, K, T, step):
    H, W = D.shape
    v, u = np.mgrid[0:H:step, 0:W:step]
    z = D[::step, ::step].astype(np.float64)
    ok = z > 0
    x = (u[ok] + 0.5 - K[0, 2]) / K[0, 0] * z[ok]
    y = (v[ok] + 0.5 - K[1, 2]) / K[1, 1] * z[ok]
    return apply(T, np.c_[x, y, z[ok]]), ok


def disc_mask(M, xy, r):
    o, c = M["origin"], float(M["cell"])
    nx, ny = M["shape"]
    ii, jj = np.mgrid[0:nx, 0:ny]
    return (o[0] + (ii + 0.5) * c - xy[0]) ** 2 + (o[1] + (jj + 0.5) * c - xy[1]) ** 2 <= r * r


def blobs(mask, M, Hd, min_cells=3):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    out = []
    o, c = M["origin"], float(M["cell"])
    for k in range(1, n + 1):
        ij = np.argwhere(lab == k)
        if len(ij) < min_cells:
            continue
        xy = o + (ij + 0.5) * c
        (cx, cy), (w, h), ang = cv2.minAreaRect(xy.astype(np.float32))
        top = float(np.max(Hd[lab == k])) if Hd is not None else 0.3
        out.append(
            dict(
                cells=int(len(ij)),
                centre=[round(cx, 3), round(cy, 3), round(top / 2, 3)],
                size=[round(max(w, c), 3), round(max(h, c), 3), round(max(top, 0.05), 3)],
                yaw=round(float(np.radians(ang)), 3),
                ij=ij.tolist(),
            )
        )
    return out
