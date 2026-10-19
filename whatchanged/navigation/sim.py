"""Closed-loop robot run with a simulated head camera (splat env, GPU; one process renders, runs DINOv2 + the learned
checker, updates the belief and plans).

    python -m whatchanged.navigation.run --memory config1 --episode config1_after --prompt "check the hammer is here" \
        --run r01

The robot is a camera at eye height on a round footprint. Its sensor renders the episode world splat (world.py) from its
current pose only; its memory is the memory splat + inventory + walkable map. Nothing else from the episode is used (the
episode walk's real frame nearest the doorway pose is only shown next to the render, to show that the simulated view
matches reality).

Perception per tick (every sim.perceive_every frames): episode render (RGB + depth = the sensor), memory render at the
same pose (raw colour, depth, inventory id map, view support), learned checker (whatchanged.checker, DINOv2 PCA-128
features, with the episode render as the real image and its depth as the observed depth) -> P(appeared), P(missing).
Belief (belief.py): per-object log-odds from calibrated q_gone / q_cov; walkable-map log-odds from observed depth; seen
cells; target re-identification on 'new here' regions by fingerprint (calibrated threshold); target state.
Behaviour (fixed for every scene and prompt):
  0. start = the first frame of the episode's walk where a standing camera is over memory floor with clearance >= the
     smallest footprint (common.doorway_start); worlds without a walk use the memory walk;
  1. doorway scan: one slow 360 degree turn on the spot; then the room change description (template);
  2. 'check X' / 'go to X': goal = a reachable standing pose with a clear view of X's believed place (nbv visibility
     from the memory splat, 0.6-2 m away, new obstacles occlude in 2.5-D); for 'go to', the goal is within reach (body
     edge <= reach) facing it; look (face + pitch to the place, look_s); confirmed -> done ('go to' then approaches);
  3. place seen empty -> search = next-best-view over likely unseen places (search candidates: reachable, most
     not-yet-seen surface cells in their look-around, minus 0.1 x path length), one full turn at each; a fingerprint
     match elsewhere -> go and look -> 'moved N cm'; nothing after max_search_views or gain < search_min_gain ->
     'not here (checked X % of reachable views)';
  4. no path at the default radius -> the shrink ladder (reported); none at all -> 'no path, blocked by <the obstacle at
     the narrowest point of the squeeze path>'.
Outputs: <runs>/navigation/videos/<scene>_<run>.mp4, <runs>/navigation/qc/<scene>_<run>_summary.jpg (top-down) and
stills, <data>/<scene>/navigation/runs/<run>/{task.json, log.json, eval.json}.
"""

from __future__ import annotations

import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from whatchanged.common.io import log, read_json, write_json
from whatchanged.common.scene import Scene
from whatchanged.navigation import planner as PL
from whatchanged.navigation.common import (
    RUNS,
    cfg,
    frame_file,
    inventory_sha,
    look_at,
    memory_paths,
    nav_dir,
    scene_spec,
)
from whatchanged.navigation.helpers import remaining
from whatchanged.navigation.overlay import CYAN, FONT, FONT_R, TYPE_COL, Labeler, proj, ribbon
from whatchanged.navigation.world import nearest_real

EP_MIN_VIEWS = 3  # a sensor reading needs >= 3 episode cameras on that surface (the view-support rule)


# ------------------------------------------------------------------------------------------------ sensing
class Sensor:
    """The simulated head camera (episode splat) and the robot's memory renders at the same pose."""

    def __init__(self, scene: str, world_splat, mem: dict, K: np.ndarray, W: int, H: int):
        import torch

        from whatchanged.splat import gaussians as gs
        from whatchanged.splat.settings import load_settings
        from whatchanged.splat.support import reference_depths

        self.gs, self.torch = gs, torch
        self.support_cfg = load_settings().support
        self.mem = gs.to_torch(gs.load(mem["splat"]))
        self.Kr, self.Tr, self.Dr = reference_depths(Scene(scene), self.mem)
        self.gid = np.load(mem["gid"])["gid"]
        inv = read_json(mem["inv"])["objects"]
        self.ids = [0] + sorted(int(o["id"][1:]) for o in inv)
        self.ep = gs.to_torch(gs.load(world_splat))
        self.K, self.W, self.H = K, W, H
        self._idx = {}
        # the episode scan's own cameras (its reference depths, already in the memory frame): the simulated sensor only
        # returns a reading where the episode walk actually filmed that surface (>= EP_MIN_VIEWS cameras), like a real
        # depth sensor that never sees reconstruction floaters; the memory itself as the world has no separate scan
        ep_ref = Path(world_splat).parent / "ref_depth.npz"
        self.ep_ref = None
        if Path(world_splat).resolve() != Path(mem["splat"]).resolve() and ep_ref.exists():
            z = np.load(ep_ref)
            self.ep_ref = (z["K"], z["T"], z["D"].astype(np.float32))

    def ep_support(self, D, T_S):
        """Per-pixel count of episode cameras that saw the surface under each pixel (255 everywhere without an episode
        scan)."""
        from whatchanged.splat.support import support_map

        if self.ep_ref is None:
            return np.full(D.shape, 255, np.uint8)
        Kh = self.K.copy()
        Kh[:2] *= 0.5
        Dh = cv2.resize(D, (self.W // 2, self.H // 2), interpolation=cv2.INTER_NEAREST)
        S = support_map(Dh, Kh, T_S, *self.ep_ref, self.support_cfg)
        return cv2.resize(S, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

    def view(self, T_S, half_depth=True):
        r = self.gs.render(self.ep, self.K, T_S, self.W, self.H)
        a = r["alpha"]
        rgb = np.clip(
            r["rgb"] / np.maximum(a, 1e-3)[..., None] * (a[..., None] > 0.05) + (a[..., None] <= 0.05) * 0.08, 0, 1
        )
        return (rgb * 255 + 0.5).astype(np.uint8), r["depth"]

    def memory(self, T_S):
        from whatchanged.splat.support import support_map

        r = self.gs.render(self.mem, self.K, T_S, self.W, self.H)
        a = r["alpha"]
        raw = np.clip(np.where(a[..., None] > 0.05, r["rgb"] / np.maximum(a, 1e-3)[..., None], 0), 0, 1)
        inst = self.gs.render_ids(self.mem, self.gid, self.K, T_S, self.W, self.H, self.ids)
        Kh = self.K.copy()
        Kh[:2] *= 0.5
        Dh = cv2.resize(r["depth"], (self.W // 2, self.H // 2), interpolation=cv2.INTER_NEAREST)
        S = support_map(Dh, Kh, T_S, self.Kr, self.Tr, self.Dr, self.support_cfg)
        S = cv2.resize(S, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        # full silhouettes of the visible objects, rendered alone on a canvas covering 2x the field of view (so
        # truncated objects count their off-frame part): full-resolution pixel units
        ids, cnt = np.unique(inst[inst > 0], return_counts=True)
        Kc = self.K.copy()  # (W/2 x H/2) canvas with 2x the field of view: focal / 4, principal point / 4 + W/8
        Kc[:2] /= 4
        Kc[0, 2] += self.W / 8
        Kc[1, 2] += self.H / 8
        full = {}
        for k, n in zip(ids.tolist(), cnt.tolist()):
            if n < 100:
                continue
            idx = self._idx.get(k)
            if idx is None:
                idx = self._idx.setdefault(
                    k, self.torch.as_tensor(np.flatnonzero(self.gid == k), device=self.mem["means"].device)
                )
            sub = {kk: vv[idx] for kk, vv in self.mem.items()}
            a = self.gs.render(sub, Kc, T_S, self.W // 2, self.H // 2, depth=False)["alpha"]
            full[k] = int((a >= 0.5).sum()) * 16
        return (raw * 255 + 0.5).astype(np.uint8), r["depth"], inst.astype(np.int32), S, full


class ChangeNet:
    """DINOv2 PCA features of both renders and the learned checker's P(appeared), P(missing) per pixel."""

    def __init__(self, spec: dict):
        import torch

        from whatchanged.checker import model as checker_model
        from whatchanged.common.dino_worker import MODEL
        from whatchanged.common.workers import environment

        self.torch = torch
        self.dino = torch.hub.load(str(environment().dino_hub), MODEL, source="local").eval().cuda()
        z = np.load(spec["feats_pca"])
        self.mu, self.comp = torch.as_tensor(z["mean"]).cuda(), torch.as_tensor(z["comp"]).cuda()
        self.model = checker_model.load(spec["checker_ckpt"], "cuda")

    def tokens(self, rgb):
        from whatchanged.common.dino_worker import MEAN, STD, grid_size

        torch = self.torch
        W, H = grid_size(rgb.shape[1], rgb.shape[0])
        x = (cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 - MEAN) / STD
        x = torch.from_numpy(x).permute(2, 0, 1)[None].cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            o = self.dino.forward_features(x)
        tk = torch.nn.functional.normalize(o["x_norm_patchtokens"][0].float(), dim=-1)
        return ((tk - self.mu) @ self.comp).reshape(H // 14, W // 14, -1).permute(2, 0, 1)

    def __call__(self, ren, real, Dp, Dq, known):
        from whatchanged.checker.model import make_low

        torch = self.torch
        fr, fq = self.tokens(ren), self.tokens(real)
        H, W = Dp.shape

        def rs(x, it=cv2.INTER_AREA):
            return cv2.resize(x, (W // 4, H // 4), interpolation=it)

        low = make_low(
            rs(ren),
            rs(real),
            rs(Dp, cv2.INTER_NEAREST),
            rs(Dq.astype(np.float32), cv2.INTER_NEAREST),
            rs(known.astype(np.uint8), cv2.INTER_NEAREST).astype(bool),
        )
        with torch.no_grad():
            p = (
                torch.softmax(self.model(fr[None].float(), fq[None].float(), torch.as_tensor(low)[None].cuda()), 1)[0]
                .cpu()
                .numpy()
            )

        def up(x):
            return cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR)

        return up(p[1]), up(p[2]), fq.cpu().numpy().astype(np.float32), fr.cpu().numpy().astype(np.float32)


def memory_refs(scene, sensor, net, P, ids, step=2, min_px=800):
    """Fingerprint references of inventory objects in the sensor's domain (memory splat renders): DINOv2 PCA-128
    embedding pooled over the object's rendered id mask (>= 800 px) on every 2nd memory frame + mean render colour;
    cached per scene and inventory."""
    from whatchanged.common.features import pool

    # ids are only valid within one inventory, so the cache is keyed by it
    out_p = nav_dir(scene) / f"fingerprint_refs_{inventory_sha(scene)}.json"
    cache = read_json(out_p) if out_p.exists() else {}
    todo = [i for i in ids if i not in cache]
    if not todo:
        return cache
    K = np.array(P["K"], float)
    W, H = P["W"], P["H"]
    acc = {i: [] for i in todo}
    cols = {i: [] for i in todo}
    for f in P["frames"][::step]:
        T = np.array(f["T_wc"])
        inst = sensor.gs.render_ids(sensor.mem, sensor.gid, K, T, W, H, [0] + [int(i[1:]) for i in todo])
        cnt = {int(k): int(n) for k, n in zip(*np.unique(inst, return_counts=True))}
        hit = [i for i in todo if cnt.get(int(i[1:]), 0) >= min_px]
        if not hit:
            continue
        r = sensor.gs.render(sensor.mem, K, T, W, H, depth=False)
        a = r["alpha"]
        rgb = (np.clip(np.where(a[..., None] > 0.05, r["rgb"] / np.maximum(a, 1e-3)[..., None], 0), 0, 1) * 255).astype(
            np.uint8
        )
        fq = net.tokens(rgb).cpu().numpy().astype(np.float32)
        for i in hit:
            m = inst == int(i[1:])
            e = pool(fq, m)
            if e is not None:
                acc[i].append(e * m.sum())
                cols[i].append(rgb[m].astype(np.float32).mean(0) / 255)
    for i in todo:
        if acc[i]:
            e = np.sum(acc[i], 0)
            cache[i] = dict(
                emb=(e / np.linalg.norm(e)).tolist(),
                col=np.mean(cols[i], 0).tolist(),
                n_views=len(acc[i]),
                domain="memory splat renders",
            )
    write_json(cache, out_p)
    log(f"fingerprint refs: {len(todo)} objects ({sum(1 for i in todo if i in cache)} with views)")
    return cache


# ------------------------------------------------------------------------------------------------ robot
class Robot:
    def __init__(self, xy, yaw, M, eye_h):
        self.xy, self.yaw, self.pan, self.pitch = (
            np.array(xy, float),
            float(yaw),
            0.0,
            math.radians(cfg()["sim"]["pitch_walk_deg"]),
        )
        self.M, self.eye_h = M, eye_h
        self.trace = [self.xy.copy()]

    def cam_F(self) -> np.ndarray:
        fz = float(self.M["floor_level"][PL.to_cell(self.xy, self.M["origin"], float(self.M["cell"]), self.M["shape"])])
        eye = np.r_[self.xy, fz + self.eye_h]
        a = self.yaw + self.pan
        d = np.array([math.cos(a) * math.cos(self.pitch), math.sin(a) * math.cos(self.pitch), math.sin(self.pitch)])
        return look_at(eye, eye + d)


def ang_wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------------------------------------ main run
def run(scene: str, world: str, prompt: str, run_name: str, world_splat=None, start: dict | None = None) -> dict:
    """Run the robot in `world` with the memory of `scene` for one prompt; returns the run log."""
    C = cfg()
    Cs = C["sim"]
    from whatchanged.common.gpu import gpu_stage
    from whatchanged.navigation.belief import Belief
    from whatchanged.navigation.memory_map import load_map
    from whatchanged.navigation.task import parse_prompt

    mem = memory_paths(scene)
    M, info = load_map(scene, ensure_current=True)
    T_F_S = M["T_F_S"]
    T_S_F = np.linalg.inv(T_F_S)
    P = Scene(scene).poses()
    W, H = Cs["render_w"], Cs["render_h"]
    K = np.array(P["K"], float)
    K[0] *= W / P["W"]
    K[1] *= H / P["H"]
    rd = nav_dir(scene) / "runs" / run_name
    rd.mkdir(parents=True, exist_ok=True)
    dbg = rd / "qc"
    dbg.mkdir(parents=True, exist_ok=True)
    task = parse_prompt(scene, prompt, rd)
    radius = C["robots"]["default"]["radius_m"]
    # start pose: the doorway frame of the episode's own walk (worlds without a walk: the memory walk's)
    if start is None:
        from whatchanged.navigation.common import doorway_start

        f0, T0 = doorway_start(P["frames"], M, min(C["robots"]["shrink_ladder_m"]))
        start = dict(
            xy=T0[:2, 3].tolist(),
            yaw=float(math.atan2(T0[1, 2], T0[0, 2])),
            source=f"memory video frame {f0['name']} (t {f0['t']} s)",
            real_image=str(frame_file(Scene(scene).root, f0["name"])),
            T_S=f0["T_wc"],
        )
    # eye height: the spec's ~1.5 m, clamped to the memory scan's camera heights (p25-p75) so the simulated camera stays
    # where both splats were filmed from (renders far from the filmed heights show floaters, which the checker would
    # read as changes)
    hz = np.array([(T_F_S @ np.array(f["T_wc"]))[2, 3] for f in P["frames"]])
    eye_h = float(np.clip(C["robot"]["eye_height_m"], np.percentile(hz, 25), np.percentile(hz, 75)))
    robot = Robot(start["xy"], start["yaw"], M, eye_h)
    log(
        f"eye height {eye_h:.2f} m (spec {C['robot']['eye_height_m']} m; memory camera heights p25/p75 "
        f"{np.percentile(hz, 25):.2f}/{np.percentile(hz, 75):.2f} m)"
    )
    B = Belief(scene, M, info, task, radius, K, W, H)
    fps = Cs["fps"]
    dt = 1.0 / fps
    vid = RUNS / "videos" / f"{scene}_{run_name}.mp4"
    vid.parent.mkdir(parents=True, exist_ok=True)
    (RUNS / "qc").mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(vid),
        ],
        stdin=subprocess.PIPE,
    )
    lab = Labeler(16)
    f_log = ImageFont.truetype(FONT_R, 16)
    f_chip = ImageFont.truetype(FONT, 14)
    f_big = ImageFont.truetype(FONT, 20)
    t_wall = time.time()
    stills = []
    with gpu_stage("nav_sim", need_mib=4000) as wd:
        import torch

        sensor = Sensor(scene, world_splat, mem, K, W, H)
        net = ChangeNet(scene_spec(scene))
        B.refs = memory_refs(scene, sensor, net, P, sorted(set(B.movable) | {i for g in B.groups for i in g["ids"]}))
        from whatchanged.navigation.namer import Namer

        B.namer = Namer(scene, info, run_name)
        B.eye_h = eye_h
        B.load_nbv(sensor, P, scene)
        # ---------------- real-vs-render panel at the doorway pose
        intro = None
        if start.get("real_image"):
            real = cv2.imread(start["real_image"])[..., ::-1]
            rgb0, _ = sensor.view(np.array(start["T_S"]))
            intro = (
                cv2.resize(real, (W // 2, H // 2), interpolation=cv2.INTER_AREA),
                cv2.resize(rgb0, (W // 2, H // 2), interpolation=cv2.INTER_AREA),
            )
        # ---------------- behaviour state machine
        plan_actions = scan_actions(robot, Cs)
        phase = "scan"
        path = None
        look_target = None
        look_left = 0.0
        visited = []
        n_search = 0
        verdict = None
        t = 0.0
        frame_i = 0
        rgb, Dep = None, None
        last_state = None
        while t < Cs["max_s"] and verdict is None or (verdict is not None and look_left > -1.0):
            # --- act
            if verdict is not None:
                look_left -= dt  # 1 s hold after the verdict
            elif plan_actions:
                act = plan_actions[0]
                done = step_action(robot, act, dt, Cs, B)
                if done:
                    plan_actions.pop(0)
            elif phase in ("goto", "search_goto", "approach", "confirm_goto") and path is not None:
                arrived = follow(robot, path, dt, Cs, B)
                if arrived:
                    path = None
                    if phase == "approach":
                        plan_actions = [dict(kind="face", xy=look_target[:2], pitch_to=look_target)] + [
                            dict(kind="wait", s=Cs["look_s"])
                        ]
                        phase = "approach_look"
                    elif phase in ("goto", "confirm_goto"):
                        plan_actions = [
                            dict(kind="face", xy=look_target[:2], pitch_to=look_target),
                            dict(kind="wait", s=Cs["look_s"]),
                        ]
                        phase = "look" if phase == "goto" else "confirm_look"
                    else:
                        plan_actions = look_around_actions(robot, Cs)
                        phase = "search_look"
            # --- perceive + decide
            TF = robot.cam_F()
            T_S = T_S_F @ TF
            perceive = frame_i % Cs["perceive_every"] == 0
            if perceive:
                rgb, Dep = sensor.view(T_S)
                Se = sensor.ep_support(Dep, T_S)
                Dep_obs = np.where(Se >= EP_MIN_VIEWS, Dep, 0).astype(
                    Dep.dtype
                )  # sensor dropout where the episode never filmed
                ren, Dp, inst, S, full = sensor.memory(T_S)
                known = (S >= 3) & (Se >= EP_MIN_VIEWS)
                Pn, Pg, fq, fr = net(ren, rgb, Dp, Dep, known)
                B.update(t, TF, rgb, Dep_obs, Dp, inst, S, Pn, Pg, fq, robot, full, ren=ren, fr=fr)
                if (
                    frame_i % (Cs["perceive_every"] * (1 if Cs["max_s"] < 20 else 5)) == 0
                ):  # checker QC panel every 5th perception tick
                    qc_panel(dbg / f"tick_{frame_i:05d}.jpg", ren, rgb, Pn, Pg, known, inst, B)
                if phase == "scan" and not plan_actions:
                    B.describe(t, "start")
                    phase = "decide"
                cur_state = B.target_state()[1].get("state")
                if cur_state != last_state and cur_state in ("confirmed", "missing", "moved"):
                    try:
                        verify_still(
                            sensor,
                            B,
                            robot,
                            K,
                            T_S_F,
                            start,
                            RUNS / "qc" / f"decide_{run_name}_{cur_state}.jpg",
                            f"{run_name} t {t:.1f} s: the {B.main['name']} became '{cur_state}' in this view",
                        )
                    except Exception as e:
                        log(f"decide still failed: {e}")
                changed = cur_state != last_state and phase not in ("scan",)
                last_state = cur_state
                if verdict is None and phase not in ("scan",) and ((not plan_actions and path is None) or changed):
                    plan_actions = []
                    phase, path, look_target, plan_actions, verdict, n_search = decide(
                        B, robot, phase, path, look_target, plan_actions, visited, n_search, t, Cs, radius
                    )
                    if verdict is not None:
                        B.describe(t, "end")
                        B.emit(
                            t, "verdict", **{k: v for k, v in verdict.items() if k != "kind"}, verdict=verdict["kind"]
                        )
                        look_left = 1.0
                elif (
                    verdict is None
                    and path is not None
                    and phase in ("goto", "search_goto", "approach", "confirm_goto")
                ):
                    path = B.replan_if_needed(t, robot, path)
            else:
                rgb, Dep = sensor.view(T_S)
            # --- draw
            frame = draw_frame(rgb, Dep, TF, K, B, robot, path, phase, t, lab, f_log, f_chip, M, T_F_S, verdict)
            if intro is not None and t < 3.0:
                frame = intro_panel(frame, intro, f_big, f_chip, start)
            ff.stdin.write(frame.tobytes())
            if (
                B.events
                and B.events[-1]["t"] == round(t, 2)
                and B.events[-1]["kind"]
                in ("target_confirmed", "target_gone", "target_moved", "new_obstacle", "verdict", "room")
            ):
                p = RUNS / "qc" / f"{scene}_{run_name}_t{t:05.1f}.jpg"
                cv2.imwrite(str(p), frame[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
                stills.append(str(p))
            t = round(t + dt, 4)
            frame_i += 1
        # verification still at the final pose (the decisive views were saved above when the target's state changed)
        try:
            final_T_S = verify_still(
                sensor,
                B,
                robot,
                K,
                T_S_F,
                start,
                RUNS / "qc" / f"verify_{run_name}.jpg",
                f"{run_name}: {prompt} -> {verdict['text'] if verdict else 'no verdict'}",
            )
        except Exception as e:  # never lose a run over the QC still
            log(f"verify still failed: {e}")
            final_T_S = None
        B.namer.close()
        peak = int(torch.cuda.max_memory_reserved() / 2**20)
    ff.stdin.close()
    ff.wait()
    if verdict is None:  # time budget used up: state the belief, honestly
        g_, st_ = B.target_state()
        if st_.get("state") == "missing":
            X = B.checked_frac()
            verdict = dict(
                kind="not_here",
                text=f"The {g_['name']} is not here (checked {100 * X:.0f} % of reachable views, time budget used).",
                conf=round(st_["p_gone"], 3),
                state="missing",
                checked=round(X, 3),
            )
        else:
            verdict = dict(
                kind="unverified",
                text=B.hidden_text(g_) + " (time budget used)",
                conf=round(1 - abs(st_["p_present"] - 0.5) * 2, 3),
                state=st_.get("state"),
            )
        B.emit(t, "verdict", **{k: v for k, v in verdict.items() if k != "kind"}, verdict=verdict["kind"])
    try:
        a_ = B.minimap_img(None, None, scale=6, memory_only=True)
        b_ = B.minimap_img(robot, None, scale=6, trace=np.array(robot.trace), final=False)
        for im_, tt in (
            (a_, "MEMORY map (prior)"),
            (b_, "OBSERVED map after the run (red = new obstacle, green = freed)"),
        ):
            cv2.putText(im_, tt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3)
            cv2.putText(im_, tt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imwrite(
            str(RUNS / "qc" / f"map_{run_name}.jpg"),
            np.concatenate([a_, b_], 1)[..., ::-1],
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
    except Exception as e:
        log(f"map compare failed: {e}")
    summ = RUNS / "qc" / f"{scene}_{run_name}_summary.jpg"
    cv2.imwrite(
        str(summ), summary_still(B, robot, M, verdict, task, scene, run_name)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 93]
    )
    out = dict(
        scene=scene,
        world=world,
        world_splat=str(world_splat),
        prompt=prompt,
        task=task,
        inventory_sha=inventory_sha(scene),
        final_T_S=final_T_S,
        start=start,
        radius_m=B.radius_used,
        eye_h=eye_h,
        verdict=verdict,
        events=B.events,
        trace=np.round(np.array(robot.trace), 3).tolist(),
        sim_s=t,
        frames=frame_i,
        wall_s=round(time.time() - t_wall, 1),
        torch_peak_mib=peak,
        gpu_peak_used_all_mib=wd.peak_used,
        video=str(vid),
        summary=str(summ),
        stills=stills,
        final_belief=B.snapshot(),
    )
    write_json(out, rd / "log.json")
    log(f"run {run_name}: {verdict['text']} ({t:.1f} s sim, {out['wall_s']} s wall) -> {vid}")
    return out


# ------------------------------------------------------------------------------------------------ verification stills
def verify_still(sensor, B, robot, K, T_S_F, start, path, caption):
    """Memory render | episode render (the sensor) at the robot's pose with the target box drawn, and the episode
    walk's nearest real frame."""
    TF = robot.cam_F()
    T_S = T_S_F @ TF
    ver_ep, _ = sensor.view(T_S)
    ver_mem = sensor.memory(T_S)[0]
    Tcw_f = np.linalg.inv(TF)
    tiles = []
    for im, ttl in ((ver_mem, "memory render"), (ver_ep, "episode render (what the robot sees)")):
        im = np.ascontiguousarray(im.copy())
        for g in B.groups:
            c, sz, yw = np.array(g["centre"], float), np.array(g["size"], float), g["yaw"]
            cs_, sn_ = np.cos(yw), np.sin(yw)
            R_ = np.array([[cs_, -sn_, 0], [sn_, cs_, 0], [0, 0, 1.0]])
            Pc = (
                np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * (sz / 2 + 0.03) @ R_.T
                + c
            )
            uv, z_ = proj(Pc, Tcw_f, K)
            if (z_ > 0.2).all():
                x0, y0 = uv.min(0).astype(int)
                x1, y1 = uv.max(0).astype(int)
                cv2.rectangle(im, (x0, y0), (x1, y1), (80, 220, 255), 2)
        tiles.append(label_tile(im, ttl))
    nr = nearest_real(start, T_S)
    if nr is not None:
        im, nm, dpos, dang = nr
        im = cv2.resize(im, (tiles[0].shape[1], tiles[0].shape[0]), interpolation=cv2.INTER_AREA)
        tiles.append(label_tile(im, f"nearest real episode frame {nm} ({dpos:.2f} m, {dang:.0f} deg off)"))
    tiles = [cv2.resize(x, (640, 360), interpolation=cv2.INTER_AREA) for x in tiles]
    vimg = np.concatenate(tiles, 1)
    bar = np.full((44, vimg.shape[1], 3), 245, np.uint8)
    pil = Image.fromarray(bar)
    ImageDraw.Draw(pil).text(
        (12, 22), caption[:220], fill=(10, 10, 10), anchor="lm", font=ImageFont.truetype(FONT_R, 17)
    )
    vimg = np.concatenate([vimg, np.asarray(pil)], 0)
    cv2.imwrite(str(path), vimg[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    return T_S.round(6).tolist()


def label_tile(im, txt):
    pil = Image.fromarray(np.ascontiguousarray(im))
    d = ImageDraw.Draw(pil)
    f = ImageFont.truetype(FONT, 22)
    tw = d.textlength(txt, font=f)
    d.rounded_rectangle((10, 10, 10 + tw + 20, 44), radius=10, fill=(16, 18, 22))
    d.text((20, 27), txt, fill=(255, 255, 255), font=f, anchor="lm")
    return np.asarray(pil)


# ------------------------------------------------------------------------------------------------ actions
def scan_actions(robot, Cs):
    """doorway scan: a slow 360 degree turn on the spot (scan_spin_s seconds), looking slightly down."""
    return [dict(kind="spin", deg=360.0, s=Cs["scan_spin_s"])]


def look_around_actions(robot, Cs):
    """Search look-around: one full turn on the spot at the scan speed, looking a little further down than when walking
    (a +-70 deg head pan left the side behind the robot unseen, so a banana moved onto the sofa behind it was never
    looked at)."""
    return [
        dict(kind="pitch", to=math.radians(-30)),
        dict(kind="spin", deg=360.0, s=Cs["scan_spin_s"]),
        dict(kind="pitch", to=math.radians(Cs["pitch_walk_deg"])),
    ]


def turn_towards(robot, want, dt, Cs):
    """Eased yaw motion: angular speed <= turn_dps, angular acceleration <= turn_acc_dps2, slowing down as the target
    is approached."""
    vmax, amax = math.radians(Cs["turn_dps"]), math.radians(Cs["turn_acc_dps2"])
    err = ang_wrap(want - robot.yaw)
    v_des = float(np.clip(err / 0.7, -vmax, vmax))
    robot.omega = getattr(robot, "omega", 0.0) + float(
        np.clip(v_des - getattr(robot, "omega", 0.0), -amax * dt, amax * dt)
    )
    robot.yaw = ang_wrap(robot.yaw + robot.omega * dt)
    return err


def step_action(robot, act, dt, Cs, B):
    rate = math.radians(Cs["turn_dps"]) * dt
    k = act["kind"]
    if k == "spin":
        left = act.setdefault("left", act["deg"])
        step = min(left, act["deg"] / act["s"] * dt)
        robot.yaw = ang_wrap(robot.yaw + math.radians(step))
        robot.omega = math.radians(step) / dt
        robot.pan += float(np.clip(-robot.pan, -rate, rate))
        act["left"] = left - step
        return act["left"] <= 1e-6
    if k == "pan":
        d = act["to"] - robot.pan
        robot.pan += float(np.clip(d, -rate * 0.6, rate * 0.6))
        return abs(act["to"] - robot.pan) < 1e-3
    if k == "pitch":
        d = act["to"] - robot.pitch
        robot.pitch += float(np.clip(d, -rate, rate))
        return abs(act["to"] - robot.pitch) < 1e-3
    if k == "wait":
        act["s"] -= dt
        return act["s"] <= 0
    if k == "face":
        want = math.atan2(act["xy"][1] - robot.xy[1], act["xy"][0] - robot.xy[0])
        d = turn_towards(robot, want, dt, Cs)
        robot.pan += float(np.clip(-robot.pan, -rate, rate))
        if act.get("pitch_to") is not None:
            tgt = np.asarray(act["pitch_to"], float)
            eye = robot.cam_F()[:3, 3]
            want_p = math.atan2(tgt[2] - eye[2], max(np.linalg.norm(tgt[:2] - eye[:2]), 0.1))
            want_p = float(np.clip(want_p, math.radians(-60), math.radians(10)))
            robot.pitch += float(np.clip(want_p - robot.pitch, -rate, rate))
            return abs(d) < 0.02 and abs(want_p - robot.pitch) < 0.02 and abs(robot.pan) < 0.02
        return abs(d) < 0.02 and abs(robot.pan) < 0.02
    if k == "step":
        rem = act.setdefault("left", act["m"])
        s = min(Cs["speed_mps"] * dt, rem)
        nxt = robot.xy + s * np.array([math.cos(robot.yaw), math.sin(robot.yaw)])
        if not B.standable(nxt):
            return True  # not free ahead: skip the step
        robot.xy = nxt
        robot.trace.append(robot.xy.copy())
        act["left"] = rem - s
        return act["left"] <= 1e-6
    return True


def follow(robot, path, dt, Cs, B):
    """Move along the path (eased turns <= turn_dps; turn in place when the heading error > 20 deg). Returns True when
    the end is reached."""
    P = np.asarray(path, float)
    rate = math.radians(Cs["turn_dps"]) * dt
    robot.pan += float(np.clip(-robot.pan, -rate, rate))
    want_pitch = math.radians(Cs["pitch_walk_deg"])
    robot.pitch += float(np.clip(want_pitch - robot.pitch, -rate, rate))
    if len(P) < 2 or np.linalg.norm(P[-1] - robot.xy) < 0.03:
        return True
    rest = np.asarray(remaining(P, robot.xy), float)
    far = [
        q for q in rest[1:] if np.linalg.norm(q - robot.xy) > 0.3
    ]  # look-ahead 0.3 m along the path: smooth steering
    tgt = far[0] if far else P[-1]
    dpt = rest[1] if len(rest) > 1 else P[-1]
    d = dpt - robot.xy
    if np.linalg.norm(d) <= 0.02 and not far:
        return True
    want = (
        math.atan2(tgt[1] - robot.xy[1], tgt[0] - robot.xy[0]) if np.linalg.norm(tgt - robot.xy) > 0.02 else robot.yaw
    )
    err = turn_towards(robot, want, dt, Cs)
    if abs(err) > math.radians(20):  # face the direction of travel before walking (turn in place)
        return False
    s = min(Cs["speed_mps"] * dt * max(0.3, math.cos(err)), float(np.linalg.norm(P[-1] - robot.xy)))
    dirv = np.array([math.cos(robot.yaw), math.sin(robot.yaw)])
    robot.xy = robot.xy + s * dirv
    robot.trace.append(robot.xy.copy())
    return bool(np.linalg.norm(P[-1] - robot.xy) < 0.03)


# ------------------------------------------------------------------------------------------------ decisions
def decide(B, robot, phase, path, look_target, plan_actions, visited, n_search, t, Cs, radius):
    """returns phase, path, look_target, plan_actions, verdict, n_search (fixed rules, see the module docstring)."""
    tk = B.task
    g, st = B.target_state()
    s = st["state"]
    go = tk["intent"] == "go"

    def V(kind, text, conf, **kw):
        return phase, None, None, [], dict(kind=kind, text=text, conf=round(float(conf), 3), state=s, **kw), n_search

    def nothere():
        X = B.checked_frac()
        return V(
            "not_here",
            f"The {g['name']} is not here (checked {100 * X:.0f} % of reachable views).",
            st["p_gone"],
            checked=round(X, 3),
        )

    moved_txt = f" It moved {100 * st['moved_m']:.0f} cm: now {st['rel_new']}." if s == "moved" else ""
    looked = len(visited) > 0 or phase in ("look", "confirm_look", "approach_look", "search_look", "search_goto")
    if phase == "look" and not visited:
        visited.append(robot.xy.copy())
    if not looked and not go and s in ("confirmed", "missing"):
        # the doorway view is not enough for a verdict: first go to a pose with a clear view of the remembered place and
        # look
        res, tgt = B.plan_to(robot, "look", g)
        if res is not None and res["status"] == "ok":
            if np.linalg.norm(np.asarray(res["path_xy"][-1]) - robot.xy) < 0.05:
                return (
                    "look",
                    None,
                    tgt,
                    [dict(kind="face", xy=tgt[:2], pitch_to=tgt), dict(kind="wait", s=Cs["look_s"])],
                    None,
                    n_search,
                )
            return "goto", res["path_xy"], tgt, [], None, n_search
    # ---- located (at its place, or found elsewhere)
    if s in ("confirmed", "moved") and not (s == "confirmed" and phase == "search_look" and False):
        loc = np.array(st["loc"], float)
        if go:
            if phase == "approach_look":
                conf = st["p_present"] if s == "confirmed" else st["p_moved"]
                if B.target_visible_px < B.Cb["min_vis_px"]:
                    return V(
                        "reached_not_visible",
                        f"Next to the {g['name']} ({B.dist_body_loc(robot, loc):.1f} m) but it is not visible from "
                        f"here ({B.target_visible_px} px).{moved_txt}",
                        conf,
                        moved_m=st.get("moved_m"),
                    )
                return V(
                    "reached" if s == "confirmed" else "reached_moved",
                    f"Ready to pick up the {g['name']}: {B.dist_body_loc(robot, loc):.1f} m away, facing it "
                    f"({B.target_visible_px} px visible).{moved_txt}",
                    conf,
                    moved_m=st.get("moved_m"),
                )
            res, tgt = B.plan_to(robot, "reach", g, loc=loc)
            if res is not None and res["status"] == "ok":
                if np.linalg.norm(np.asarray(res["path_xy"][-1]) - robot.xy) < 0.05:
                    return (
                        "approach_look",
                        None,
                        tgt,
                        [dict(kind="face", xy=tgt[:2], pitch_to=tgt), dict(kind="wait", s=Cs["look_s"])],
                        None,
                        n_search,
                    )
                return "approach", res["path_xy"], tgt, [], None, n_search
            return V("blocked", B.blocked_text(g, res), 1.0)
        if s == "confirmed":
            return V("confirmed", f"The {g['name']} is here ({B.relation_text(g)}).", st["p_present"])
        if phase == "confirm_look" or st["n_elsewhere"] >= 3:
            return V("moved", f"The {g['name']} is not at its place.{moved_txt}", st["p_moved"], moved_m=st["moved_m"])
        res, tgt = B.plan_to(robot, "look_at_loc", g, loc=loc)
        if res is not None and res["status"] == "ok":
            if np.linalg.norm(np.asarray(res["path_xy"][-1]) - robot.xy) < 0.05:
                return (
                    "confirm_look",
                    None,
                    tgt,
                    [dict(kind="face", xy=tgt[:2], pitch_to=tgt), dict(kind="wait", s=Cs["look_s"])],
                    None,
                    n_search,
                )
            return "confirm_goto", res["path_xy"], tgt, [], None, n_search
        return V(
            "moved",
            f"The {g['name']} is not at its place.{moved_txt} (could not get closer to confirm)",
            st["p_moved"],
            moved_m=st["moved_m"],
        )
    # ---- place seen empty -> search
    if s == "missing":
        if phase not in ("search_look", "search_goto"):
            B.emit(t, "search_start", name=g["name"])
        B.reachable_cands(robot)
        if n_search >= Cs["max_search_views"] or B.search_gain_frac() < Cs["search_min_gain"]:
            return nothere()
        res, tgt = B.plan_to(robot, "search")
        if res is None or res["status"] != "ok":
            return nothere()
        return "search_goto", res["path_xy"], tgt, [], None, n_search + 1
    # ---- not yet seen / unverified / seen elsewhere (remembered place not checked): go and look at that place
    if phase == "look" and (not visited or np.linalg.norm(visited[-1] - robot.xy) > 0.05):
        visited.append(robot.xy.copy())
    if len(visited) >= 3:
        return V("unverified", B.hidden_text(g), 1 - abs(st["p_present"] - 0.5) * 2)
    res, tgt = B.plan_to(robot, "look", g, exclude=visited)
    if (
        res is None or res["status"] != "ok"
    ):  # no reachable look pose: walk towards the place itself (it may come into view)
        res2, tgt2 = B.plan_to(robot, "reach", g)
        if res2 is not None and res2["status"] == "ok" and PL.length_m(res2["path_xy"]) > 0.05:
            return "goto", res2["path_xy"], tgt2, [], None, n_search
        if visited:  # it did get there and look: the place is hidden, not unreachable
            return V("unverified", B.hidden_text(g), 1 - abs(st["p_present"] - 0.5) * 2)
        return V("blocked", B.blocked_text(g, res2 if res2 is not None else res), 1.0)
    if np.linalg.norm(np.asarray(res["path_xy"][-1]) - robot.xy) < 0.05:
        return (
            "look",
            None,
            tgt,
            [dict(kind="face", xy=tgt[:2], pitch_to=tgt), dict(kind="wait", s=Cs["look_s"])],
            None,
            n_search,
        )
    return "goto", res["path_xy"], tgt, [], None, n_search


# ------------------------------------------------------------------------------------------------ drawing
def draw_frame(rgb, Dep, TF, K, B, robot, path, phase, t, lab, f_log, f_chip, M, T_F_S, verdict):
    H, W = rgb.shape[:2]
    img = rgb.copy()
    o, cell = M["origin"], float(M["cell"])
    fl = M["floor_level"]

    def floor_z(Pp):
        return np.array([fl[PL.to_cell(p, o, cell, M["shape"])] for p in np.atleast_2d(Pp)])

    Dh = cv2.resize(Dep, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
    if path is not None and len(path) > 1:
        rest = remaining(path, robot.xy)
        A, Ch = ribbon(img.shape, rest, floor_z, TF, K, Dh, t)
        glow = cv2.GaussianBlur(A, (0, 0), 6) * 0.9
        col = np.array(CYAN, np.float32) / 255
        im = img.astype(np.float32) / 255
        im = im * (1 - 0.55 * glow[..., None]) + col * 0.55 * glow[..., None]
        im = im * (1 - 0.75 * A[..., None]) + col * 0.75 * A[..., None]
        im = im * (1 - 0.8 * Ch[..., None]) + 0.8 * Ch[..., None]
        img = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    # newly observed obstacles: translucent red floor footprints (depth-tested) + outline
    nb = B.new_blobs()
    if nb:
        Tcw_ = np.linalg.inv(TF)
        patch = np.zeros((H, W), np.uint8)
        for b in nb:
            for ci, cj in b["ij"]:
                xy = o + (np.array([ci, cj]) + np.array([[0, 0], [1, 0], [1, 1], [0, 1]])) * cell
                P4 = np.c_[xy, np.full(4, fl[ci, cj] + 0.015)]
                uv, zz = proj(P4, Tcw_, K)
                if (zz < 0.2).any():
                    continue
                zc = float(zz.mean())
                uc, vc = (
                    int(np.clip(uv[:, 0].mean() / 2, 0, W // 2 - 1)),
                    int(np.clip(uv[:, 1].mean() / 2, 0, H // 2 - 1)),
                )
                dd = Dh[vc, uc]
                if dd > 0 and zc > dd + 0.06 + 0.04 * dd:
                    continue
                cv2.fillConvexPoly(patch, np.round(uv).astype(np.int32), 255)
        if patch.any():
            pm = cv2.GaussianBlur(patch.astype(np.float32) / 255, (0, 0), 1.0)
            im = img.astype(np.float32)
            im = im * (1 - 0.5 * pm[..., None]) + np.array([232, 48, 40], np.float32) * 0.5 * pm[..., None]
            cs, _ = cv2.findContours((patch > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img = np.clip(im, 0, 255).astype(np.uint8)
            cv2.drawContours(img, cs, -1, (255, 255, 255), 2, cv2.LINE_AA)
    base = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    Tcw = np.linalg.inv(TF)
    # marks: green seen as expected, red seen changed / missing, grey unseen (movable inventory objects in view,
    # depth-tested)
    labels = []
    for oid, ob in B.objs.items():
        if ob["cls"] == "static" or ob["n_gauss"] < 1500:
            continue
        c = np.array(ob["box"]["centre"], float)
        top = c + np.array([0, 0, ob["box"]["size"][2] / 2 + 0.02])
        uv, z = proj(top[None], Tcw, K)
        if not (0.3 < z[0] < 5.0 and 0 <= uv[0, 0] < W and 0 <= uv[0, 1] < H):
            continue
        st = B.obj_state(oid)
        u, v = int(uv[0, 0]), int(uv[0, 1])
        dd = Dep[min(v, H - 1), min(u, W - 1)]
        if st != "gone" and dd > 0 and z[0] > dd + 0.08 + 0.05 * dd:
            continue
        colr = {"present": TYPE_COL["added"], "gone": TYPE_COL["removed"]}.get(st, TYPE_COL["unseen"])
        rr = 5 if st != "not observed" else 4
        d.ellipse(
            (u - rr, v - rr, u + rr, v + rr),
            fill=colr + (230,) if st != "not observed" else None,
            outline=colr + (230,),
            width=2,
        )
    # target
    g, st = B.target_state()
    if g is not None:
        old = np.array(g["centre"], float)
        uv, z = proj(old[None] + np.array([[0, 0, g["size"][2] / 2 + 0.12]]), Tcw, K)
        if 0.3 < z[0] < 6 and -50 < uv[0, 0] < W + 50 and -50 < uv[0, 1] < H + 50:
            s = st["state"]
            colr = (
                TYPE_COL["added"]
                if s == "confirmed"
                else (TYPE_COL["removed"] if s in ("missing", "moved") else TYPE_COL["unseen"])
            )
            txt = {
                "confirmed": "here",
                "missing": "not here",
                "moved": "moved",
                "seen elsewhere": "not checked yet",
                "not yet seen": "not seen yet",
                "unverified": "checking",
            }.get(s, s)
            labels.append(((float(uv[0, 0]), float(uv[0, 1])), f"{g['name']} · {txt}", colr, 3))
        if st["state"] in ("moved", "seen elsewhere"):
            loc = np.array(st["loc"], float)
            uv, z = proj(loc[None] + np.array([[0, 0, 0.1]]), Tcw, K)
            if 0.3 < z[0] < 6 and 0 <= uv[0, 0] < W and 0 <= uv[0, 1] < H:
                labels.append(
                    (
                        (float(uv[0, 0]), float(uv[0, 1])),
                        f"{g['name']} · found, {100 * st['moved_m']:.0f} cm away",
                        TYPE_COL["moved"],
                        4,
                    )
                )
    for b in B.reported_blobs():
        uv, z = proj(np.array(b["centre"])[None] + np.array([[0, 0, b["size"][2] / 2]]), Tcw, K)
        if 0.3 < z[0] < 6 and 0 <= uv[0, 0] < W and 0 <= uv[0, 1] < H:
            labels.append(
                (
                    (float(uv[0, 0]), float(uv[0, 1])),
                    f"new {b.get('name', 'obstacle')}" if b.get("name", "object") != "object" else "new obstacle",
                    TYPE_COL["removed"],
                    2,
                )
            )
    taken = []
    for an, txt, colr, _ in sorted(labels, key=lambda x: -x[3])[:3]:
        if not (-60 <= an[0] < W + 60 and -60 <= an[1] < H + 60):
            continue
        an = (float(np.clip(an[0], 8, W - 8)), float(np.clip(an[1], 8, H - 8)))
        b = lab.place(an, txt, taken, W, H, avoid=[(W - 410, 0, W, 380), (0, H - 95, 640, H), (0, 0, 420, 44)])
        taken.append(b)
        lab.draw(d, an, b, txt, colr)
    # mini-map
    mm = B.minimap_img(robot, path)
    mh, mw = mm.shape[:2]
    sc = min(380 / mw, 330 / mh)  # large mini-map (>= 60 % of the frame height)
    mm = cv2.resize(mm, (int(mw * sc), int(mh * sc)), interpolation=cv2.INTER_AREA)
    mh, mw = mm.shape[:2]
    x0, y0 = W - mw - 12, 12
    d.rounded_rectangle((x0 - 7, y0 - 7, x0 + mw + 7, y0 + mh + 24), radius=11, fill=(14, 16, 20, 150))
    lx = x0
    for colr, tx in (
        ((230, 60, 50), "new"),
        ((80, 200, 110), "freed"),
        ((255, 225, 70), "path"),
        ((214, 228, 246), "seen"),
    ):
        d.ellipse((lx, y0 + mh + 5, lx + 9, y0 + mh + 14), fill=colr + (255,))
        d.text((lx + 12, y0 + mh + 9), tx, fill=(230, 230, 230, 230), font=f_chip, anchor="lm")
        lx += 62
    # chip + log
    chip = f"t {t:4.1f} s · {phase.replace('_', ' ')}"
    if g is not None:
        chip += f" · {g['name']}: {st['state']}"
    tw = d.textlength(chip, font=f_chip)
    d.rounded_rectangle((12, 12, 12 + tw + 22, 38), radius=12, fill=(16, 18, 22, 160))
    d.text((23, 25), chip, fill=(240, 240, 240, 235), font=f_chip, anchor="lm")
    recent = [e for e in B.events if e["t"] <= t + 1e-6 and t - e["t"] <= 6.0 and e["kind"] != "task"][-2:]
    if verdict is not None:
        recent = [e for e in B.events if e["kind"] == "verdict"][-1:]
    rows = []
    for e in recent[::-1]:
        age = t - e["t"]
        al = float(np.clip(1.0 - (age - 4.0) / 2.0, 0, 1)) if verdict is None else 1.0
        rows += [(ln, al) for ln in wrap(e["text"], 95)[::-1]]
    for n, (txt, al) in enumerate(rows[:2]):
        y = H - 26 - 28 * n
        tw = d.textlength(txt, font=f_log)
        d.rounded_rectangle((12, y - 12, 12 + tw + 22, y + 12), radius=11, fill=(16, 18, 22, int(165 * al)))
        d.text((23, y), txt, fill=(255, 255, 255, int(255 * al)), font=f_log, anchor="lm")
    comp = Image.alpha_composite(base, ov)
    mmi = Image.fromarray(mm).convert("RGBA")
    mmi.putalpha(215)
    comp.alpha_composite(mmi, (x0, y0))
    return np.asarray(comp.convert("RGB"))


def intro_panel(frame, intro, f_big, f_chip, start):
    H, W = frame.shape[:2]
    canvas = np.full_like(frame, 18)
    real, ren = intro
    y = (H - real.shape[0]) // 2
    canvas[y : y + real.shape[0], : real.shape[1]] = real
    canvas[y : y + ren.shape[0], W - ren.shape[1] :] = ren
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    d.text((W // 4, y - 24), "real frame (episode walk)", fill=(240, 240, 240), font=f_big, anchor="mm")
    d.text((3 * W // 4, y - 24), "simulated head camera (episode splat)", fill=(240, 240, 240), font=f_big, anchor="mm")
    d.text(
        (W // 2, y + real.shape[0] + 22),
        f"doorway pose: {start['source']}",
        fill=(200, 200, 200),
        font=f_chip,
        anchor="mm",
    )
    return np.asarray(im)


def summary_still(B, robot, M, verdict, task, scene, run_name):
    img = B.minimap_img(robot, None, scale=9, trace=np.array(robot.trace), final=True)
    h, w = img.shape[:2]
    pad = np.full((h, 420, 3), 250, np.uint8)
    im = Image.fromarray(np.concatenate([img, pad], 1))
    d = ImageDraw.Draw(im)
    fb = ImageFont.truetype(FONT, 17)
    fr = ImageFont.truetype(FONT_R, 14)
    x = w + 16
    d.text((x, 18), f"{scene} · {run_name}", fill=(0, 0, 0), font=fb)
    d.text((x, 44), f'prompt: "{task["prompt"]}"', fill=(30, 30, 30), font=fr)
    yy = 72
    for line in wrap(f"Verdict: {verdict['text']}{B.evidence_text(verdict.get('kind'))}", 38):
        d.text((x, yy), line, fill=(0, 0, 0), font=fb)
        yy += 22
    yy += 8
    for e in [e for e in B.events if e["kind"] == "room"][-1:]:
        for line in wrap(e["text"], 50):
            d.text((x, yy), line, fill=(40, 40, 40), font=fr)
            yy += 18
    yy += 10
    leg = [
        ((52, 199, 89), "seen, as expected"),
        ((235, 64, 52), "seen, changed / missing / new obstacle"),
        ((150, 150, 150), "not seen"),
        ((80, 200, 110), "freed space"),
        ((255, 225, 70), "robot path"),
        ((214, 228, 246), "floor the robot saw"),
        ((150, 150, 150), "floor not seen"),
        ((72, 74, 78), "occupied (memory)"),
    ]
    for c, tx in leg:
        d.rectangle((x, yy + 2, x + 12, yy + 14), fill=c)
        d.text((x + 18, yy), tx, fill=(20, 20, 20), font=fr)
        yy += 20
    return np.asarray(im)


def qc_panel(path, ren, rgb, Pn, Pg, known, inst, B):
    """memory render | episode render (sensor) | P(new) | P(gone) with the target's memory mask outlined."""

    def heat(p):
        h = cv2.applyColorMap((np.clip(p, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
        h[~known] = (h[~known] * 0.35).astype(np.uint8)
        return h

    tiles = [ren, rgb, heat(Pn), heat(Pg)]
    if B.main is not None:
        m = np.isin(inst, B.main["gids"]).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tiles = [cv2.drawContours(np.ascontiguousarray(x.copy()), cs, -1, (80, 220, 255), 2) for x in tiles]
    names = ["memory render", "sensor (episode render)", "P(new)", "P(gone)"]
    tiles = [
        cv2.putText(cv2.resize(x, (480, 270)), n, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for x, n in zip(tiles, names)
    ]
    im = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
    cv2.imwrite(str(path), im[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])


def wrap(s, n):
    out, cur = [], ""
    for wd in s.split():
        if len(cur) + len(wd) + 1 > n:
            out.append(cur)
            cur = wd
        else:
            cur = (cur + " " + wd).strip()
    return out + [cur]
