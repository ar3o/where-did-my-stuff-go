"""Calibrated change confidences from held-out synthetic edits (the validation pairs, never the query photos).

    python -m whatchanged.checker.calibrate --scene config1

Per validation pair the checker gives P(appeared) and P(missing). Regions:
  positives  missing  = "missing" regions of add / move / rotate edits
             appeared = "appeared" regions of remove / move / rotate edits
             moved    = move edits with both regions (mean of the two)
  negatives  inventory objects visible on >= 300 known pixels that were not edited (missing: their mean P(missing),
             appeared: their mean P(appeared)); moved: two such objects (P(missing) of one, P(appeared) of the other)
x = [mean class probability over the region's known pixels, log(object size in m)] -> Platt scaling per type. Two-fold
cross-fitting by frame gives out-of-fold probabilities for the reliability curve and ECE (10 bins); the final model is
fitted on all regions. At test time several agreeing photos are combined by noisy-OR, an extrapolation because every
synthetic pair has one view.
Outputs <runs>/<scene>/checker/conf_calibration.{json,png}; checker/pairprobs/<pair>.npy caches the probabilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from whatchanged.checker import model as checker_model
from whatchanged.checker.settings import load_settings
from whatchanged.common.gpu import gpu_stage
from whatchanged.common.io import log, read_json, write_json
from whatchanged.common.scene import Scene
from whatchanged.splat import gaussians as gs

MIN_REGION_PX = 100
MIN_NEGATIVE_PX = 300
MAX_NEGATIVE_OVERLAP = 0.1


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_platt(X, y, l2: float = 1e-3, iters: int = 3000, lr: float = 0.5) -> np.ndarray:
    Xb = np.c_[X, np.ones(len(X))]
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = sigmoid(Xb @ w)
        w -= lr * (Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0])
    return w


def predict_platt(w, X):
    return sigmoid(np.c_[X, np.ones(len(X))] @ w)


def ece(p, y, bins: int = 10) -> tuple[float, list]:
    e, curve = 0.0, []
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if m.sum() == 0:
            continue
        e += m.mean() * abs(p[m].mean() - y[m].mean())
        curve.append(
            dict(
                bin=[round(edges[i], 2), round(edges[i + 1], 2)],
                mean_p=round(float(p[m].mean()), 3),
                frac_pos=round(float(y[m].mean()), 3),
                n=int(m.sum()),
            )
        )
    return float(e), curve


def cross_fit(X, y, frames, fit, predict):
    """Two folds by frame -> out-of-fold predictions; final model on all data."""
    uf = np.unique(frames)
    fold = np.isin(frames, uf[::2])
    oof = np.zeros(len(y))
    for f in (True, False):
        oof[fold == f] = predict(fit(X[fold != f], y[fold != f]), X[fold == f])
    return oof, fit(X, y)


def pair_probabilities(scene: Scene, net, d: Path) -> np.ndarray:
    """(3, H/4, W/4) probabilities of a validation pair, cached."""
    cache = scene.checker / "pairprobs" / f"{d.name}.npy"
    if cache.exists():
        return np.load(cache)
    ren = cv2.imread(str(d / "render.jpg"))[..., ::-1]
    real = cv2.imread(str(d / "real.jpg"))[..., ::-1]
    low = checker_model.quarter_inputs(
        ren, real, np.load(d / "depth_r.npy"), np.load(d / "depth_q.npy"), np.load(d / "known.npy")
    )
    p = checker_model.probabilities(net, np.load(d / "f_render.npy"), np.load(d / "f_real.npy"), low)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, p.astype(np.float32))
    return p


def upsample(p: np.ndarray, W: int, H: int) -> np.ndarray:
    return cv2.resize(p.astype(np.float32), (W, H))


def regions(scene: Scene, net, tsp, objects, gid) -> dict:
    ids = [0] + [int(o["id"][1:]) for o in objects]
    size_of = {int(o["id"][1:]): max(o["box"]["size"]) for o in objects if o["cls"] != "static"}
    rows = {"missing": [], "appeared": [], "moved": []}
    pairs = sorted(d for d in (scene.checker / "pairs" / "val").glob("*") if (d / "f_real.npy").exists())
    for d in pairs:
        meta = read_json(d / "meta.json")
        K, T = np.array(meta["K"]), np.array(meta["T_wc"])
        known = np.load(d / "known.npy")
        H, W = known.shape
        p = pair_probabilities(scene, net, d)
        P_app, P_mis = upsample(p[1], W, H), upsample(p[2], W, H)
        em = np.load(d / "edit_masks.npz")
        frame = meta["frame"]
        for k, e in enumerate(meta["edits"]):
            if e["type"] == "decoy_fill":
                continue
            s = float(e.get("size", 0.2))
            mk, ak = f"e{k}_missing", f"e{k}_appeared"
            mis = em[mk] & known if mk in em.files else None
            app = em[ak] & known if ak in em.files else None
            if mis is not None and mis.sum() >= MIN_REGION_PX:
                rows["missing"].append(dict(x=[float(P_mis[mis].mean()), np.log(s)], y=1, frame=frame))
            if app is not None and app.sum() >= MIN_REGION_PX:
                rows["appeared"].append(dict(x=[float(P_app[app].mean()), np.log(s)], y=1, frame=frame))
            if (
                e["type"] == "move"
                and mis is not None
                and app is not None
                and mis.sum() >= MIN_REGION_PX
                and app.sum() >= MIN_REGION_PX
            ):
                rows["moved"].append(
                    dict(x=[0.5 * (float(P_mis[mis].mean()) + float(P_app[app].mean())), np.log(s)], y=1, frame=frame)
                )
        inst = gs.render_ids(tsp, gid, K, T, W, H, ids)
        any_edit = np.zeros((H, W), bool)
        for f in em.files:
            any_edit |= em[f]
        negatives = []
        for k in np.unique(inst):
            if k == 0 or int(k) not in size_of:
                continue
            R = (inst == k) & known
            if R.sum() < MIN_NEGATIVE_PX or (R & any_edit).sum() > MAX_NEGATIVE_OVERLAP * R.sum():
                continue
            negatives.append((float(P_mis[R].mean()), float(P_app[R].mean()), np.log(size_of[int(k)])))
        for pm, pa, ls in negatives:
            rows["missing"].append(dict(x=[pm, ls], y=0, frame=frame))
            rows["appeared"].append(dict(x=[pa, ls], y=0, frame=frame))
        rng = np.random.default_rng(len(rows["moved"]))
        for _ in range(min(3, len(negatives) // 2)):
            a, b = rng.choice(len(negatives), 2, replace=False)
            rows["moved"].append(
                dict(
                    x=[0.5 * (negatives[a][0] + negatives[b][1]), 0.5 * (negatives[a][2] + negatives[b][2])],
                    y=0,
                    frame=frame,
                )
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()
    cfg = load_settings(args.config).calibrate
    scene = Scene(args.scene)
    inv = read_json(scene.inventory / "inventory.json")["objects"]
    gid = np.load(scene.inventory / "inventory.npz")["gid"]
    net = checker_model.load(scene.runs / "checker" / "best.pt", "cpu")
    with gpu_stage("checker_calibrate", need_mib=cfg.need_mib):
        rows = regions(scene, net, gs.to_torch(gs.load(scene.splat)), inv, gid)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    out = {}
    for ax, (typ, R) in zip(axs, rows.items()):
        X = np.array([r["x"] for r in R], float)
        y = np.array([r["y"] for r in R], float)
        oof, w = cross_fit(X, y, np.array([r["frame"] for r in R]), fit_platt, predict_platt)
        e, curve = ece(oof, y)
        out[typ] = dict(
            weights=w.tolist(),
            features=["mean class probability", "log size m", "bias"],
            ece_out_of_fold=round(e, 4),
            n_pos=int(y.sum()),
            n_neg=int((1 - y).sum()),
            curve=curve,
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot([c["mean_p"] for c in curve], [c["frac_pos"] for c in curve], "o-")
        ax.set_title(f"{typ}: ECE {e:.3f} (n+ {int(y.sum())}, n- {int((1 - y).sum())})")
        ax.set_xlabel("predicted change probability (out of fold)")
        ax.set_ylabel("observed fraction of true edits")
    fig.tight_layout()
    dst = scene.runs / "checker"
    fig.savefig(dst / "conf_calibration.png", dpi=110)
    write_json(
        dict(
            types=out,
            data="validation pairs (synthetic edits of held-out walk frames)",
            combine="noisy-OR over agreeing photos",
        ),
        dst / "conf_calibration.json",
    )
    log({t: (v["ece_out_of_fold"], v["n_pos"], v["n_neg"]) for t, v in out.items()})


if __name__ == "__main__":
    main()
