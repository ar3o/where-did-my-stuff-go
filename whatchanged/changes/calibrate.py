"""Re-identification calibration on the held-out walk frames and their synthetic-edit pairs, never on the query photos.

* p_same(detection, object): positives are detections lying on the object (Ochiai >= 0.5 with its rendered mask),
  negatives are detections with the same phrase (the object's name or runner-up) lying on other things.
* r_ref: 95th percentile of |lifted detection centre - reference box centre| / max(size, 0.15 m) over the positives:
  the "still at its reference place" radius, relative to object size because the lifting error grows with size.
* p_added(detection): over real-frame detections that no inventory object still present in the edited render claims
  (Ochiai >= 0.5) and that no same-phrase detection of the edited render explains (IoU >= 0.3); positives lie on an
  "appeared" edit region.
Two folds by frame give out-of-fold ECE and AUC; the final fits use everything.
Output: <runs>/<scene>/changes/reid_calibration.json.
"""

from __future__ import annotations

import numpy as np

from whatchanged.changes.reid import Appearance, auc, fit_logit, pair_features, phrases_of, predict_logit
from whatchanged.checker.calibrate import cross_fit, ece, pair_probabilities, upsample
from whatchanged.common.images import ochiai
from whatchanged.common.io import log, read_json, unpack_masks, write_json
from whatchanged.common.scene import Scene

R_MIN_SIZE_M = 0.15  # below this size the lifting noise stops shrinking
ON_OBJECT = 0.5
EXPLAINED_IOU = 0.3


def on_object(m, inst, cnt, byk) -> int | None:
    labs, inter = np.unique(inst[m], return_counts=True)
    on = None
    for lab, n_in in zip(labs, inter):
        if lab in byk and ochiai(n_in, m.sum(), cnt[lab]) >= ON_OBJECT:
            on = int(lab)
    return on


def rows(scene: Scene, net, objs, app: Appearance):
    byk = {int(o["id"][1:]): o for o in objs}
    X, y, fq, dist = [], [], [], []
    Xa, ya, fa = [], [], []
    for jf in sorted((scene.changes / "dets").glob("val_*.json")):
        n = jf.stem[4:]
        J = read_json(jf)
        z = np.load(jf.with_suffix(".npz"))
        M = unpack_masks(z["masks"], z["shape"])
        inst = z["inst"].astype(np.int64)
        cnt = np.bincount(inst.ravel(), minlength=max(byk) + 1)
        for row, m in zip(J["rows"], M):
            on = on_object(m, inst, cnt, byk)
            if row.get("box") is None or row.get("emb") is None:
                continue
            for o in objs:
                if row["phrase"] not in phrases_of(o) or o["id"] not in app.fp:
                    continue
                k = int(o["id"][1:])
                X.append(pair_features(row, o, app))
                y.append(int(on == k))
                fq.append(n)
                if on == k:
                    d = np.linalg.norm(np.subtract(row["box"]["centre"], o["box"]["centre"]))
                    dist.append(float(d) / max(max(o["box"]["size"]), R_MIN_SIZE_M))
        for pd in sorted((scene.checker / "pairs" / "val").glob(f"{n}_*")):
            if f"r_{pd.name}" not in z.files:
                continue
            em = np.load(pd / "edit_masks.npz")
            appeared = np.zeros(inst.shape, bool)
            for f in em.files:
                if f.endswith("_appeared"):
                    appeared |= em[f]
            Mr = unpack_masks(z[f"r_{pd.name}"], z[f"rs_{pd.name}"])
            rph = [r["phrase"] for r in J["renders"][pd.name]]
            P_app = upsample(pair_probabilities(scene, net, pd)[1], inst.shape[1], inst.shape[0])
            for row, m in zip(J["rows"], M):
                if row.get("box") is None:
                    continue
                labs, inter = np.unique(inst[m], return_counts=True)
                claimed = False
                for lab, n_in in zip(labs, inter):
                    if lab in byk and ochiai(n_in, m.sum(), cnt[lab]) >= ON_OBJECT:
                        R = inst == lab
                        if (R & appeared).sum() < 0.5 * R.sum():
                            claimed = True
                if claimed:
                    continue
                if any(
                    q == row["phrase"] and (m & mm).sum() / max((m | mm).sum(), 1) >= EXPLAINED_IOU
                    for q, mm in zip(rph, Mr)
                ):
                    continue
                label = (
                    int(ochiai((m & appeared).sum(), m.sum(), max(appeared.sum(), 1)) >= ON_OBJECT)
                    if appeared.any()
                    else 0
                )
                Xa.append([float(P_app[m].mean()), row["z"], np.log(max(max(row["box"]["size"]), 0.02))])
                ya.append(label)
                fa.append(pd.name[:4])
    return (np.array(X), np.array(y, float), np.array(fq), dist), (np.array(Xa), np.array(ya, float), np.array(fa))


def calibrate(scene: Scene, net) -> dict:
    objs = [o for o in read_json(scene.inventory / "inventory.json")["objects"] if o["cls"] != "static"]
    (X, y, fq, dist), (Xa, ya, fa) = rows(scene, net, objs, Appearance(scene))
    out = {}
    names = (
        (
            "p_same",
            X,
            y,
            fq,
            ["siglip2 max cos to reference crops", "dinov2 cos", "size term", "colour term", "sam3 z"],
        ),
        ("p_added", Xa, ya, fa, ["mean P(appeared)", "sam3 z", "log size"]),
    )
    for name, XX, yy, ff, feats in names:
        oof, model = cross_fit(XX, yy, ff, fit_logit, predict_logit)
        e, curve = ece(oof, yy)
        prec = float(yy[oof >= 0.5].mean()) if (oof >= 0.5).any() else 1.0
        rec = float((oof[yy > 0] >= 0.5).mean()) if yy.sum() else 0.0
        out[name] = dict(
            model=model,
            features=feats,
            ece_out_of_fold=round(e, 4),
            auc_out_of_fold=round(auc(oof, yy), 4),
            n_pos=int(yy.sum()),
            n_neg=int((1 - yy).sum()),
            precision_at_0_5=round(prec, 3),
            recall_at_0_5=round(rec, 3),
            curve=curve,
        )
    out["r_ref_rel"] = round(float(np.percentile(dist, 95)) if dist else 1.0, 3)
    out["r_ref_min_size_m"] = R_MIN_SIZE_M
    out["data"] = "validation part of the walk and its synthetic-edit pairs"
    dst = scene.runs / "changes"
    dst.mkdir(parents=True, exist_ok=True)
    write_json(out, dst / "reid_calibration.json")
    log(
        {
            k: (v["ece_out_of_fold"], v["auc_out_of_fold"], v["n_pos"], v["n_neg"])
            for k, v in out.items()
            if isinstance(v, dict)
        }
    )
    log(f"same-place radius {out['r_ref_rel']} x max(size, {R_MIN_SIZE_M} m)")
    return out
