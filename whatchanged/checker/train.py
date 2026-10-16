"""Train and validate the change checker on the render-vs-real pairs.

    python -m whatchanged.checker.train --scene config1

All pairs are held in memory (fp16 features, 1/4-res images, depths, labels). Augmentation on the cached inputs: feature
channel dropout (10 %), Gaussian feature noise and brightness jitter of the real image. Labels are taken down to 1/4 res
by a 4 x 4 block vote in which a change class wins with 25 % of the block, which keeps thin slivers of small moves.
Loss: class-weighted cross-entropy (weights sqrt(f0 / fc), capped at 20) + 0.5 x focal loss (gamma 2). AdamW with a
one-cycle schedule. Validation on the held-out last 20 % of the walk: pixel precision / recall / F1 of both change
classes, object recall per edit (an edit counts as found when 25 % of its region gets its class), and the
false-positive area and false components (>= 60 px at 1/4 res) on unedited pairs. Checkpoints every epoch; a rerun
resumes. Outputs <runs>/<scene>/checker/: best.pt, ckpt.pt, train_log.json, val_report.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from whatchanged.checker.model import APPEARED, MISSING, Checker, quarter_inputs
from whatchanged.checker.settings import load_settings
from whatchanged.common.gpu import gpu_stage
from whatchanged.common.io import log, read_json, write_json
from whatchanged.common.scene import Scene

IGNORE = 255
CLASS_WIN_FRAC = 0.25
MAX_CLASS_WEIGHT = 20.0


def quarter_labels(lab: np.ndarray, H4: int, W4: int) -> np.ndarray:
    L = lab[: H4 * 4, : W4 * 4].reshape(H4, 4, W4, 4)

    def frac(c):
        return (L == c).mean((1, 3))

    out = np.zeros((H4, W4), np.uint8)
    out[frac(IGNORE) >= 0.5] = IGNORE
    out[(frac(APPEARED) >= CLASS_WIN_FRAC) & (frac(APPEARED) >= frac(MISSING))] = APPEARED
    out[(frac(MISSING) >= CLASS_WIN_FRAC) & (frac(MISSING) > frac(APPEARED))] = MISSING
    return out


def load_pair(d: Path) -> dict:
    ren = cv2.imread(str(d / "render.jpg"))[..., ::-1]
    real = cv2.imread(str(d / "real.jpg"))[..., ::-1]
    low = quarter_inputs(ren, real, np.load(d / "depth_r.npy"), np.load(d / "depth_q.npy"), np.load(d / "known.npy"))
    lab = cv2.imread(str(d / "label.png"), cv2.IMREAD_UNCHANGED)
    return dict(
        fr=np.load(d / "f_render.npy"),
        fq=np.load(d / "f_real.npy"),
        low=low,
        lab=quarter_labels(lab, ren.shape[0] // 4, ren.shape[1] // 4),
        meta=read_json(d / "meta.json"),
        dir=d,
    )


def focal_ce(logits, y, w):
    m = y != IGNORE
    if m.sum() == 0:
        return logits.sum() * 0
    ce = F.cross_entropy(logits, y, weight=w, ignore_index=IGNORE, reduction="none")
    plain = F.cross_entropy(logits, y, ignore_index=IGNORE, reduction="none")
    focal = (1 - torch.exp(-plain)) ** 2 * plain
    return ce[m].mean() + 0.5 * focal[m].mean()


def batchify(items, dev, train: bool):
    fr = torch.as_tensor(np.stack([x["fr"] for x in items])).float().to(dev)
    fq = torch.as_tensor(np.stack([x["fq"] for x in items])).float().to(dev)
    low = torch.as_tensor(np.stack([x["low"] for x in items])).to(dev)
    y = torch.as_tensor(np.stack([x["lab"] for x in items])).long().to(dev)
    if train:
        fr = fr + torch.randn_like(fr) * 0.02
        fq = fq + torch.randn_like(fq) * 0.02
        keep = (torch.rand(fr.shape[0], fr.shape[1], 1, 1, device=dev) > 0.1).float()
        fr, fq = fr * keep, fq * keep
        g = torch.empty(low.shape[0], 1, 1, 1, device=dev).uniform_(0.85, 1.15)
        low = torch.cat([low[:, :3], (low[:, 3:6] * g).clamp(0, 1), low[:, 6:]], 1)
    return fr, fq, low, y


def edit_buckets(o: dict) -> list[str]:
    keys = [o["type"]]
    if o["type"] == "move":
        keys.append("move_small_2-10cm" if o["small"] else "move_large")
        if o["dist"] is not None and o["dist"] <= 0.05:
            keys.append("move_2-5cm")
    s = o["size"] or 0
    keys.append("size<15cm" if s < 0.15 else "size15-40cm" if s < 0.4 else "size>40cm")
    return keys


def evaluate(model, items, dev) -> dict:
    model.eval()
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    objects, fp_area, fp_comp, n_none = [], [], 0, 0
    with torch.no_grad():
        for i in range(0, len(items), 16):
            batch = items[i : i + 16]
            fr, fq, low, _ = batchify(batch, dev, False)
            p = torch.softmax(model(fr, fq, low), 1).cpu().numpy()
            for j, x in enumerate(batch):
                pred = p[j].argmax(0)
                lab = x["lab"]
                ok = lab != IGNORE
                for c in (APPEARED, MISSING):
                    tp[c] += ((pred == c) & (lab == c) & ok).sum()
                    fp[c] += ((pred == c) & (lab != c) & ok).sum()
                    fn[c] += ((pred != c) & (lab == c)).sum()
                if x["meta"].get("kind") == "none":
                    n_none += 1
                    ch = (pred > 0) & ok
                    fp_area.append(float(ch.sum() / max(ok.sum(), 1)))
                    _, _, st, _ = cv2.connectedComponentsWithStats(ch.astype(np.uint8))
                    fp_comp += int((st[1:, cv2.CC_STAT_AREA] >= 60).sum())
                em_path = x["dir"] / "edit_masks.npz"
                em = np.load(em_path) if em_path.exists() else None
                H4, W4 = pred.shape
                for k, e in enumerate(x["meta"].get("edits", [])):
                    if e["type"] == "decoy_fill" or em is None:
                        continue
                    cover = {}
                    for c, key in ((APPEARED, f"e{k}_appeared"), (MISSING, f"e{k}_missing")):
                        if key in em.files:
                            m = cv2.resize(em[key].astype(np.uint8), (W4, H4), interpolation=cv2.INTER_AREA) > 0
                            if m.sum() >= 3:
                                cover[c] = float((pred[m] == c).mean())
                    if cover:
                        objects.append(
                            dict(
                                type=e["type"],
                                size=e.get("size"),
                                dist=e.get("dist"),
                                small=e.get("small", False),
                                detected=bool(max(cover.values()) >= 0.25),
                            )
                        )
    pixel = {
        name: dict(
            precision=float(tp[c] / max(tp[c] + fp[c], 1)),
            recall=float(tp[c] / max(tp[c] + fn[c], 1)),
            f1=float(2 * tp[c] / max(2 * tp[c] + fp[c] + fn[c], 1)),
        )
        for c, name in ((APPEARED, "appeared"), (MISSING, "missing"))
    }
    by = {}
    for o in objects:
        for k in edit_buckets(o):
            by.setdefault(k, []).append(o["detected"])
    return dict(
        pixel=pixel,
        object_recall={k: dict(recall=float(np.mean(v)), n=len(v)) for k, v in sorted(by.items())},
        object_recall_all=float(np.mean([o["detected"] for o in objects])) if objects else None,
        n_edits=len(objects),
        nochange_fp_area_mean=float(np.mean(fp_area)) if fp_area else None,
        nochange_false_components=fp_comp,
        n_nochange=n_none,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()
    cfg = load_settings(args.config).train
    scene = Scene(args.scene)
    out = scene.runs / "checker"
    out.mkdir(parents=True, exist_ok=True)
    root = scene.checker / "pairs"
    t0 = time.time()
    tr = [load_pair(d) for d in sorted((root / "train").glob("*")) if (d / "f_real.npy").exists()]
    va = [load_pair(d) for d in sorted((root / "val").glob("*")) if (d / "f_real.npy").exists()]
    log(f"checker: {len(tr)} train / {len(va)} val pairs loaded in {time.time() - t0:.0f} s")
    cnt = np.array([sum(int((x["lab"] == c).sum()) for x in tr) for c in range(3)], float)
    w = np.minimum(np.sqrt(cnt[0] / np.maximum(cnt, 1)), MAX_CLASS_WEIGHT)
    with gpu_stage("checker_train", need_mib=cfg.need_mib) as watchdog:
        dev = "cuda"
        model = Checker().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        steps_per = int(np.ceil(len(tr) / cfg.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per, pct_start=0.1
        )
        wt = torch.as_tensor(w, dtype=torch.float32, device=dev)
        ep0, best, history = 0, -1.0, []
        ck = out / "ckpt.pt"
        if ck.exists():
            c = torch.load(ck, map_location=dev)
            model.load_state_dict(c["model"])
            opt.load_state_dict(c["opt"])
            sched.load_state_dict(c["sched"])
            ep0, best, history = c["epoch"] + 1, c["best"], c["log"]
            log(f"checker: resumed at epoch {ep0}")
        rng = np.random.default_rng(0)
        t_train = time.time()
        for ep in range(ep0, cfg.epochs):
            model.train()
            order = rng.permutation(len(tr))
            losses = []
            for s in range(steps_per):
                fr, fq, low, y = batchify([tr[i] for i in order[s * cfg.batch : (s + 1) * cfg.batch]], dev, True)
                loss = focal_ce(model(fr, fq, low), y, wt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                losses.append(float(loss.detach()))
            ev = evaluate(model, va, dev)
            score = (ev["pixel"]["appeared"]["f1"] + ev["pixel"]["missing"]["f1"]) / 2
            history.append(
                dict(
                    epoch=ep,
                    loss=float(np.mean(losses)),
                    val_f1_appeared=ev["pixel"]["appeared"]["f1"],
                    val_f1_missing=ev["pixel"]["missing"]["f1"],
                    obj_recall=ev["object_recall_all"],
                    fp_area=ev["nochange_fp_area_mean"],
                    s=round(time.time() - t0, 1),
                )
            )
            log("checker", history[-1])
            if score > best:
                best = score
                torch.save(dict(model=model.state_dict(), epoch=ep, val=ev), out / "best.pt")
            torch.save(
                dict(
                    model=model.state_dict(),
                    opt=opt.state_dict(),
                    sched=sched.state_dict(),
                    epoch=ep,
                    best=best,
                    log=history,
                ),
                ck,
            )
            if (time.time() - t_train) / 60 > cfg.max_minutes:
                log("checker: time budget reached")
                break
        best_ck = torch.load(out / "best.pt", map_location=dev)
        model.load_state_dict(best_ck["model"])
        ev = evaluate(model, va, dev)
        ev.update(
            best_epoch=best_ck["epoch"],
            train_pairs=len(tr),
            val_pairs=len(va),
            label_weights=w.tolist(),
            train_min=round((time.time() - t_train) / 60, 1),
            torch_peak_mib=int(torch.cuda.max_memory_reserved() / 2**20),
            gpu_peak_used_all_mib=watchdog.peak_used,
        )
    write_json(ev, out / "val_report.json")
    write_json(history, out / "train_log.json")
    log(json.dumps({k: v for k, v in ev.items() if k != "object_recall"}))


if __name__ == "__main__":
    main()
