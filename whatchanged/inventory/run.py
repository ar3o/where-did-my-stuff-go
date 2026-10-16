"""Labelled 3D inventory of the reference scene. The only scene-specific input is the name list (SAM 3 prompts).

    python -m whatchanged.inventory.run --scene config1

1. SAM 3 on every `frame_every`-th registered walk frame: work image + 2 x 2 tiles, all phrases (SAM 3 env worker).
2. DINOv2 PCA features of all walk frames (vision env worker); the PCA is fitted here and reused by the checker.
3. Gaussian voting, association, region growing and fragment merging (voting.py) -> inventory/raw.{json,npz}.
4. Fingerprints of the raw objects, geometric merge, SAM 3 votes, best-view crops, SigLIP 2 names (naming.py)
   -> inventory/inventory.json (objects with id, name, name_conf, runner_up, cls, box {centre, size, yaw}) and
   inventory/inventory.npz (gid: object number per Gaussian, 0 = none), inventory/fingerprints.npz.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from whatchanged import config
from whatchanged.common import features
from whatchanged.common.gpu import gpu_stage
from whatchanged.common.io import log, read_json, save_npz, write_json
from whatchanged.common.scene import Scene
from whatchanged.common.workers import run_worker
from whatchanged.inventory import detect, fingerprint, naming, voting
from whatchanged.splat import gaussians as gs


@dataclass
class InventoryConfig:
    frame_every: int = 2
    sam3_min_p: float = 0.35
    sam3_topk: int = 6
    name_margin: float = 0.25
    sam3_need_mib: int = 5000
    dino_need_mib: int = 3000
    siglip_need_mib: int = 2500
    render_need_mib: int = 3500


def inventory_frames(scene: Scene, every: int) -> list[str]:
    return [f["name"] for f in scene.poses()["frames"]][::every]


def run_sam3(scene: Scene, names: list[str], cfg: InventoryConfig) -> None:
    P = scene.poses()
    items = []
    for n in names:
        items += detect.jobs_for_image(
            scene.frame_image(n),
            scene.frames_hi / f"{n}.jpg",
            scene.inventory / "sam3" / f"frame_{n}",
            scene.inventory / "tiles",
            P["W"],
            P["H"],
        )
    jobs = dict(prompts=scene.phrases(), min_p=cfg.sam3_min_p, topk=cfg.sam3_topk, items=items)
    run_worker("sam3", "whatchanged.common.sam3_worker", jobs, scene.inventory / "jobs_sam3.json", cfg.sam3_need_mib)


def frame_features(scene: Scene, need_mib: int) -> None:
    frames = [f["name"] for f in scene.poses()["frames"]]
    items = [(scene.frame_image(n), scene.features / "frames" / f"{n}.npy") for n in frames]
    features.encode(
        items,
        scene.features / "pca.npz",
        [scene.frame_image(n) for n in frames],
        scene.features / "jobs_frames.json",
        need_mib,
    )


def detections(scene: Scene, names: list[str]) -> dict:
    P = scene.poses()
    phrases = set(scene.phrases())
    return {n: detect.merged(scene.inventory / "sam3" / f"frame_{n}", P["W"], P["H"], phrases) for n in sorted(names)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()
    cfg = config.load(InventoryConfig, "inventory", args.config)
    scene = Scene(args.scene)
    inv = scene.inventory
    names = inventory_frames(scene, cfg.frame_every)
    run_sam3(scene, names, cfg)
    frame_features(scene, cfg.dino_need_mib)
    P = scene.poses()
    K, W, H = np.array(P["K"]), P["W"], P["H"]
    T_of = {f["name"]: np.array(f["T_wc"]) for f in P["frames"]}
    dets_of = detections(scene, names)
    sp = gs.core(gs.load(scene.splat))
    with gpu_stage("inventory", need_mib=cfg.render_need_mib):
        tsp = gs.to_torch(sp)
        if not (inv / "raw.json").exists():
            with_dets = {n: d for n, d in dets_of.items() if d}
            objs, gid, stats = voting.build(sp, T_of, with_dets, K, W, H, scene.phrase_class(), cfg.name_margin)
            save_npz(inv / "raw.npz", compressed=True, gid=gid)
            write_json(dict(objects=objs, phrase_score_stats=stats, n_frames=len(with_dets)), inv / "raw.json")
        raw = read_json(inv / "raw.json")
        gid_raw = np.load(inv / "raw.npz")["gid"]
        refs = fingerprint.fingerprints(scene, tsp, raw["objects"], gid_raw)
        ids = sorted(refs)
        save_npz(
            inv / "fingerprints.npz",
            ids=np.array(ids),
            emb=np.array([refs[i]["emb"] for i in ids]),
            col=np.array([refs[i]["col"] for i in ids]),
            size=np.array([refs[i]["size"] for i in ids]),
        )
        alive, merges, gid = naming.geometric_merge(raw["objects"], gid_raw, sp, {i: refs[i]["emb"] for i in ids})
        log(f"inventory: geometric merges {merges}")
        keep_ids = sorted(alive, key=naming.num)
        votes, views = naming.votes_and_views(scene, tsp, gid, keep_ids, dets_of, raw["phrase_score_stats"])
        crops = naming.best_view_crops(scene, tsp, gid, keep_ids, views, inv / "crops")
    phrases = scene.phrases()
    run_worker(
        "vision",
        "whatchanged.common.siglip_worker",
        dict(phrases=phrases, crops=crops, out=str(inv / "siglip.npz")),
        inv / "jobs_siglip.json",
        cfg.siglip_need_mib,
    )
    z = np.load(inv / "siglip.npz")
    siglip = {str(k): z["probs"][i].astype(np.float64) for i, k in enumerate(z["keys"])}
    objects = naming.name_objects(alive, keep_ids, votes, crops, siglip, phrases, scene.phrase_class(), cfg.name_margin)
    save_npz(inv / "inventory.npz", compressed=True, gid=gid)
    write_json(
        dict(
            objects=objects,
            merges=merges,
            phrase_score_stats=raw["phrase_score_stats"],
            rule="name list only: no renames, merges or aliases by hand",
        ),
        inv / "inventory.json",
    )
    uncertain = sum("(uncertain)" in o["name"] for o in objects)
    log(f"inventory: {len(objects)} objects ({uncertain} uncertain names)")


if __name__ == "__main__":
    main()
