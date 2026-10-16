"""Changes stage: re-identification inputs, SAM 3 on the photos, detection features, calibration, and the per-group
comparison with the reference.

    python -m whatchanged.changes.run --scene config1

Outputs changes/gXX/changes.json per group (events with kind, name, confidence, 3D centres, per-photo outlines,
evidence and a sentence; the room state after the group), changes/summary.json and changes/confidences.md.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from whatchanged import config
from whatchanged.changes import calibrate, detections, infer, references
from whatchanged.checker import model as checker_model
from whatchanged.common.gpu import gpu_stage
from whatchanged.common.io import read_json
from whatchanged.common.scene import Scene
from whatchanged.common.workers import run_worker
from whatchanged.splat import gaussians as gs


@dataclass
class ChangesConfig:
    sam3_min_p: float = 0.2
    sam3_topk: int = 8
    name_margin: float = 0.25
    sam3_need_mib: int = 5000
    siglip_need_mib: int = 2500
    render_need_mib: int = 3000


def confidence_table(scene: Scene) -> None:
    lines = ["| group | event | change % | name % | runner-up name |", "|---|---|---|---|---|"]
    for k in sorted(scene.groups()):
        rec = read_json(scene.changes / f"g{k:02d}" / "changes.json")
        if not rec["events"]:
            lines.append(f"| {k} | (no confident change) | | | |")
        for e in rec["events"]:
            verb = {"removed": "removed", "added": "added"}.get(e["kind"], f"moved {100 * e.get('dist_m', 0):.0f} cm")
            lines.append(
                f"| {k} | {e['name'].replace(' (uncertain)', '?')} {verb} | {100 * e['change_conf']:.0f} | "
                f"{100 * e.get('name_conf', 0):.0f} | {e.get('runner_up') or ''} |"
            )
    (scene.changes / "confidences.md").write_text(
        "Every group compared with the reference splat and inventory on its own. Change % is calibrated on held-out "
        "synthetic edits; name % combines SAM 3 votes and SigLIP 2.\n\n" + "\n".join(lines) + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()
    cfg = config.load(ChangesConfig, "changes", args.config)
    scene = Scene(args.scene)
    groups = scene.groups()
    photos = [p for k in sorted(groups) for p in groups[k]]
    sp = gs.load(scene.splat)
    with gpu_stage("changes_prepare", need_mib=cfg.render_need_mib):
        items = references.prepare(scene, gs.to_torch(sp), photos)
    jobs = dict(prompts=scene.phrases(with_static=False), min_p=cfg.sam3_min_p, topk=cfg.sam3_topk, items=items)
    run_worker("sam3", "whatchanged.common.sam3_worker", jobs, scene.changes / "jobs_sam3.json", cfg.sam3_need_mib)
    with gpu_stage("changes_detections", need_mib=cfg.render_need_mib):
        crops = detections.compute(scene, gs.to_torch(sp), photos)
    run_worker(
        "vision",
        "whatchanged.common.siglip_worker",
        dict(phrases=scene.phrases(with_static=False), crops=crops, out=str(scene.changes / "siglip.npz")),
        scene.changes / "jobs_siglip.json",
        cfg.siglip_need_mib,
    )
    calibrate.calibrate(scene, checker_model.load(scene.runs / "checker" / "best.pt", "cpu"))
    with gpu_stage("changes_infer", need_mib=cfg.render_need_mib):
        ctx = infer.Context(scene, gs.to_torch(sp), sp, cfg.name_margin)
        infer.run_all(ctx, groups)
    confidence_table(scene)


if __name__ == "__main__":
    main()
