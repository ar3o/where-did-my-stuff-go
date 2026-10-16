"""Whole pipeline: reference walk video + later photos (or a video) + a name list -> what changed, as JSON, stills
and a film.

    python -m whatchanged.run --reference ref.MOV --queries photos/*.HEIC --names names.json --out results/ \\
        [--groups groups.json] [--scene NAME] [--from-stage splat]

Stages, each also runnable alone (python -m whatchanged.<stage>.run --scene NAME): ingest, sfm, splat, inventory,
checker, changes, render. Every stage skips work whose outputs exist, so a rerun continues where it stopped. The final
JSON, stills and film are copied to --out.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from whatchanged.common.io import log, read_json
from whatchanged.common.scene import Scene

STAGES = ["ingest", "sfm", "splat", "inventory", "checker", "changes", "render"]
MODULES = {s: f"whatchanged.{s}.run" for s in STAGES} | {"render": "whatchanged.render.film"}


def stage_command(stage: str, args) -> list[str]:
    cmd = [sys.executable, "-m", MODULES[stage], "--scene", args.scene]
    if stage == "ingest":
        cmd += ["--reference", str(args.reference), "--names", str(args.names), "--queries", *map(str, args.queries)]
        if args.groups:
            cmd += ["--groups", str(args.groups)]
    return cmd


def collect(scene: Scene, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for k in sorted(scene.groups()):
        src = scene.changes / f"g{k:02d}" / "changes.json"
        if src.exists():
            shutil.copy(src, out / f"changes_g{k:02d}.json")
    for name in ("summary.json", "confidences.md"):
        if (scene.changes / name).exists():
            shutil.copy(scene.changes / name, out / name)
    film = scene.runs / "film"
    if (film / "film.mp4").exists():
        shutil.copy(film / "film.mp4", out / "film.mp4")
    if (film / "stills").exists():
        shutil.copytree(film / "stills", out / "stills", dirs_exist_ok=True)
    for k, lines in read_json(scene.changes / "summary.json")["groups"].items():
        log(f"group {k}: " + ("; ".join(lines) if lines else "no confident change"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--reference", required=True, type=Path, help="walk-around video of the room in its reference state"
    )
    ap.add_argument(
        "--queries", required=True, nargs="+", type=Path, help="photos (HEIC / JPEG) or one video, taken later"
    )
    ap.add_argument("--names", required=True, type=Path, help="name list JSON: furniture, items, static")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--groups", type=Path, help="JSON of photo groups, one per room state (default: all photos in one)")
    ap.add_argument("--scene", help="scene name under WHATCHANGED_DATA (default: the reference video's name)")
    ap.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    args = ap.parse_args()
    args.scene = args.scene or args.reference.stem
    for stage in STAGES[STAGES.index(args.from_stage) :]:
        log(f"stage {stage}")
        subprocess.run(stage_command(stage, args), check=True)
    collect(Scene(args.scene), args.out)


if __name__ == "__main__":
    main()
