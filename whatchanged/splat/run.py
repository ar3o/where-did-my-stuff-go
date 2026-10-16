"""Splat stage: train, optional cleanup, photo localisation with predicted views, support / known maps.

python -m whatchanged.splat.run --scene config1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from whatchanged.common.scene import Scene
from whatchanged.splat.settings import load_settings


def step(module: str, scene: str, cfg: Path | None) -> None:
    cmd = [sys.executable, "-m", module, "--scene", scene] + (["--config", str(cfg)] if cfg else [])
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()
    settings = load_settings(args.config)
    if not Scene(args.scene).splat.exists():
        step("whatchanged.splat.train", args.scene, args.config)
        if settings.cleanup.enabled:
            step("whatchanged.splat.cleanup", args.scene, args.config)
    step("whatchanged.splat.localize", args.scene, args.config)
    step("whatchanged.splat.support", args.scene, args.config)


if __name__ == "__main__":
    main()
