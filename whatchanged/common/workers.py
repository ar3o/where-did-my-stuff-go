"""Model workers that run in their own Python environments (SAM 3, and MoGe-3 / DINOv2 / SigLIP 2 in the vision env).

Each worker reads one JSON job file and writes its outputs next to the job's targets. The pipeline starts them as
subprocesses with the repository on PYTHONPATH; configs/environment.yaml says which interpreter runs which worker.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from whatchanged import config, paths
from whatchanged.common.io import log, write_json


@dataclass
class EnvironmentConfig:
    vision_python: str = "python"  # MoGe-3, DINOv2, SigLIP 2
    vision_pythonpath: str = ""  # extra entries, e.g. a MoGe checkout when it is not pip-installed
    sam3_python: str = "python"
    sam3_repo: str = ""
    sam3_checkpoint: str = str(paths.WEIGHTS / "sam3" / "sam3.pt")
    dino_hub: str = str(paths.WEIGHTS / "torch_hub" / "facebookresearch_dinov2_main")
    siglip: str = "google/siglip2-so400m-patch14-384"
    moge: str = "Ruicheng/moge-3-vitl"


def environment() -> EnvironmentConfig:
    return config.load(EnvironmentConfig, "environment")


def run_worker(kind: str, module: str, jobs: dict, job_path: Path, need_mib: int) -> None:
    """Write `jobs` to job_path and run `python -m module job_path` in the environment of `kind` (vision or sam3)."""
    env_cfg = environment()
    jobs = dict(jobs, need_mib=need_mib)
    if kind == "sam3":
        python, extra = env_cfg.sam3_python, env_cfg.sam3_repo
        jobs.setdefault("checkpoint", env_cfg.sam3_checkpoint)
    else:
        python, extra = env_cfg.vision_python, env_cfg.vision_pythonpath
        jobs.setdefault("dino_hub", env_cfg.dino_hub)
        jobs.setdefault("siglip", env_cfg.siglip)
        jobs.setdefault("moge", env_cfg.moge)
    write_json(jobs, job_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(p for p in (extra, str(paths.REPO), env.get("PYTHONPATH", "")) if p)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log(f"worker {module}: {job_path}")
    subprocess.run([python, "-m", module, str(job_path)], env=env, check=True)
