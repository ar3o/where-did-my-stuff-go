"""Polite use of a shared GPU: one stage of this pipeline at a time, wait for free memory, and stop only this process
when the whole GPU runs low. Other processes are never touched.

    with gpu_stage("train", need_mib=5000) as watchdog: ...
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
import threading
import time

from whatchanged import paths
from whatchanged.common.io import log

ABORT_FREE_MIB = int(os.environ.get("WHATCHANGED_GPU_FLOOR_MIB", "3000"))
ABORT_EXIT_CODE = 3


def gpu_mem() -> tuple[int, int, int]:
    """(free, used, total) MiB of GPU 0, or zeros when nvidia-smi is unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,memory.used,memory.total", "--format=csv,noheader,nounits"],
            timeout=20,
        )
        free, used, total = (int(x) for x in out.decode().split("\n")[0].split(","))
        return free, used, total
    except Exception:
        return 0, 0, 0


def wait_for_memory(need_mib: int, name: str, poll_s: int = 20) -> None:
    """Block until the minimum free memory over about 6 s covers need_mib plus the abort floor."""
    target = need_mib + ABORT_FREE_MIB
    t0, said = time.time(), False
    while True:
        readings = []
        for _ in range(3):
            readings.append(gpu_mem()[0])
            time.sleep(2)
        if min(readings) >= target:
            if said:
                log(f"[gpu] {name}: {min(readings)} MiB free after {time.time() - t0:.0f} s")
            return
        if not said:
            log(f"[gpu] {name}: waiting for {target} MiB free, {min(readings)} MiB available")
            said = True
        time.sleep(poll_s)


class Watchdog(threading.Thread):
    """Exits this process when whole-GPU free memory stays below the floor for two readings; records peak use."""

    def __init__(self, name: str, period: float = 1.0):
        super().__init__(daemon=True)
        self.stage, self.period, self.stopped = name, period, False
        self.peak_used, self.min_free = 0, 10**9

    def run(self) -> None:
        low = 0
        while not self.stopped:
            free, used, _ = gpu_mem()
            if used:
                self.peak_used = max(self.peak_used, used)
                self.min_free = min(self.min_free, free)
                low = low + 1 if free < ABORT_FREE_MIB else 0
                if low >= 2:
                    log(f"[gpu] {self.stage}: {free} MiB free < {ABORT_FREE_MIB} MiB, stopping this process")
                    os._exit(ABORT_EXIT_CODE)
            time.sleep(self.period)


@contextlib.contextmanager
def gpu_stage(name: str, need_mib: int = 4000):
    lock_path = paths.RUNS / "gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock:
        t0 = time.time()
        fcntl.flock(lock, fcntl.LOCK_EX)
        if time.time() - t0 > 1:
            log(f"[gpu] {name}: got the pipeline lock after {time.time() - t0:.0f} s")
        wait_for_memory(need_mib, name)
        watchdog = Watchdog(name)
        watchdog.start()
        try:
            yield watchdog
        finally:
            watchdog.stopped = True
