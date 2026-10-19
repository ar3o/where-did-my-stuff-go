"""Robot run: check or fetch a remembered object in an episode world, then evaluate the run when ground truth exists.

    python -m whatchanged.navigation.run --memory config1 --episode config1_after \
        --prompt "check the hammer is here" --run r01

--memory is the scene whose splat, inventory and walkable map the robot remembers; --episode is the world its head
camera sees: another scene registered into the memory frame, a prepared world folder (world.py), or 'memory' for the
no-change control. See sim.py for the behaviour and the outputs.
"""

from __future__ import annotations

import argparse

from whatchanged.navigation import common
from whatchanged.navigation.evaluate import evaluate
from whatchanged.navigation.sim import run
from whatchanged.navigation.world import resolve_world


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory", required=True, help="scene the robot remembers")
    ap.add_argument("--episode", required=True, help="scene (or prepared world, or 'memory') the head camera sees")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--run", required=True, help="run name")
    ap.add_argument("--max_s", type=float, default=0, help="shorter time budget, for debugging only")
    a = ap.parse_args()
    if a.max_s:
        common.override("sim", "max_s", a.max_s)
    world_splat, start = resolve_world(a.memory, a.episode)
    run(a.memory, a.episode, a.prompt, a.run, world_splat=world_splat, start=start)
    evaluate(a.memory, a.run)


if __name__ == "__main__":
    main()
