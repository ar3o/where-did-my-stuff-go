# Where did my stuff go: navigation

The navigation module puts a simulated robot in the room. Its memory is what the pipeline built from the reference walk:
the splat, the labelled inventory and a walkable map derived from them. Its head camera renders a different splat of
the same room, the episode, from wherever the robot stands. Given a prompt such as "check the hammer is here" or "bring
me the banana", the robot scans from the doorway, walks to a pose with a clear view of the remembered place, compares
what it sees with its memory through the learned checker, and either confirms the object, finds it moved, searches the
parts of the room it has not seen, or says why it cannot tell. Every sentence it produces is a template filled from
measured facts.

## Before a run

The memory scene needs the full pipeline up to the checker (`whatchanged.splat.run`, `whatchanged.inventory.run`,
`whatchanged.checker.run`). Two further files belong to the memory scene and live in `<runs>/<scene>/navigation/`:
`nochange_tau.json`, written by `calib_nochange`, which measures what an unchanged object looks like to the checker on
the memory alone, and `move_match.json`, which holds the fingerprint score threshold (`{"threshold": ...}`) above which a
region is taken to be the target seen somewhere else. A scene without its own threshold can borrow another scene's with
`move_match_from` in the `scenes` section of `configs/navigation.json`; the same section can point any scene at other
checker, PCA or calibration files.

The episode is another scene of the same room whose camera poses, splat and reference-depth cache are already expressed
in the memory scene's frame. Its `poses_memory.json` is used when present, otherwise its `poses.json`. The robot starts
at the first frame of the episode's walk where a standing camera is over free floor of the memory map. Passing
`--episode memory` runs the no-change control, with the memory as the world.

## Commands

```bash
$SPLAT_PY -m whatchanged.navigation.memory_map --scene config1
$SPLAT_PY -m whatchanged.navigation.calib_nochange --scene config1
$SPLAT_PY -m whatchanged.navigation.run --memory config1 --episode config1_after \
    --prompt "check the hammer is here" --run r01
python -m whatchanged.navigation.collect --scene config1
python -m whatchanged.navigation.score --results runs/navigation/RESULTS_config1.json
```

`memory_map` finds the floor plane, the local floor height (rugs are the walking surface), the occupied, free and
unknown cells from the reference depths and the inventory footprints, and writes `<data>/<scene>/navigation/
memory_map.{npz,json}` with a top-down still in `<runs>/navigation/qc/`. The robot run rebuilds the map by itself
whenever the inventory has changed since the map was made. `run` writes the film to `<runs>/navigation/videos/`, stills
and a summary map to `<runs>/navigation/qc/`, and `task.json`, `log.json` and `eval.json` to
`<data>/<scene>/navigation/runs/<run>/`. The evaluation reads ground truth only when a hand-written
`<data>/<scene>/navigation/truth/<episode>.json` exists. `collect` gathers every run of a scene into one table and
`score` compares the verdicts with by-eye labels if you write some in `<runs>/navigation/truth/truth.json`; none are shipped here.

The run uses the GPU for rendering, DINOv2 and the checker, inside the pipeline's GPU lock. The prompt is parsed by the
SigLIP 2 text tower on the CPU through the usual vision worker, and new obstacles are named by a SigLIP 2 server in the
vision environment that stays loaded for the whole run, both with the interpreter and weights set in
`configs/environment.yaml`.

## Module layout

`run.py` is the command line entry. `sim.py` holds the robot loop: the sensor (episode render and memory renders at the
same pose), the checker wrapper, the motion primitives, the fixed decision rules and the film frames. `belief.py`
accumulates the evidence: per-object log-odds of present or gone from calibrated checker probabilities and appearance
scores, an occupancy grid observed from the sensor depth with the memory map as its prior, re-identification of the
target in newly appeared regions, and the room description. `planner.py` is A* on the grid for a round footprint, with
string pulling, a squeeze path that measures the narrowest gap, and the shrink ladder of footprints applied by the
belief. `nbv.py` renders the visibility and coverage tables used to choose where to look from. `task.py` resolves the
prompt to inventory objects, `language.py` builds the sentences and spatial relations, `namer.py` and
`siglip_server.py` name new obstacles, `text_worker.py` embeds the prompt. `memory_map.py`, `calib_nochange.py`,
`appearance.py` and `boxmask.py` prepare the memory side; `world.py` resolves episodes; `overlay.py` and `helpers.py`
hold drawing and path helpers; `evaluate.py`, `collect.py` and `score.py` measure the runs. Every threshold is in
`configs/navigation.json`, with the reason for each in its `_why` section.

## What I had to fix on the way

The robot's first runs were confidently wrong in ways that taught me more than the ones that worked. In one later walk it said it could not see the hammer even though the hammer was plainly on its pillow in the film, and the reason was that its first full turn at the doorway looked back into the hallway, which the later walk never filmed, so the splat there was a fog of floaters that the robot took for a wall of new obstacles all round it. A real depth camera does not see floaters, so the simulated one should not either, and the robot now only gets a reading where at least three cameras of that later walk saw the surface, the same rule the memory uses for what it knows. In the untouched room it kept reporting small new obstacles that turned out to be rims of furniture the memory map had drawn a few centimetres too small, which the memory-render test fixed, and the one new obstacle that is left in the untouched room is the door, which really was at a different angle that day.

Two of the most confident mistakes were "not here" answers for things that were there, a hammer with a cereal box put down right in front of it and a banana with another box standing on its side table, and both came from the checker reading a place hidden behind something new as a place that is empty. The robot now calls a place covered when its depth over the object is clearly in front of where the memory has the object, and it says it could not verify instead of saying the object has gone, because a robot that tells you something is lost when it is only hidden is worse than one that says it does not know. The search used to pan its head left and right without turning its body and never looked at the sofa behind it, so it now turns all the way round at every stop, and once it found the wrong object, taking the pipe wrench, nudged a little on its cabinet, for the moved hammer, so a candidate standing on another remembered object's place now counts as that object.

None of this is scored. I checked the robot's answers by eye against the real frames of each later walk while developing it, and I changed it while watching those same runs, so the films are demonstrations and not a test.

## The fingerprint threshold used in the films

The films use a fingerprint match threshold of 0.45, measured on held-out frames of the photo scene's walk for a precision of 0.95 and borrowed by the navigation memory scene. It is in [examples/navigation/move_match.json](../examples/navigation/move_match.json); copy it to `<runs>/<scene>/navigation/move_match.json` before a run, since nothing in the pipeline writes that file yet.
