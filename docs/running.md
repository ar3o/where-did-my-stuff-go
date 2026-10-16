# Where did my stuff go: running the pipeline

Set up the environments first ([installation.md](installation.md)), then from the repository root:

```bash
export PYTHONPATH=$PWD
```

Every stage reads and writes under `$WHATCHANGED_DATA/<scene>/` and `$WHATCHANGED_RUNS/<scene>/` (default `data/` and
`runs/` in the checkout) and skips work whose outputs exist, so a stopped run continues where it stopped. Settings live in
`configs/<stage>.yaml`; their defaults are the settings behind the config1 results. GPU steps wait until enough memory
is free and stop themselves (exit code 3) if the whole GPU drops below 3 GB free, so the pipeline can share a GPU.

## All at once

```bash
$SPLAT_PY -m whatchanged.run --reference ref.MOV --queries photos/*.HEIC --names names.json \
    --groups groups.json --out results/ [--scene config1] [--from-stage splat]
```

Inputs:

- `--reference`: a walk-around video of the room in its reference state. iPhone HDR is fine. Film slowly, at the height
  you will take photos from, and pass close to the places you care about.
- `--queries`: photos (HEIC, JPEG, PNG) or one video taken later.
- `--names`: `{"furniture": [...], "items": [...], "static": [...]}`. These are the SAM 3 prompts and the only scene input.
  "static" parts (walls, floor, rug) are detected so they never become objects.
- `--groups` (optional): `{"groups": {"0": ["IMG_2570", ...], "1": [...]}}`, the photos of each room state. Each group is
  compared with the reference on its own. Without it all photos form one group.

## Stage by stage

```bash
S=config1
$SPLAT_PY -m whatchanged.ingest.run --scene $S --reference ref.MOV --queries photos/*.HEIC --groups groups.json --names names.json
$SPLAT_PY -m whatchanged.sfm.run --scene $S
$SPLAT_PY -m whatchanged.splat.run --scene $S        # train, [cleanup], localize, support
$SPLAT_PY -m whatchanged.inventory.run --scene $S
$SPLAT_PY -m whatchanged.checker.run --scene $S      # pairs, features, train, infer, calibrate
$SPLAT_PY -m whatchanged.changes.run --scene $S
$SPLAT_PY -m whatchanged.render.film --scene $S [--groups 5 8]
```

`scripts/run_scene.sh` runs the same sequence with a log per stage. The sub-steps of the splat and checker stages also
run alone: `whatchanged.splat.{train,cleanup,localize,support}`, `whatchanged.checker.{pairs,train,infer,calibrate}`.

Measured on config1 (RTX 4090 shared with other jobs); blank means it was not timed on its own:

| Stage | Time | Main outputs |
|:--|:--|:--|
| ingest | | `frames/`, `frames_hi/`, `frames.json`, `photos/`, `photos_hi/`, `groups.json` |
| sfm | COLMAP 4.6 min (CPU) | `depth/`, `sfm/`, `poses.json`, `photos_sfm.json`, `sfm_points.npz` |
| splat | training 13.3 min, localisation 19 s per photo, support 63 s for 71 photos | `splat/splat.npz`, `predicted/<photo>/`, `frame_known/` |
| inventory | Gaussian voting 15 min | `inventory/inventory.{json,npz}`, `inventory/raw.{json,npz}` |
| checker | training 2.8 min, calibration 5 min | `checker/pairs/`, `<runs>/checker/best.pt`, `conf_calibration.json` |
| changes | SAM 3 about 1 h, decisions for 23 groups 76 s | `changes/gXX/changes.json`, `summary.json`, `confidences.md` |
| render | | `<runs>/film/gXX.mp4`, `film.mp4`, `stills/` |

## Output format

`changes/gXX/changes.json`:

```
group, photos, good_photos (usable after the pose and depth checks)
events: [{kind: removed | moved | added, object (inventory id I<k>, or g<k>A<n> for an added thing),
          name, name_conf, runner_up, change_conf (calibrated), centre_before / centre_after (world, metres, z up),
          box_after, dist_m (moved), photos, outlines {photo: {before, after}: pixel polygons},
          evidence {photo: per-photo state and probabilities}, sentence}]
state_after: removed ids, moved {id: centre}, added things
```

World frame: metres, z up, the floor (or the largest support plane under the cameras) at z = 0, origin where the walk's
cameras looked on average.

## Notes for a new room

- 50 names are plenty. Name things by what they look like ("blue plastic bottle"), since SigLIP 2 and SAM 3 both read the
  name.
- Photos taken far from the walk's viewpoints fail the depth check and are left out (see `good_photos`).
- The splat cleanup (`splat.cleanup.enabled`) helps rooms with many floaters; the config1 results use the raw splat.
