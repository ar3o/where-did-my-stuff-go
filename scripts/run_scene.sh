#!/usr/bin/env bash
# All stages of one scene, one log per stage under $WHATCHANGED_RUNS/<scene>/logs/.
#   bash scripts/run_scene.sh <scene> <reference video> <names.json> [groups.json] <photo or video>...
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY="${SPLAT_PY:-python}"
SCENE="$1"; REF="$2"; NAMES="$3"; shift 3
GROUPS=()
if [[ "${1:-}" == *.json ]]; then GROUPS=(--groups "$1"); shift; fi
LOGS="${WHATCHANGED_RUNS:-runs}/$SCENE/logs"
mkdir -p "$LOGS"
run() { echo "$(date +%T) $1"; "$PY" -m "whatchanged.$1" --scene "$SCENE" "${@:2}" > "$LOGS/$1.log" 2>&1; }
run ingest.run --reference "$REF" --names "$NAMES" "${GROUPS[@]}" --queries "$@"
run sfm.run
run splat.run
run inventory.run
run checker.run
run changes.run
run render.film
echo "$(date +%T) done: ${WHATCHANGED_DATA:-data}/$SCENE/changes/summary.json"
