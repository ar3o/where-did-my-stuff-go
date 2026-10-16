#!/usr/bin/env bash
# Splat training alone, resumable: a rerun continues from the last 5000-step checkpoint.
#   bash scripts/train_splat.sh <scene> [iterations]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${SPLAT_PY:-python}" -m whatchanged.splat.train --scene "$1" ${2:+--iters "$2"}
