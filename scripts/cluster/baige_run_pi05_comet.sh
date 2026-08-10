#!/usr/bin/env bash
# One command for Demo Pipeline π0.5-Comet continuation on every Baige node.
# Baige injects node-level RANK/WORLD_SIZE and runs this same command on master
# and workers; one JAX process controls all NPROC_PER_NODE GPUs on its node.
set -euo pipefail

PROJECT_ROOT="${EMBODIED_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

for name in MASTER_ADDR MASTER_PORT RANK WORLD_SIZE NPROC_PER_NODE; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: missing Baige environment variable: $name" >&2
    exit 2
  fi
done

profile="${PI05_COMET_PROFILE:-formal}"
run_id="${BAIGE_RUN_ID:-${AIHC_JOB_ID:-${JOB_ID:-pi05-comet-${MASTER_ADDR}}}}"
args=(
  --baige
  --profile "$profile"
  --run-id "$run_id"
)
if [[ "${PI05_COMET_RESUME:-0}" == "1" ]]; then
  args+=(--resume)
fi

exec python experiments/custom/pi05_comet_behavior1k_all/run.py "${args[@]}" "$@"
