#!/usr/bin/env bash
set -euo pipefail

# Data-only WSA1.5 smoke test. This script never constructs a policy or VLM.
# It discovers LeRobotV3 roots, classifies each source by robot_type, decodes a
# random eligible frame, and prints the complete supervision contract.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_ROOT="${DATASET_ROOT:-${PROJ_ROOT}/data}"
HUMAN_STATS_DIR="${HUMAN_STATS_DIR:-}"
HUMAN_ACTION_FILE="${HUMAN_ACTION_FILE:-meta/annotations/action.parquet}"
ACTION_MODE="${ACTION_MODE:-abs}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
NUM_FUTURE_FRAMES="${NUM_FUTURE_FRAMES:-4}"
SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-1}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
SEED="${SEED:-42}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: DATASET_ROOT does not exist: ${DATASET_ROOT}" >&2
  exit 1
fi

ARGS=(
  --dataset-root "${DATASET_ROOT}"
  --human-action-file "${HUMAN_ACTION_FILE}"
  --action-mode "${ACTION_MODE}"
  --chunk-size "${CHUNK_SIZE}"
  --num-future-frames "${NUM_FUTURE_FRAMES}"
  --samples-per-dataset "${SAMPLES_PER_DATASET}"
  --video-backend "${VIDEO_BACKEND}"
  --seed "${SEED}"
)

if [[ -n "${HUMAN_STATS_DIR}" ]]; then
  ARGS+=(--human-stats-dir "${HUMAN_STATS_DIR}")
fi

# Optional space-separated relative dataset roots and action representations.
for repo_id in ${REPO_IDS:-}; do
  ARGS+=(--repo-id "${repo_id}")
done
for embodiment in ${HUMAN_EMBODIMENTS:-}; do
  ARGS+=(--human-embodiment "${embodiment}")
done

python util_scripts/inspect_wsa1_5_dataflow.py "${ARGS[@]}"
