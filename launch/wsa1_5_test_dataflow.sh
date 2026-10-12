#!/usr/bin/env bash
set -euo pipefail

# Data-only WSA1.5 smoke test. This script never constructs a policy or VLM.
# It discovers LeRobotV3 roots, classifies each source by robot_type, decodes a
# random eligible frame, and prints the complete supervision contract.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Record startup configuration, inspection results, and errors in one report.
# Append when OUTPUT_TXT names an existing file; keep real-time console output.
OUTPUT_TXT="${OUTPUT_TXT:-${PROJ_ROOT}/outputs/wsa1_5/dataflow/dataflow_$(date +'%Y%m%d_%H%M%S')_$$.txt}"
mkdir -p "$(dirname "${OUTPUT_TXT}")"
exec > >(tee -a "${OUTPUT_TXT}") 2>&1
echo "OUTPUT_TXT=${OUTPUT_TXT}"

DATASET_ROOT="${DATASET_ROOT:-${PROJ_ROOT}/data}"
HUMAN_STATS_DIR="${HUMAN_STATS_DIR:-}"
HUMAN_ACTION_FILE="${HUMAN_ACTION_FILE:-meta/annotations/action.parquet}"
HUMAN_EMBODIMENTS="${HUMAN_EMBODIMENTS:-human_eef}"
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
DATASET_ROOT="$(cd "${DATASET_ROOT}" && pwd)"

# Match the training launch flow: discover datasets in the parent directory,
# then pass repo_id paths relative to DATASET_ROOT to the inspection tool.
DISCOVERED_REPO_IDS=()
while IFS= read -r -d '' data_dir; do
  dataset_dir="${data_dir%/data}"
  if [[ -f "${dataset_dir}/meta/info.json" ]]; then
    DISCOVERED_REPO_IDS+=("${dataset_dir#"${DATASET_ROOT}/"}")
  fi
done < <(find -L "${DATASET_ROOT}" -type d -name data -print0)

if (( ${#DISCOVERED_REPO_IDS[@]} == 0 )); then
  echo "ERROR: no LeRobotV3 datasets found under ${DATASET_ROOT}" >&2
  exit 1
fi

# REPO_IDS optionally selects a subset of the discovered relative paths.
SELECTED_REPO_IDS=("${DISCOVERED_REPO_IDS[@]}")
if [[ -n "${REPO_IDS:-}" ]]; then
  read -r -a SELECTED_REPO_IDS <<< "${REPO_IDS}"
  for repo_id in "${SELECTED_REPO_IDS[@]}"; do
    found=false
    for discovered_id in "${DISCOVERED_REPO_IDS[@]}"; do
      if [[ "${repo_id}" == "${discovered_id}" ]]; then
        found=true
        break
      fi
    done
    if [[ "${found}" == false ]]; then
      echo "ERROR: REPO_IDS entry was not discovered under DATASET_ROOT: ${repo_id}" >&2
      exit 1
    fi
  done
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

# Explicit repo_id arguments: the tool itself does not discover datasets.
for repo_id in "${SELECTED_REPO_IDS[@]}"; do
  ARGS+=(--repo-id "${repo_id}")
done
read -r -a SELECTED_EMBODIMENTS <<< "${HUMAN_EMBODIMENTS}"
for embodiment in "${SELECTED_EMBODIMENTS[@]}"; do
  ARGS+=(--human-embodiment "${embodiment}")
done

echo "DATASET_ROOT=${DATASET_ROOT}"
printf 'REPO_ID=%s\n' "${SELECTED_REPO_IDS[@]}"
echo "HUMAN_EMBODIMENTS=${HUMAN_EMBODIMENTS}"

python -u util_scripts/inspect_wsa1_5_dataflow.py "${ARGS[@]}"
