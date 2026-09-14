#!/usr/bin/env bash

###############################################################################
################################# ENV config ##################################

# CONDA_ROOT=${_CONDA_ROOT}
# CONDA_ENV=internvla_a1_5

# source ${CONDA_ROOT}/etc/profile.d/conda.sh
# conda activate ${CONDA_ENV}
export HF_HOME=${HF_HOME}
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export CUDAToolkit_ROOT="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"
PRETRAINED_CKPT="${1:-${PRETRAINED_CKPT:-InternRobotics/InternVLA-A1.5-RoboTwin}}"
TASK_CONFIG="${3:-${TASK_CONFIG:-demo_clean}}"
TASK_IDX="${4:-${TASK_IDX:-44}}"
OUTPUT_PATH="${2:-${OUTPUT_PATH:-outputs/robotwin/internvla_a1_5/${TASK_CONFIG}/${TASK_IDX}}}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
ACTION_MODE="${5:-${ACTION_MODE:-abs}}"
INFER_HORIZON="${6:-${INFER_HORIZON:-20}}"
NUM_EPISODES="${7:-${NUM_EPISODES:-100}}"
DTYPE="${8:-${DTYPE:-float32}}"

if [[ -z "${PRETRAINED_CKPT}" ]]; then
  echo "Usage: bash evaluation/RoboTwin/eval.sh <checkpoint> [output_path] [task_config] [task_idx] [action_mode] [infer_horizon] [num_episodes] [dtype]" >&2
  exit 2
fi

if [[ -d "${PRETRAINED_CKPT}" ]]; then
  PRETRAINED_CKPT="$(cd "${PRETRAINED_CKPT}" && pwd)"
fi

mkdir -p "${OUTPUT_PATH}"
OUTPUT_PATH="$(cd "${OUTPUT_PATH}" && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/RoboTwin:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd ${REPO_ROOT}/third_party/RoboTwin

python ../../evaluation/RoboTwin/inference.py \
  --ckpt-path "${PRETRAINED_CKPT}" \
  --video-dir "${OUTPUT_PATH}" \
  --task-config "${TASK_CONFIG}" \
  --task-idx "${TASK_IDX}" \
  --resize-size "${RESIZE_SIZE}" \
  --action-mode "${ACTION_MODE}" \
  --infer-horizon "${INFER_HORIZON}" \
  --num-episodes "${NUM_EPISODES}" \
  --dtype "${DTYPE}" \
  --inference-backend "${INFERENCE_BACKEND}"
