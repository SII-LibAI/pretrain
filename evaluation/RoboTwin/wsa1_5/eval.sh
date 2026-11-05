#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export CUDA_HOME="$CONDA_PREFIX"
  export CUDA_PATH="$CONDA_PREFIX"
  export CUDAToolkit_ROOT="$CONDA_PREFIX"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

PRETRAINED_CKPT="${1:-${PRETRAINED_CKPT:-}}"
TASK_CONFIG="${3:-${TASK_CONFIG:-demo_clean}}"
TASK_IDX="${4:-${TASK_IDX:-0}}"
OUTPUT_PATH="${2:-${OUTPUT_PATH:-outputs/robotwin/wsa1_5/${TASK_CONFIG}/${TASK_IDX}}}"
ACTION_MODE="${5:-${ACTION_MODE:-auto}}"
INFER_HORIZON="${6:-${INFER_HORIZON:-0}}"
NUM_EPISODES="${7:-${NUM_EPISODES:-100}}"
DTYPE="${8:-${DTYPE:-auto}}"
TEXT_ENCODER_PATH="${9:-${TEXT_ENCODER_PATH:-}}"
TOKENIZER_PATH="${10:-${TOKENIZER_PATH:-}}"
if [[ ! "$NUM_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_EPISODES must be a positive integer: $NUM_EPISODES" >&2
  exit 2
fi
if [[ ! -f "$PRETRAINED_CKPT/config.json" || ! -f "$PRETRAINED_CKPT/model.safetensors" ]]; then
  echo "Usage: bash evaluation/RoboTwin/wsa1_5/eval.sh CHECKPOINT [OUTPUT] [TASK_CONFIG] [TASK_IDX] [ACTION_MODE] [HORIZON] [EPISODES] [DTYPE] [TEXT_ENCODER_PATH] [TOKENIZER_PATH]" >&2
  echo "Provide a local WSA1.5 Action checkpoint with config.json and model.safetensors" >&2
  exit 2
fi
PRETRAINED_CKPT="$(cd "$PRETRAINED_CKPT" && pwd)"
mkdir -p "$OUTPUT_PATH"
OUTPUT_PATH="$(cd "$OUTPUT_PATH" && pwd)"
completed=0
for ((episode_idx = 1; episode_idx <= NUM_EPISODES; episode_idx++)); do
  if [[ -s "$OUTPUT_PATH/success_${episode_idx}.mp4" || -s "$OUTPUT_PATH/failure_${episode_idx}.mp4" ]]; then
    completed=$((completed + 1))
  fi
done
if (( completed == NUM_EPISODES )); then
  echo "SKIP task=$TASK_IDX status=complete ($completed/$NUM_EPISODES) output=$OUTPUT_PATH"
  exit 0
fi
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/RoboTwin:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

COMMAND=(python -u "${SCRIPT_DIR}/inference.py"
  --ckpt-path "$PRETRAINED_CKPT" --video-dir "$OUTPUT_PATH"
  --task-config "$TASK_CONFIG" --task-idx "$TASK_IDX"
  --action-mode "$ACTION_MODE" --infer-horizon "$INFER_HORIZON"
  --num-episodes "$NUM_EPISODES" --dtype "$DTYPE"
  --robot-type "${ROBOT_TYPE:-aloha}" --device "${DEVICE:-cuda}"
  --text-device "${TEXT_DEVICE:-cpu}" --instruction-type "${INSTRUCTION_TYPE:-unseen}"
  --seed "${SEED:-42}" --fps "${FPS:-30}" --log-level "${LOG_LEVEL:-INFO}")
STATS_PATH="${STATS_PATH:-${EXTERNAL_STATS_PATH:-}}"
[[ -z "${ACTION_SAMPLER_STEPS:-}" ]] || COMMAND+=(--action-sampler-steps "$ACTION_SAMPLER_STEPS")
[[ -z "$STATS_PATH" ]] || COMMAND+=(--stats-path "$(realpath "$STATS_PATH")")
[[ -z "${TEXT_CACHE:-}" ]] || COMMAND+=(--text-cache "$(realpath "$TEXT_CACHE")")
[[ -z "${TEXT_ENCODER_PATH:-}" ]] || COMMAND+=(--text-encoder-path "$(realpath "$TEXT_ENCODER_PATH")")
[[ -z "${TOKENIZER_PATH:-}" ]] || COMMAND+=(--tokenizer-path "$(realpath "$TOKENIZER_PATH")")
if [[ "${ROBOTWIN_DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
else
  cd "${REPO_ROOT}/third_party/RoboTwin"
  shopt -s nullglob
  previous_results=()
  for result in "$OUTPUT_PATH"/success_*.mp4 "$OUTPUT_PATH"/failure_*.mp4 "$OUTPUT_PATH"/.complete_*; do
    [[ ! -f "$result" ]] || previous_results+=("$result")
  done
  if (( ${#previous_results[@]} )); then
    mkdir -p "$OUTPUT_PATH/previous_results"
    backup="$(mktemp -d "$OUTPUT_PATH/previous_results/run.XXXXXX")"
    mv -- "${previous_results[@]}" "$backup/"
    echo "Archived previous results to $backup; restarting task from episode 1"
  fi
  exec "${COMMAND[@]}"
fi
