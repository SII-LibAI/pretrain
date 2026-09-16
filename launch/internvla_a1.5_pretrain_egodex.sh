#!/usr/bin/env bash
set -euo pipefail

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-6300}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"

PROC_PER_NODE="${PROC_PER_NODE:-2}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"

POLICY="internvla_a1_5"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-/path/to/qwen3.5-2b}"
DATASET_ROOT="${DATASET_ROOT:-/path/to/Egodex-LerobotV3/extra}"
DATASET_ROOT="${DATASET_ROOT%/}"
STATS_DIR="${STATS_DIR:-${DATASET_ROOT}/synthetic_embodiment_stats}"
EMBODIMENTS="${EMBODIMENTS:-[\"franka\"]}"
MAX_STATE_DIM="${MAX_STATE_DIM:-52}"
MAX_ACTION_DIM="${MAX_ACTION_DIM:-52}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-12}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 1
fi

SMOKE_NUM_DATASETS="${SMOKE_NUM_DATASETS:-}"
SHUF_ARGS=()
if [[ -n "${SMOKE_NUM_DATASETS}" ]]; then
  SHUF_ARGS=(-n "${SMOKE_NUM_DATASETS}")
fi

# Dataset discovery belongs here, not inside SyntheticEmbodimentDataset.
# repo_id values are paths relative to DATASET_ROOT, as expected by the factory.
DATASET_REPO_ID="$(
  find -L "${DATASET_ROOT}" -type d -name data 2>/dev/null \
    | while read -r data_dir; do
        dataset_dir="$(dirname "${data_dir}")"
        if [[ -d "${dataset_dir}/meta" \
              && -d "${dataset_dir}/videos" \
              && -f "${dataset_dir}/meta/annotations/action.parquet" ]]; then
          echo "${dataset_dir#"${DATASET_ROOT}"/}"
        fi
      done \
    | shuf "${SHUF_ARGS[@]}" \
    | tr '\n' ' ' \
    | xargs
)"

if [[ -z "${DATASET_REPO_ID}" ]]; then
  echo "No EgoDex LeRobotV3 datasets found under ${DATASET_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${STATS_DIR}/manifest.json" ]]; then
  echo "Missing normalization manifest: ${STATS_DIR}/manifest.json" >&2
  exit 1
fi

JOB_NAME="$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-egodex-delta-pretrain"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${POLICY}/${JOB_NAME}}"

ARGS=(
  --multi_gpu
  --num_processes="${NUM_PROCESSES}"
  --num_machines="${NODE_COUNT}"
  --machine_rank="${NODE_RANK}"
  --main_process_ip="${MASTER_ADDR}"
  --main_process_port="${MASTER_PORT}"
  src/lerobot/scripts/lerobot_train.py

  --output_dir="${OUTPUT_DIR}"
  --num_workers=8
  --job_name="${JOB_NAME}"

  --policy.type="${POLICY}"
  --policy.vlm_model_name_or_path="${VLM_MODEL_PATH}"
  --policy.repo_id="lerobot_lab/${POLICY}"
  --policy.push_to_hub=false
  --policy.max_state_dim="${MAX_STATE_DIM}"
  --policy.max_action_dim="${MAX_ACTION_DIM}"
  --policy.chunk_size="${CHUNK_SIZE}"
  --policy.n_action_steps="${CHUNK_SIZE}"
  --policy.gradient_checkpointing=false
  --policy.dtype=bfloat16
  --policy.optimizer_lr=5e-5
  --policy.scheduler_warmup_steps=5000
  --policy.scheduler_decay_steps=200000
  --policy.scheduler_decay_lr=5e-5
  --policy.freeze_vision_encoder=false
  --policy.train_expert_only=false
  --policy.enable_vqa_loss=false
  --policy.tokenize_state=true
  --policy.knowledge_insulation=false
  --policy.video_loss_only=false
  --policy.video_loss_weight=1
  --policy.action_loss_only=true
  --policy.freeze_learnable_tokens=false
  --policy.num_learnable_tokens=50

  --dataset.type=synthetic_embodiment
  --dataset.root="${DATASET_ROOT}"
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.embodiments="${EMBODIMENTS}"
  --dataset.stats_dir="${STATS_DIR}"
  --dataset.dist_loading=true
  --dataset.tokenize_state=true
  --dataset.max_prompt_length=650
  --dataset.video_backend=pyav

  --seed=42
  --batch_size="${BATCH_SIZE_PER_DEVICE}"
  --steps=200000
  --save_freq=20000
  --log_freq=100

  --wandb.enable=true
  --wandb.project="${POLICY}"
  --wandb.mode="${WANDB_MODE}"
)

echo "DATASET_ROOT=${DATASET_ROOT}"
echo "DATASET_REPO_ID=${DATASET_REPO_ID}"
echo "EMBODIMENTS=${EMBODIMENTS}"
echo "MAX_STATE_DIM=${MAX_STATE_DIM}, MAX_ACTION_DIM=${MAX_ACTION_DIM}, CHUNK_SIZE=${CHUNK_SIZE}"

accelerate launch "${ARGS[@]}"
