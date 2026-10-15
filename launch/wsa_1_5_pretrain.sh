#!/usr/bin/env bash
set -euo pipefail

# Server entry point; WSA_DRY_RUN=1 only prints the exact command.
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_ROOT"
export PYTHONPATH="$PROJ_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ -n "${RESUME_CONFIG:-}" ]]; then
    TRAIN_ARGS=(--config_path="$RESUME_CONFIG" --resume=true)
else
    : "${DATA_ROOT:?Set the dataset collection root separately from model/output directories}"
    : "${WAN_MODEL_PATH:?Set the local Wan2.2-TI2V-5B directory}"
    DATASET_REPO_IDS="${DATASET_REPO_IDS:-$(python util_scripts/discover_wsa1_5_datasets.py --root "$DATA_ROOT")}"
    TRAIN_ARGS=(
        --policy.type=WSA_1_5
        --policy.push_to_hub=false
        --policy.wan_model_path="$WAN_MODEL_PATH"
        --policy.vae_path="${VAE_PATH:-$WAN_MODEL_PATH/Wan2.2_VAE.pth}"
        --policy.text_encoder_path="${TEXT_ENCODER_PATH:-$WAN_MODEL_PATH/models_t5_umt5-xxl-enc-bf16.pth}"
        --policy.tokenizer_path="${TOKENIZER_PATH:-$WAN_MODEL_PATH/google/umt5-xxl}"
        --policy.omega_checkpoint_path="${OMEGA_CHECKPOINT_PATH:-}"
        --policy.enable_3d_queries="${ENABLE_3D_QUERIES:-true}"
        --policy.enable_3d_supervision="${ENABLE_3D_SUPERVISION:-true}"
        --policy.lambda_3d="${LAMBDA_3D:-1.0}"
        --policy.dtype=bfloat16
        --policy.gradient_checkpointing=true
        --policy.gradient_accumulation_steps="${GRAD_ACCUM_STEPS:-1}"
        --policy.rgb_lr="${RGB_LR:-1e-5}"
        --policy.motion_lr="${MOTION_LR:-5e-5}"
        --policy.goal_lr="${GOAL_LR:-5e-5}"
        --policy.heads_lr="${HEADS_LR:-1e-4}"
        --dataset.type=wsa1_5
        --dataset.root="$DATA_ROOT"
        --dataset.repo_id="$DATASET_REPO_IDS"
        --dataset.video_backend="${VIDEO_BACKEND:-pyav}"
        --dataset.streaming=false
        --dataset.dist_loading="${DIST_LOADING:-false}"
        --output_dir="${OUTPUT_DIR:-outputs/wsa1_5/$(date +%Y_%m_%d_%H_%M_%S)}"
        --batch_size="${BATCH_SIZE:-1}"
        --num_workers="${NUM_WORKERS:-4}"
        --steps="${STEPS:-600000}"
        --save_freq="${SAVE_FREQ:-10000}"
        --log_freq="${LOG_FREQ:-10}"
        --wandb.enable="${WANDB_ENABLE:-false}"
    )
    [[ -z "${WSA_CACHE_DIR:-}" ]] || TRAIN_ARGS+=(--policy.cache_dir="$WSA_CACHE_DIR")
    [[ -z "${WEIGHT_RULES_PATH:-}" ]] || TRAIN_ARGS+=(--dataset.weight_rules_path="$WEIGHT_RULES_PATH")
fi

PROC_PER_NODE="${PROC_PER_NODE:-1}"
NODE_COUNT="${NODE_COUNT:-1}"
NUM_PROCESSES=$((PROC_PER_NODE * NODE_COUNT))
ACCEL_ARGS=(--num_processes="$NUM_PROCESSES" --num_machines="$NODE_COUNT"
    --machine_rank="${NODE_RANK:-0}" --main_process_ip="${MASTER_ADDR:-127.0.0.1}"
    --main_process_port="${MASTER_PORT:-29500}" --mixed_precision=bf16)
if ((NUM_PROCESSES > 1)); then ACCEL_ARGS+=(--multi_gpu); fi
COMMAND=(accelerate launch "${ACCEL_ARGS[@]}" src/lerobot/scripts/lerobot_train.py "${TRAIN_ARGS[@]}" "$@")
if [[ "${WSA_DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
else
    exec "${COMMAND[@]}"
fi
