#!/usr/bin/env bash
set -euo pipefail
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_ROOT"
export PYTHONPATH="$PROJ_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ -n "${RESUME_CONFIG:-}" ]]; then
    TRAIN_ARGS=(--config_path="$RESUME_CONFIG" --resume=true)
else
    : "${DATA_ROOT:?Set DATA_ROOT to the LeRobot collection directory}"
    : "${INIT_CHECKPOINT_PATH:?Set INIT_CHECKPOINT_PATH to Stage-1 pretrained_model directory}"
    REPOS="${DATASET_REPO_ID:-}"
    if [[ -z "$REPOS" ]]; then
        REPOS="$(bash util_scripts/discover_wsa1_5_datasets.sh "$DATA_ROOT")"
    fi
    [[ -n "$REPOS" ]] || { echo "No LeRobot datasets found under $DATA_ROOT" >&2; exit 1; }
    TRAIN_ARGS=(
        --config_path=configs/wsa1_5_action_posttrain.json
        --policy.init_checkpoint_path="$INIT_CHECKPOINT_PATH"
        --policy.rgb_finetune_mode="${RGB_FINETUNE_MODE:-lora}"
        --policy.state_dim="${STATE_DIM:-16}"
        --policy.action_dim="${ACTION_DIM:-16}"
        --policy.action_chunk_size="${ACTION_CHUNK_SIZE:-32}"
        --policy.action_frame_stride="${ACTION_FRAME_STRIDE:-1}"
        --policy.action_sampler_steps="${ACTION_SAMPLER_STEPS:-10}"
        --policy.num_history_frame="${NUM_HISTORY_FRAME:-4}"
        --policy.history_frame_stride="${HISTORY_FRAME_STRIDE:-2}"
        --policy.sparse_frame_encoding="${SPARSE_FRAME_ENCODING:-joint}"
        --policy.action_mode="${ACTION_MODE:-delta}"
        --policy.lambda_rgb="${LAMBDA_RGB:-1.0}"
        --policy.lambda_motion="${LAMBDA_MOTION:-1.0}"
        --policy.lambda_3d="${LAMBDA_3D:-0.01}"
        --policy.lambda_action="${LAMBDA_ACTION:-1.0}"
        --policy.action_lr="${ACTION_LR:-1e-4}"
        --policy.omega_checkpoint_path="${OMEGA_CHECKPOINT_PATH:-}"
        --policy.text_encoder_path="${TEXT_ENCODER_PATH:-}"
        --policy.tokenizer_path="${TOKENIZER_PATH:-}"
        --policy.load_text_encoder="${LOAD_TEXT_ENCODER:-false}"
        --policy.gradient_accumulation_steps="${GRAD_ACCUM_STEPS:-1}"
        --dataset.root="$DATA_ROOT"
        --dataset.repo_id="$REPOS"
        --dataset.action_mode="${ACTION_MODE:-delta}"
        --dataset.use_external_stats=true
        --dataset.use_subtask="${USE_SUBTASK:-false}"
        --dataset.allow_endframe_3d="${ALLOW_ENDFRAME_3D:-true}"
        --dataset.image_augment="${IMAGE_AUGMENT:-false}"
        --dataset.object_paste="${OBJECT_PASTE:-false}"
        --dataset.image_augmentation_config="${IMAGE_AUGMENTATION_CONFIG:-configs/augmentation/robotwin_conservative.yaml}"
        --dataset.dist_loading="${DIST_LOADING:-false}"
        --output_dir="${OUTPUT_DIR:-outputs/wsa1_5_action/$(date +%Y_%m_%d_%H_%M_%S)}"
        --batch_size="${BATCH_SIZE:-1}"
        --num_workers="${NUM_WORKERS:-4}"
        --steps="${STEPS:-100000}"
        --wandb.enable="${WANDB_ENABLE:-false}"
    )
    [[ -z "${EXTERNAL_STATS_PATH:-}" ]] || TRAIN_ARGS+=(--dataset.external_stats_path="$EXTERNAL_STATS_PATH")
fi
NODE_COUNT="${PET_NNODES:-${NODE_COUNT:-1}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
PROC_PER_NODE="${PET_NPROC_PER_NODE:-${PROC_PER_NODE:-1}}"
ACCEL_ARGS=(--num_machines="$NODE_COUNT" --num_processes="$((NODE_COUNT * PROC_PER_NODE))"
            --machine_rank="$NODE_RANK" --mixed_precision=bf16)
if (( NODE_COUNT * PROC_PER_NODE > 1 )); then
    ACCEL_ARGS+=(--multi_gpu)
fi
if (( NODE_COUNT > 1 )); then
    : "${PET_MASTER_ADDR:=${MASTER_ADDR:?Set MASTER_ADDR for multiple nodes}}"
    : "${PET_MASTER_PORT:=${MASTER_PORT:-29500}}"
    ACCEL_ARGS+=(--main_process_ip="$PET_MASTER_ADDR" --main_process_port="$PET_MASTER_PORT" --same_network)
fi
COMMAND=(accelerate launch "${ACCEL_ARGS[@]}" src/lerobot/scripts/lerobot_train.py "${TRAIN_ARGS[@]}" "$@")
if [[ "${WSA_DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
else
    exec "${COMMAND[@]}"
fi
