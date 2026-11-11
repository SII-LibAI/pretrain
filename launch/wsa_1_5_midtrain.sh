#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_ROOT"
export PYTHONPATH="$PROJ_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_HOME="$CONDA_PREFIX" CUDA_PATH="$CONDA_PREFIX" CUDAToolkit_ROOT="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_LEROBOT_HOME="$PROJ_ROOT/data"

if [[ -n "${RESUME_CONFIG:-}" ]]; then
    TRAIN_ARGS=(--config_path="$RESUME_CONFIG" --resume=true)
else
    : "${INIT_CHECKPOINT_PATH:?Set INIT_CHECKPOINT_PATH to the Stage-1 pretrained_model directory}"
    : "${RETARGET_STATS_DIR:?Set RETARGET_STATS_DIR to unified72 per-embodiment statistics}"
    DATA_ROOT="${DATA_ROOT:-$PROJ_ROOT/data}"
    DATASET_REPO_ID="${DATASET_REPO_ID:-}"
    if [[ -z "$DATASET_REPO_ID" ]]; then
        DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
        DATASET_REPO_ID="$(
            find -L "$DATA_ROOT" -type d -name data \
            | while IFS= read -r d; do
                root="$(dirname "$d")"
                if [[ -f "$root/meta/info.json" && -d "$root/videos" ]]; then
                    echo "$root"
                fi
            done | shuf | tr '\n' ' ' | xargs
        )"
    fi
    [[ -n "$DATASET_REPO_ID" ]] || { echo "No LeRobot datasets found under $DATA_ROOT" >&2; exit 1; }
    WAN_MODEL_PATH="${WAN_MODEL_PATH:-/inspire/hdd/global_user/zhangjianing-253108140206/dataset/model/wan}"
    TRAIN_ARGS=(
        --policy.type=WSA_1_5 --policy.push_to_hub=false
        --policy.enable_action=true --policy.action_space=unified72
        --policy.state_dim=72 --policy.action_dim=72
        --policy.init_checkpoint_path="$INIT_CHECKPOINT_PATH"
        --policy.load_text_encoder=true
        --policy.text_encoder_path="${TEXT_ENCODER_PATH:-$WAN_MODEL_PATH/models_t5_umt5-xxl-enc-bf16.pth}"
        --policy.tokenizer_path="${TOKENIZER_PATH:-$WAN_MODEL_PATH/google/umt5-xxl}"
        --policy.omega_checkpoint_path="${OMEGA_CHECKPOINT_PATH:-/inspire/hdd/global_user/zhangjianing-253108140206/dataset/model/vggt_omega_1b_256_text.pt}"
        --policy.rgb_finetune_mode="${RGB_FINETUNE_MODE:-full}"
        --policy.action_hidden_dim="${ACTION_HIDDEN_DIM:-1024}" --policy.action_ffn_dim="${ACTION_FFN_DIM:-4096}"
        --policy.action_chunk_size="${ACTION_CHUNK_SIZE:-32}" --policy.action_frame_stride="${ACTION_FRAME_STRIDE:-1}"
        --policy.action_sampler_steps="${ACTION_SAMPLER_STEPS:-10}"
        --policy.num_history_frame="${NUM_HISTORY_FRAME:-4}" --policy.history_frame_stride="${HISTORY_FRAME_STRIDE:-4}"
        --policy.sparse_frame_encoding="${SPARSE_FRAME_ENCODING:-joint}"
        --policy.action_mode="${ACTION_MODE:-delta}"
        --policy.future_frames="${FUTURE_FRAMES:-8}"
        --policy.rgb_height="${RGB_HEIGHT:-384}" --policy.rgb_width="${RGB_WIDTH:-320}"
        --policy.motion_height="${MOTION_HEIGHT:-192}" --policy.motion_width="${MOTION_WIDTH:-160}"
        --policy.lambda_rgb="${LAMBDA_RGB:-1.0}" --policy.lambda_motion="${LAMBDA_MOTION:-0.5}"
        --policy.lambda_3d="${LAMBDA_3D:-0.5}" --policy.lambda_action="${LAMBDA_ACTION:-1.0}"
        --policy.rgb_lr="${RGB_LR:-5e-5}" --policy.motion_lr="${MOTION_LR:-5e-5}"
        --policy.goal_lr="${GOAL_LR:-5e-5}" --policy.heads_lr="${HEADS_LR:-5e-5}" --policy.action_lr="${ACTION_LR:-5e-5}"
        --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS:-2000}"
        --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS:-100000}"
        --policy.scheduler_decay_lr_ratio="${SCHEDULER_DECAY_LR_RATIO:-0.1}"
        --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING:-true}"
        --policy.gradient_accumulation_steps="${GRAD_ACCUM_STEPS:-1}"
        --dataset.type=wsa1_5 --dataset.root="$DATA_ROOT" --dataset.repo_id="$DATASET_REPO_ID"
        --dataset.action_mode="${ACTION_MODE:-delta}"
        --dataset.use_external_stats=true
        --dataset.retarget_stats_dir="$RETARGET_STATS_DIR" --dataset.embodiments="${EMBODIMENTS:-[]}"
        --dataset.annotation_row_group_cache_size="${ANNOTATION_ROW_GROUP_CACHE_SIZE:-2}"
        --dataset.future_frames="${FUTURE_FRAMES:-8}" --dataset.frame_stride="${FRAME_STRIDE:-4}"
        --dataset.rgb_height="${RGB_HEIGHT:-384}" --dataset.rgb_width="${RGB_WIDTH:-320}"
        --dataset.motion_height="${MOTION_HEIGHT:-192}" --dataset.motion_width="${MOTION_WIDTH:-160}"
        --dataset.use_subtask="${USE_SUBTASK:-false}" --dataset.allow_endframe_3d="${ALLOW_ENDFRAME_3D:-true}"
        --dataset.video_backend="${VIDEO_BACKEND:-pyav}" --dataset.video_decode_threads="${VIDEO_DECODE_THREADS:-1}"
        --dataset.dist_loading="${DIST_LOADING:-false}"
        --output_dir="${OUTPUT_DIR:-outputs/wsa1_5_midtrain/$(date +%Y_%m_%d_%H_%M_%S)}"
        --steps="${STEPS:-100000}" --batch_size="${BATCH_SIZE:-8}" --num_workers="${NUM_WORKERS:-8}"
        --log_freq="${LOG_FREQ:-20}" --save_freq="${SAVE_FREQ:-10000}" --eval_freq="${EVAL_FREQ:-5000}"
        --wandb.enable="${WANDB_ENABLE:-true}" --wandb.mode="${WANDB_MODE:-offline}"
    )
    [[ -z "${EXTERNAL_STATS_PATH:-}" ]] || TRAIN_ARGS+=(--dataset.external_stats_path="$EXTERNAL_STATS_PATH")
    [[ -z "${WEIGHT_RULES_PATH:-}" ]] || TRAIN_ARGS+=(--dataset.weight_rules_path="$WEIGHT_RULES_PATH")
fi
NODE_COUNT="${PET_NNODES:-${NODE_COUNT:-1}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
PROC_PER_NODE="${PET_NPROC_PER_NODE:-${PROC_PER_NODE:-8}}"
ACCEL_ARGS=(--num_machines="$NODE_COUNT" --num_processes="$((NODE_COUNT * PROC_PER_NODE))"
            --machine_rank="$NODE_RANK" --mixed_precision=bf16)
if (( NODE_COUNT * PROC_PER_NODE > 1 )); then ACCEL_ARGS+=(--multi_gpu); fi
if (( NODE_COUNT > 1 )); then
    : "${PET_MASTER_ADDR:=${MASTER_ADDR:?Set MASTER_ADDR for multiple nodes}}"
    ACCEL_ARGS+=(--main_process_ip="$PET_MASTER_ADDR" --main_process_port="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}" --same_network)
fi
COMMAND=(accelerate launch "${ACCEL_ARGS[@]}" src/lerobot/scripts/lerobot_train.py "${TRAIN_ARGS[@]}" "$@")
if [[ "${WSA_DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; else exec "${COMMAND[@]}"; fi
