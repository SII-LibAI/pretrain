#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_ROOT"
export PYTHONPATH="$PROJ_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export CUDAToolkit_ROOT="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export HF_LEROBOT_HOME="$PWD/data"

if [[ -n "${RESUME_CONFIG:-}" ]]; then
    TRAIN_ARGS=(--config_path="$RESUME_CONFIG" --resume=true)
else
    DATA_ROOT=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series/data/RoboTwin
    INIT_CHECKPOINT_PATH=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series/outputs/wsa1_5/pretrain-h200-gf-bs64/checkpoints/520000/pretrained_model
    DATASET_REPO_ID="${DATASET_REPO_ID:-}"
    # 只发现 Clean 数据集
    DATASET_REPO_ID="$(
        find -L "${DATA_ROOT}" \
            -mindepth 2 -maxdepth 2 \
            -type d \
            -name "aloha-agilex_clean_*" \
            | while IFS= read -r d; do
                if [[ -d "$d/meta" && -d "$d/videos" ]]; then
                    echo "$d"
                fi
            done \
            | shuf \
            | tr '\n' ' ' \
            | xargs
    )"

    echo "DATASET_REPO_ID=${DATASET_REPO_ID}"
    if [[ -z "${DATASET_REPO_ID}" ]]; then
        echo "No annotated LeRobotV3 datasets found under ${DATA_ROOT}" >&2
        exit 1
    fi
    TRAIN_ARGS=(
        --policy.type=WSA_1_5
        --policy.push_to_hub=false
        --policy.enable_action=true
        --policy.init_checkpoint_path="$INIT_CHECKPOINT_PATH"
        --policy.future_frames="${FUTURE_FRAMES:-8}"
        --policy.enable_rgb="${ENABLE_RGB:-true}"
        --policy.enable_motion="${ENABLE_MOTION:-true}"
        --policy.enable_3d_queries="${ENABLE_3D_QUERIES:-true}"
        --policy.state_dim="${STATE_DIM:-16}"
        --policy.action_dim="${ACTION_DIM:-16}"
        --policy.action_hidden_dim="${ACTION_HIDDEN_DIM:-1024}"
        --policy.action_ffn_dim="${ACTION_FFN_DIM:-4096}"
        --policy.action_chunk_size="${ACTION_CHUNK_SIZE:-32}"
        --policy.action_frame_stride="${ACTION_FRAME_STRIDE:-1}"
        --policy.action_sampler_steps="${ACTION_SAMPLER_STEPS:-10}"
        --policy.num_history_frame="${NUM_HISTORY_FRAME:-4}"
        --policy.history_frame_stride="${HISTORY_FRAME_STRIDE:-4}"
        --policy.sparse_frame_encoding="${SPARSE_FRAME_ENCODING:-joint}"
        --policy.action_mode="${ACTION_MODE:-delta}"
        --policy.rgb_finetune_mode="${RGB_FINETUNE_MODE:-full}"
        --policy.lambda_rgb="${LAMBDA_RGB:-0.1}"
        --policy.lambda_motion="${LAMBDA_MOTION:-0.1}"
        --policy.lambda_3d="${LAMBDA_3D:-0.1}"
        --policy.lambda_action="${LAMBDA_ACTION:-1.0}"
        --policy.rgb_lr="${RGB_LR:-5e-5}"
        --policy.motion_lr="${MOTION_LR:-5e-5}"
        --policy.goal_lr="${GOAL_LR:-5e-5}"
        --policy.heads_lr="${HEADS_LR:-5e-5}"
        --policy.action_lr="${ACTION_LR:-5e-5}"
        --policy.optimizer_weight_decay="${WEIGHT_DECAY:-0.01}"
        --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS:-400}"
        --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS:-100000}"
        --policy.scheduler_decay_lr_ratio="${SCHEDULER_DECAY_LR_RATIO:-0.1}"
        --policy.omega_checkpoint_path="${OMEGA_CHECKPOINT_PATH:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/vggt-omega/vggt_omega_1b_256_text.pt}"
        --policy.text_encoder_path="${TEXT_ENCODER_PATH:-}"
        --policy.tokenizer_path="${TOKENIZER_PATH:-}"
        --policy.load_text_encoder="${LOAD_TEXT_ENCODER:-false}"
        --policy.trainable_weights_fp32="${TRAINABLE_WEIGHTS_FP32:-false}"
        --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING:-true}"
        --policy.gradient_accumulation_steps="${GRAD_ACCUM_STEPS:-1}"
        --dataset.type=wsa1_5
        --dataset.root="$DATA_ROOT"
        --dataset.repo_id="$DATASET_REPO_ID"
        --dataset.future_frames="${FUTURE_FRAMES:-8}"
        --dataset.frame_stride="${FRAME_STRIDE:-4}"
        --dataset.rgb_height="${RGB_HEIGHT:-384}"
        --dataset.rgb_width="${RGB_WIDTH:-320}"
        --dataset.motion_height="${MOTION_HEIGHT:-192}"
        --dataset.motion_width="${MOTION_WIDTH:-160}"
        --dataset.video_backend="${VIDEO_BACKEND:-pyav}"
        --dataset.action_mode="${ACTION_MODE:-delta}"
        --dataset.use_external_stats=true
        --dataset.use_subtask="${USE_SUBTASK:-false}"
        --dataset.allow_endframe_3d="${ALLOW_ENDFRAME_3D:-true}"
        --dataset.image_augment="${IMAGE_AUGMENT:-false}"
        --dataset.object_paste="${OBJECT_PASTE:-false}"
        --dataset.external_stats_path="${EXTERNAL_STATS_PATH:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/WSA-p/outputs/robotwin_clean/delta_32/stats.json}"
        --dataset.image_augmentation_config="${IMAGE_AUGMENTATION_CONFIG:-configs/augmentation/robotwin_conservative.yaml}"
        --dataset.dist_loading="${DIST_LOADING:-false}"
        --output_dir="${OUTPUT_DIR:-outputs/wsa1_5_robotwin/$(date +%Y_%m_%d_%H_%M_%S)}"
        --batch_size="${BATCH_SIZE:-32}"
        --num_workers="${NUM_WORKERS:-8}"
        --steps="${STEPS:-10000}"
        --seed="${SEED:-4000}"
        --log_freq="${LOG_FREQ:-20}"
        --save_freq="${SAVE_FREQ:-5000}"
        --eval_freq="${EVAL_FREQ:-5000}"
        --wandb.enable="${WANDB_ENABLE:-true}"
        --wandb.mode="${WANDB_MODE:-offline}"
    )
fi
NODE_COUNT="${PET_NNODES:-${NODE_COUNT:-1}}"
NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-0}}"
PROC_PER_NODE="${PET_NPROC_PER_NODE:-${PROC_PER_NODE:-8}}"
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