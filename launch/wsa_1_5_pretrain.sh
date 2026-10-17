#!/usr/bin/env bash
set -euo pipefail

# Server entry point; WSA_DRY_RUN=1 only prints the exact command.
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
    DATA_ROOT="${DATA_ROOT:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series/data/ego/Egoverse-Lerobot-v3/shard_000}"
    DATASET_REPO_ID="${DATASET_REPO_ID:-}}"
    DATASET_REPO_ID="$(
    {
        find -L ${DATA_ROOT} -type d -name data 2>/dev/null \
        | while read -r d; do
            root="$(dirname "$d")"
            if [[ -d "$root/meta" && -d "$root/videos" ]]; then
                echo "${root#data/}"
            fi
            done
    } | shuf | tr '\n' ' ' | xargs
    )"
    if [[ -z "${DATASET_REPO_ID}" ]]; then
    echo "No annotated LeRobotV3 datasets found under ${DATASET_ROOT}" >&2
    exit 1
    fi
    WAN_MODEL_PATH="${WAN_MODEL_PATH:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/Wan2.2-TI2V-5B}"
    TRAIN_ARGS=(
        --policy.type=WSA_1_5
        --policy.push_to_hub=false
        --policy.wan_model_path="$WAN_MODEL_PATH"
        --policy.vae_path="${VAE_PATH:-$WAN_MODEL_PATH/Wan2.2_VAE.pth}"
        --policy.text_encoder_path="${TEXT_ENCODER_PATH:-$WAN_MODEL_PATH/models_t5_umt5-xxl-enc-bf16.pth}"
        --policy.tokenizer_path="${TOKENIZER_PATH:-$WAN_MODEL_PATH/google/umt5-xxl}"
        --policy.omega_checkpoint_path="${OMEGA_CHECKPOINT_PATH:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/vggt-omega/vggt_omega_1b_256_text.pt}"
        --policy.enable_3d_queries="${ENABLE_3D_QUERIES:-true}"
        --policy.lambda_rgb="${LAMBDA_RGB:-1.0}"
        --policy.lambda_motion="${LAMBDA_MOTION:-1.0}"
        --policy.lambda_3d="${LAMBDA_3D:-1}"
        --policy.feature_mse_weight="${FEATURE_MSE_WEIGHT:-1.0}"
        --policy.feature_cosine_weight="${FEATURE_COSINE_WEIGHT:-1.0}"
        --policy.dtype=bfloat16
        --policy.load_text_encoder=false
        --policy.gradient_checkpointing=true
        --policy.gradient_accumulation_steps="${GRAD_ACCUM_STEPS:-1}"
        --policy.future_frames="${FUTURE_FRAMES:-16}"
        --policy.rgb_lr="${RGB_LR:-5e-5}"
        --policy.motion_lr="${MOTION_LR:-5e-5}"
        --policy.goal_lr="${GOAL_LR:-5e-5}"
        --policy.heads_lr="${HEADS_LR:-5e-5}"
        --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS:-5000}"
        --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS:-300000}"
        --dataset.type=wsa1_5
        --dataset.future_frames="${FUTURE_FRAMES:-16}"
        --dataset.root="$DATA_ROOT"
        --dataset.repo_id="$DATASET_REPO_ID"
        --dataset.video_backend="${VIDEO_BACKEND:-pyav}"
        --dataset.dist_loading="${DIST_LOADING:-true}"
        --output_dir="${OUTPUT_DIR:-outputs/wsa1_5/$(date +%Y_%m_%d_%H_%M_%S)}"
        --batch_size="${BATCH_SIZE:-8}"
        --num_workers="${NUM_WORKERS:-12}"
        --steps="${STEPS:-300000}"
        --save_freq="${SAVE_FREQ:-20000}"
        --log_freq="${LOG_FREQ:-10}"
        --wandb.enable="${WANDB_ENABLE:-true}"
        --wandb.mode=offline
    )
    [[ -z "${WEIGHT_RULES_PATH:-}" ]] || TRAIN_ARGS+=(--dataset.weight_rules_path="$WEIGHT_RULES_PATH")
fi

PROC_PER_NODE="${PROC_PER_NODE:-2}"
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
