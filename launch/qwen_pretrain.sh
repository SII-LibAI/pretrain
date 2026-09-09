#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

# export HF_HOME=${HF_HOME}

# WANDB_TOKEN=${WANDB_TOKEN}
# CONDA_ROOT=${_CONDA_ROOT}
# CONDA_ENV=internvla_a1_5

# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate pretrain

# wandb login ${WANDB_TOKEN}

###############################################################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-6379}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-2}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

# Uncomment the following NCCL flags when encountering NCCL hangs, silent stalls,
# or unstable behavior in multi-GPU / distributed training
# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_NCCL_BLOCKING_WAIT=1
export HF_LEROBOT_HOME="$PWD/data"

export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export CUDAToolkit_ROOT="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE=offline
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

###############################################################################
############################## TRAINING config ################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"

cd ${PROJ_ROOT}

# 1. policy config
POLICY="wsa_pretrain"
# Official Qwen3.5-2B; A1.5 adds FAST action tokens at runtime.
# The old expanded Qwen3.5-2B-Action path is still compatible.
VLM_MODEL_PATH="${VLM_MODEL_PATH:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/qwen3.5-2b}"
# WAN_MODEL_PATH="${WAN_MODEL_PATH:-${HF_HOME}/hub/Wan2.2-TI2V-5B}"
# VAE_PATH="${VAE_PATH:-${WAN_MODEL_PATH}/Wan2.2_VAE.pth}"



# 2. dataset config

SMOKE_NUM_DATASETS="${SMOKE_NUM_DATASETS:-}"
SHUF_ARGS=()
if [[ -n "$SMOKE_NUM_DATASETS" ]]; then
  SHUF_ARGS=(-n "$SMOKE_NUM_DATASETS")
fi

DATASET_REPO_ID="$(
  {
    find -L data/a1 -type d -name data 2>/dev/null \
      | while read -r d; do
          root="$(dirname "$d")"
          if [[ -d "$root/meta" && -d "$root/videos" ]]; then
            echo "${root#data/}"
          fi
        done
  } | shuf "${SHUF_ARGS[@]}" | tr '\n' ' ' | xargs
)"


# convert to single line
DATASET_REPO_ID=$(echo "$DATASET_REPO_ID" | tr '\n' ' ' | xargs)

ACTION_TYPE=delta  # abs | delta
USE_EXTERNAL_STATS=true
BATCH_SIZE_PER_DEVICE=12
# External VQA data is disabled by default. To mix VQA data, first download the
# M1-style VQA jsonl/image data, set VQA_BASE / VQA_DATASET_REPO_ID, then
# uncomment the vqa_dataset args below and keep policy.enable_vqa_loss=true.
# VQA_BASE="${VQA_BASE:-data/vqa/intervla_m1_pretrain_data}"
# VQA_DATASET_REPO_ID="$(find "${VQA_BASE}" -maxdepth 2 -name "*.jsonl" -printf "%P\n" | tr '\n' ' ' | xargs)"

# 3. output config
BASE_OUTPUT_DIR="outputs/${POLICY}"
DATASET_NAME="a1"
JOB_NAME="$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-${DATASET_NAME}-${ACTION_TYPE}-pretrain"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"

# set config_path if you want to resume training
config_path="xxx/checkpoints/last/pretrained_model/train_config.json"

ARGS=(
    # ---- Accelerate / distributed ----
    --multi_gpu
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}"
    src/lerobot/scripts/lerobot_train.py

    # ---- Output ----
    --output_dir="${OUTPUT_DIR}"
    --num_workers=12
    --job_name="${JOB_NAME}"

    # uncomment `resume` and `config_path` if you want to resume training
    # --resume=true
    # --config_path=${config_path}

    # ---- Policy ----
    --policy.type=${POLICY}
    --policy.vlm_model_name_or_path="${VLM_MODEL_PATH}"
    --policy.repo_id=lerobot_lab/${POLICY}
    --policy.push_to_hub=false
    --policy.gradient_checkpointing=false
    --policy.dtype=bfloat16
    --policy.vlm_lr=1e-5
    --policy.vlm_decay_lr=1e-6
    --policy.vlm_warmup_steps=5000
    --policy.vlm_decay_steps=200000
    --policy.action_lr=1e-4
    --policy.action_decay_lr=1e-5
    --policy.action_warmup_steps=5000
    --policy.action_decay_steps=200000 
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.enable_vqa_loss=false         # Robot text/FAST token loss; no external VQA data by default.
    --policy.tokenize_state=true
    --policy.knowledge_insulation=false
    --policy.video_loss_only=false
    --policy.video_loss_weight=1
    --policy.action_loss_only=true
    --policy.freeze_learnable_tokens=false
    --policy.num_learnable_tokens=50

    # ---- Dataset ----
    --dataset.type="$POLICY"
    --dataset.repo_id="$DATASET_REPO_ID"
    --dataset.image_transforms.enable=false
    --dataset.image_transforms.preset=pi05
    --dataset.action_mode="$ACTION_TYPE"
    --dataset.use_external_stats="$USE_EXTERNAL_STATS"
    --dataset.dist_loading=true
    --dataset.tokenize_state=true
    --dataset.use_fast_action_tokens=false
    --dataset.weight_rules_path=${PROJ_ROOT}/configs/weight_rules_pretrain.yaml # pretrain data mix weight rules
    --dataset.max_prompt_length=650

    # ---- Optional external VQA data ----
    # --vqa_dataset.type="$POLICY"
    # --vqa_dataset.root="$VQA_BASE"
    # --vqa_dataset.repo_id="$VQA_DATASET_REPO_ID"
    # --vqa_dataset.weight=0.15

    # ---- Training ----
    --seed=42
    --batch_size="${BATCH_SIZE_PER_DEVICE}"
    --steps=200000
    --save_freq=20000
    --log_freq=50

    # ---- Logging ----
    --wandb.enable=true
    --wandb.project=${POLICY}
    --wandb.mode="${WANDB_MODE}"
)

accelerate launch "${ARGS[@]}"
