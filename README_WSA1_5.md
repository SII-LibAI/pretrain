# WSA1.5 使用说明

当前 `WSA_1_5` policy 输入当前 RGB 和文本，预测未来 32 帧 RGB、未来 32 帧三通道 motion，以及可选的子任务终点 3D 特征；不预测 action。

以下安装、下载和训练命令在 **Linux GPU 服务器**执行，统一从本仓库根目录运行。本地开发只做代码、数据及小型随机模块检查。

## 1. 安装环境

示例环境为 Linux x86_64、Python 3.11、支持 BF16 的 NVIDIA GPU。先把当前含 WSA1.5 代码的 checkout 同步到服务器，并准备匹配 CUDA 的驱动。

```bash
conda create -n wsa15 python=3.11 -y
conda activate wsa15
cd /path/to/InternVLA-A-series

# Ubuntu；已有 Git 和 FFmpeg 时跳过。
sudo apt-get update && sudo apt-get install -y git ffmpeg
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple
python -m pip install -c configs/wsa1_5_constraints.txt \
  -e '.[wsa1_5,wsa1_5_download]'

# 安装已核对版本的 Omega 实现；不下载模型权重。
python -m pip install --no-deps \
  'vggt-omega @ git+https://github.com/facebookresearch/vggt-omega.git@48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6'
```

[版本约束](configs/wsa1_5_constraints.txt) 固定示例的 torch、torchvision、torchcodec、NumPy、OpenCV 和 transformers。换 CUDA/PyTorch 组合时须同步匹配这些依赖。Omega 复用主工程的 `opencv-python-headless`，不要重复安装另一份 OpenCV。当前使用 PyTorch SDPA，不要求 flash-attn。

只检查 GPU，不创建模型：

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA 不可用'
assert torch.cuda.is_bf16_supported(), '默认启动脚本需要 BF16'
print(torch.__version__, torch.cuda.get_device_name(0))
PY
```

默认使用 FP32 可训练参数、BF16 混合精度和 AdamW，另有冻结的 VAE、UMT5、Omega。完整配置的显存和主存开销很大，应先在服务器测量。DDP 每卡保留完整参数与 optimizer 状态，增加 GPU 不会自动降低这部分单卡占用。

## 2. 下载权重

内置 [ModelScope 下载脚本](util_scripts/download_wsa1_5_weights.py)：

| 组件 | ModelScope 仓库 | 下载内容 |
| --- | --- | --- |
| Wan | `Wan-AI/Wan2.2-TI2V-5B` | 原始 DiT 三个分片及索引、Wan2.2 VAE、UMT5 encoder 和 tokenizer |
| Omega | `facebook/VGGT-Omega` | 仅 `vggt_omega_1b_256_text.pt`，即 256 Text-Alignment |

```bash
# 预览：不联网、不写文件。
python util_scripts/download_wsa1_5_weights.py --output-root weights --dry-run

# 在服务器下载并校验；默认全部组件合计约 40 GB。
python util_scripts/download_wsa1_5_weights.py --output-root weights

# 可选的离线完整性复核，不修改文件。
python util_scripts/download_wsa1_5_weights.py --output-root weights --verify-only
source weights/wsa1_5_weights.env
```

脚本按固定大小和 SHA256 校验，只向 SDK 请求缺失文件。已有完整文件会复用，损坏文件直接报错且不覆盖。中断可重试；确认目标文件损坏后，先将它移出目录再运行。认证需要时设置已授权账号的 `MODELSCOPE_API_TOKEN`，不要写入仓库。

```text
weights/
├── Wan2.2-TI2V-5B/
│   ├── diffusion_pytorch_model-00001-of-00003.safetensors
│   ├── diffusion_pytorch_model-00002-of-00003.safetensors
│   ├── diffusion_pytorch_model-00003-of-00003.safetensors
│   ├── diffusion_pytorch_model.safetensors.index.json
│   ├── Wan2.2_VAE.pth
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   └── google/umt5-xxl/                 # tokenizer 文件
├── VGGT-Omega-1B-256-Text-Alignment/
│   └── vggt_omega_1b_256_text.pt
├── wsa1_5_weights.env
└── wsa1_5_weights_manifest.json
```

环境文件导出 `WAN_MODEL_PATH`、`VAE_PATH`、`TEXT_ENCODER_PATH`、`TOKENIZER_PATH`、`OMEGA_CHECKPOINT_PATH`；清单记录仓库、请求 revision、大小及 SHA256。上游 master 变化后仍须匹配固定校验，不能替换成 Diffusers/GGUF/FP8 Wan 或 Omega 512/416/MLX 权重。

无 3D 监督的消融可以省去 Omega：

```bash
python util_scripts/download_wsa1_5_weights.py --components wan --output-root weights
source weights/wsa1_5_weights.env
export LAMBDA_3D=0
```

也支持 `--components omega` 单独下载。环境文件只导出本次选中的组件；磁盘还需为下载缓存、数据和完整训练 checkpoint 预留空间。

## 3. 数据集准备

使用本仓库能读取的 **LeRobot V3** 数据，保留原始元数据和视频路径。`DATA_ROOT` 是合集目录，`repo_id` 是相对路径，两者拼接后才是单个数据集目录：

```text
/data/Egodex-LerobotV3/                   # DATA_ROOT
└── extra/knit_scarf/                    # repo_id = extra/knit_scarf
    ├── meta/
    │   ├── info.json
    │   ├── stats.json
    │   ├── tasks.parquet
    │   ├── episodes/chunk-000/file-000.parquet
    │   └── sidecar.json
    ├── data/chunk-000/file-000.parquet
    └── videos/                         # 位置以 info/episode 元数据为准
        ├── observation.image/chunk-000/file-000.mp4
        └── observation.motion/chunk-000/file-000.mp4
```

### 字段与视角

- 帧表含整数 `episode_index`、`frame_index` 和有限 `timestamp`。每个 episode 行连续，frame_index 从 0 递增，时间戳严格递增；不接受小数索引取整。
- RGB/motion 在 `info.json.features` 中为 `dtype=video`、shape=`[H,W,3]`；实际视频、帧数、FPS 和 episode offset 必须一致。
- 至少一个有效 RGB 相机，默认 motion key 为 `observation.motion`；不要求 robot action/state。
- 默认按已注册 robot_type 的 schema 选择 image0/image1/image2，对应上方、左下、右下。RGB 拼接为 `384×320`，motion 为 `192×160`。

未知 robot_type 或歧义映射应明确指定三个真实源 key；空字符串表示缺失槽。例如单相机：

```bash
# 接在训练脚本后；其余环境变量见下一节。
bash launch/wsa_1_5_pretrain.sh \
  --dataset.view_keys='["observation.image","",""]'
```

显式列表应用到所有选中的数据集；混合不同相机命名的数据时应注册各自 schema。`egodex_v` 已将 `observation.image` 映射到 image0。

RGB/motion 当前均用固定 ImageNet mean/std，不叠加外部统计、增强或自定义 transform。Motion 训练的是解码后的三通道值；通道含义、scale/offset 和物理单位须由数据制作阶段明确。代码不猜测这些参数，视频解码一致也不等于原始 XYZ 物理误差已验证。

### 子任务标注

`meta/sidecar.json` 是 episode 对象列表，索引相对该 episode，end_frame 包含在区间内。下例要求 episode 至少 100 帧：

```json
[
  {
    "episode_index": 0,
    "task_instruction": "整理柔软的绳圈",
    "scene": "室内桌面",
    "subtasks": [
      {
        "subtask_id": 0,
        "subtask_text": "双手缓慢拉开绳圈并保持张紧",
        "start_frame": 0,
        "end_frame": 99,
        "hand_used": 2
      }
    ]
  }
]
```

hand_used：0=左手、1=右手、2=双手。区间不能越界或重叠，子任务编号不能重复。文本只读取示例中的任务、子任务、手和场景字段；物理语义应写进这些文本，额外 JSON 字段不会自动输入模型。

未来 32 帧按 frame_stride 采样，默认 1，允许跨子任务边界。复用 LeRobot 的 episode 索引与 padding mask，仅在 episode 剩余帧不足时重复末帧并将越界位置标记无效，不会取到下一个 episode。文本条件和 3D 目标图像仍对应当前子任务及其终点。VAE 使用显式 33 帧布局，motion 首帧只是 padding，不进入 MoT 或 loss。缺失标注使用空文本、关闭相应 3D 标签；缺失相机/motion 用 mask 屏蔽。声明存在但解码失败的视频会报错，整批无正权重有效监督也会停止。

当前没有按监督可用性过滤采样。关闭 3D 监督或缺少终点标注时，batch_size=1 可能抽到没有完整未来 VAE 帧组的 episode 尾部而停止。子任务切换不会触发这种时间 padding。需要连续训练这类数据时应补充采样过滤，不能把报错改成零 loss 后继续更新。

### 读取检查

```bash
export DATA_ROOT=/data/Egodex-LerobotV3
python util_scripts/discover_wsa1_5_datasets.py --root "$DATA_ROOT"
python util_scripts/inspect_wsa1_5_dataset.py "$DATA_ROOT/extra/knit_scarf" \
  --workers 2 --output outputs/inspect_knit_scarf.json
```

inspect 只检查默认 schema 的首批，并要求首样本有完整未来 32 帧 motion 和有效 RGB；不适用于缺失 motion 的消融数据，也不代表已经扫描全量数据。

## 4. 启动训练

### 单机

```bash
conda activate wsa15
cd /path/to/InternVLA-A-series
source weights/wsa1_5_weights.env
export DATA_ROOT=/data/Egodex-LerobotV3
export DATASET_REPO_IDS='["extra/knit_scarf"]'
export OUTPUT_DIR="$PWD/outputs/wsa1_5/run_001"
export PROC_PER_NODE=1
export BATCH_SIZE=1
export NUM_WORKERS=4
export GRAD_ACCUM_STEPS=1

# 仅打印命令，先核对路径、数据集与消融开关。
WSA_DRY_RUN=1 bash launch/wsa_1_5_pretrain.sh

# 在服务器正式启动；确保显存足以容纳完整配置。
bash launch/wsa_1_5_pretrain.sh
```

省略 DATASET_REPO_IDS 时会扫描合集下全部匹配的数据集；首次建议显式选取已检查的数据。新实验使用新输出目录。

默认 BF16、gradient checkpointing、600000 个 optimizer steps；SAVE_FREQ=10000、LOG_FREQ=10。梯度累积按实际 optimizer update 计步，有效 batch 为 `BATCH_SIZE × 总进程数 × GRAD_ACCUM_STEPS`。

| 配置 | 默认值 |
| --- | --- |
| RGB_LR / MOTION_LR | `1e-5` / `5e-5` |
| GOAL_LR / HEADS_LR | `5e-5` / `1e-4` |
| warmup / decay | 2000 / 600000，不随较短的 STEPS 自动缩放 |
| WSA_CACHE_DIR | 可选的冻结 VAE、文本、teacher 特征缓存目录 |
| WANDB_ENABLE | `false` |

额外参数直接接在脚本后，例如 `--policy.scheduler_warmup_steps=100 --policy.scheduler_decay_steps=10000`。完整参数见 `python src/lerobot/scripts/lerobot_train.py --help`。

### 多卡与多节点

```bash
# 单机四卡，其他环境变量同上。
PROC_PER_NODE=4 bash launch/wsa_1_5_pretrain.sh

# 两节点、每节点四卡：在各节点终端设置，模型/数据/输出路径须一致。
export NODE_COUNT=2
export PROC_PER_NODE=4
export MASTER_ADDR=10.0.0.1              # 改为节点 0 实际地址
export MASTER_PORT=29500
export NODE_RANK=0                       # 节点 1 设置为 1
bash launch/wsa_1_5_pretrain.sh
```

沿用现有 Accelerate/DDP、采样和 checkpoint 机制。默认 DIST_LOADING=false，各进程访问完整数据；当前不提供自动参数/optimizer 分片。

### 3D 消融

默认 ENABLE_3D_SUPERVISION=true：

| 配置 | G | Teacher、终点图像、3D loss |
| --- | --- | --- |
| ENABLE_3D_QUERIES=false | 不创建 | 关闭 |
| queries=true，LAMBDA_3D=0 | 保留联合前向和推理输出 | 全部关闭 |
| queries=true，LAMBDA_3D>0 | 参与联合计算 | 开启 |

```bash
ENABLE_3D_QUERIES=true LAMBDA_3D=0 \
  OUTPUT_DIR="$PWD/outputs/wsa1_5/queries_no_teacher" bash launch/wsa_1_5_pretrain.sh
```

ENABLE_3D_SUPERVISION=false 或两项 feature loss 权重同时为零也关闭监督。无监督 G 冻结并移出优化器；R/M 不读取 G，因此 R/M loss 不训练无监督 G，其输出不能当成已蒸馏几何表征。

训练只做一次 O/R/M/G 联合前向：R/M 使用带噪训练状态，G 读取同次隐藏特征；不额外采样未来。推理每个去噪步也包含已启用的 G。关闭 RGB/3D 时，对应预测为 None、有效性 mask 为全 False；关闭 motion 时 future_motion 为 None。

## 5. 配置、验证与恢复

### JSON 配置

复制并修改 [配置模板](configs/wsa1_5_pretrain.json) 的数据、权重和输出路径：

```bash
accelerate launch --num_processes=1 --mixed_precision=bf16 \
  src/lerobot/scripts/lerobot_train.py --config_path=configs/wsa1_5_pretrain.json
```

JSON 不展开 shell 环境变量。Policy 注册名是 WSA_1_5，dataset 名是 wsa1_5。改变 queries、expert 尺寸或优化器分组应新开实验，不要修改旧 checkpoint 配置强行续训。

### 服务器验证与缓存

以下命令会加载真实模型，其中训练验证会更新参数，**仅在服务器执行**。先修改好 JSON 路径，每次诊断使用不同输出目录。

```bash
python util_scripts/verify_wsa1_5_vae.py --vae-path "$VAE_PATH" --execute-on-server
python util_scripts/validate_wsa1_5_server.py \
  --config-path configs/wsa1_5_pretrain.json --execute-on-server \
  --steps 1 --reload-checkpoint --report outputs/wsa1_5_smoke.json

# JSON 已设置 policy.cache_dir 时，可预计算读到的 batch；不更新参数。
python util_scripts/validate_wsa1_5_server.py \
  --config-path configs/wsa1_5_pretrain.json --execute-on-server \
  --cache-only --cache-batches 100 --report outputs/wsa1_5_cache.json
```

诊断工具为单进程，短训练显式采用短 LR 计划；DDP 用正式 launcher 验证。缓存按内容、元数据、dtype、处理版本和冻结权重指纹区分并校验完整性。开启 3D 监督时，当前计算 teacher 缓存指纹仍需可读的 Omega 文件，不能只留缓存就删除 teacher 权重。

### 断点恢复

Checkpoint 含 pretrained_model 和 training_state；checkpoints/last 指向最近保存的 step。

```bash
export RESUME_CONFIG="$PWD/outputs/wsa1_5/run_001/checkpoints/last/pretrained_model/train_config.json"
PROC_PER_NODE=4 bash launch/wsa_1_5_pretrain.sh

# 下一次从头启动新实验前清除恢复模式。
unset RESUME_CONFIG
```

必须传保存的 **train_config.json**。恢复包含 optimizer、scheduler、RNG 和 step。完整 policy 已含 Wan/VAE/UMT5 权重，但 tokenizer 路径仍须可用；启用 3D 监督时还需要 Omega。推理不加载 Omega。

## 6. 本地检查与限制

```bash
# 小型随机模块和测试数据，不加载预训练权重、不执行 optimizer.step。
python -m pytest tests/test_wsa1_5*.py -q --basetemp outputs/pytest_wsa15
```

本轮按 ponytail.md 核对调用方、配置、注册和测试，修复小数索引截断、空/非标准扩展名权重指纹，以及关闭分支后 mask 仍有效的问题；合并 VAE 返回校验，删除重复正数检查和无仓库调用方的内部 scheduler target 方法。保留权重、mask、NaN/Inf、缓存及无监督 batch 检查，以及 policy/CLI 接口和恢复兼容字段。

本地检查不覆盖 GPU 安装、真实权重加载、完整训练、DDP、显存容量和 checkpoint 恢复。这些仍需服务器验证；motion 原始物理误差也尚未验证。

上游依据：[Wan ModelScope](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)、[Omega ModelScope](https://modelscope.cn/models/facebook/VGGT-Omega)、[Omega 固定版本依赖](https://github.com/facebookresearch/vggt-omega/blob/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6/requirements.txt)、[ModelScope 下载接口](https://github.com/modelscope/modelscope/blob/v1.34.0/modelscope/hub/snapshot_download.py)、[PyTorch 安装](https://docs.pytorch.org/get-started/previous-versions/)。使用前遵守各仓库许可证和访问要求。
