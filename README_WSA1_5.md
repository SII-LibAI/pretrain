# WSA1.5 使用说明

当前 `WSA_1_5` policy 输入当前 RGB 和文本，预测未来 32 帧 RGB、未来 32 帧三通道 motion，以及可选的子任务终点 3D 特征；不预测 action。

WSA1.5 专用实现统一放在 `src/lerobot/policies/wsa1_5/`：

- [configuration_wsa1_5.py](src/lerobot/policies/wsa1_5/configuration_wsa1_5.py)：集中定义 policy 配置 `WSA1_5Config` 和数据配置 `WSA1_5DatasetConfig`，保留 `policy.type=WSA_1_5`、`dataset.type=wsa1_5` 及既有参数字段。
- [dataset_wsa1_5.py](src/lerobot/policies/wsa1_5/dataset_wsa1_5.py)：LeRobot adapter、Segment、归一化、mask 和 collate。
- [text_embeddings.py](src/lerobot/policies/wsa1_5/text_embeddings.py)：文本编码、离线缓存读取和校验。
- [modeling_wsa1_5.py](src/lerobot/policies/wsa1_5/modeling_wsa1_5.py)：模型、训练前向和推理入口。

通用 `lerobot.datasets.factory` 保留接入路由。原 `lerobot.datasets.configuration_wsa1_5`、`lerobot.datasets.wsa1_5`、`lerobot.datasets.wsa1_5_text` 的 Python 导入位置已迁移，外部脚本需同步更新；训练 JSON/CLI 和文本缓存格式保持不变。

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

默认 `dtype=bfloat16`、`trainable_weights_fp32=false`：RGB、Motion、3D、VAE 和启用的 Omega 权重使用 BF16。训练默认 `load_text_encoder=false`，只读取提前计算的文本 embedding，不构造 UMT5/tokenizer。损失、部分归一化及 VAE 标准化常数仍保留必要的 FP32 数值运算。

AdamW 使用 `foreach=false` 降低更新时的临时显存；DDP 启用 `gradient_as_bucket_view` 复用梯度与通信 bucket 存储，恢复 optimizer 后仍保持该更新路径。这些设置不修改学习率、betas、eps 或 weight decay。DDP 每卡仍保留完整模型和 optimizer 状态，实际显存峰值须在服务器测量。

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

RGB/motion 视频解码得到 `[0,1]` 的浮点张量后，统一用 `img * 2 - 1` 映射到 `[-1,1]`，再输入 WanVAE；不使用 ImageNet mean/std，不叠加外部统计、增强或自定义 transform。缺失区域在映射后置零，并通过无效 mask 排除监督。Omega 的目标图像保持原始 `[0,1]`，直接双线性缩放至 `256×256`，再交给教师内部归一化。Motion 训练的是解码后的三通道值；通道含义、scale/offset 和物理单位须由数据制作阶段明确。代码不猜测这些参数，视频解码一致也不等于原始 XYZ 物理误差已验证。

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

## 4. 提前计算文本 embedding

子任务标注固定后，在 GPU 服务器运行一次 [文本预计算脚本](util_scripts/precompute_wsa1_5_text_embeddings.py)。它只加载冻结的 UMT5 和 tokenizer，不加载 DiT、VAE、Omega，不做参数更新。

```bash
source weights/wsa1_5_weights.env
export DATA_ROOT=/data/Egodex-LerobotV3
python util_scripts/precompute_wsa1_5_text_embeddings.py \
  --data-root "$DATA_ROOT" \
  --text-encoder-path "$TEXT_ENCODER_PATH" \
  --tokenizer-path "$TOKENIZER_PATH"
```

`--data-root` 可以是单个数据集，也可以是合集目录。文本预计算与训练统一调用 [find 发现脚本](util_scripts/discover_wsa1_5_datasets.sh)，扫描该目录本身及任意层级子目录，通过 `meta/info.json`、`data/` 和 `videos/` 识别数据集；Python 文本模块不再保留另一套 discover。扫描错误、找不到数据集、错误的标注结构都会报错。

每个数据集的缓存位于 `meta/wsa1_5_text_embeddings/<文本SHA256>.safetensors`。文本复用训练适配器的 `condition_text` 规则，每个不同文本只计算一次；另计算一个用于缺失标注帧的空文本 embedding。缓存保存 BF16 context 和布尔 token mask，落盘时移除末尾 padding，读取时补回 512 tokens。训练文本超过上限时明确报错，不截断。

默认 `max_text_length=512`，使用 Wan 的 4096 维 UMT5 特征。更改长度时，预计算用 `--max-text-length`，训练同步设置 `--policy.max_text_length`。自定义标注位置时，两边同步指定 `--sidecar-file` 和 `--dataset.sidecar_file`；缓存位置分别用 `--embedding-dir` 和 `--dataset.text_embedding_dir`。

缓存不存在或不匹配时明确报错，不在线补算、不加载 UMT5、不换成零文本。只保留这一种缓存：VAE latent 和 Omega 特征每批在线提取，没有旧版 `policy.cache_dir`、权重指纹或其他缓存兼容路径。重复运行预计算会原子替换对应文本文件；标注已固定时无需再次运行。生成缓存后，训练节点不再需要 UMT5/tokenizer 文件，仍需 Wan/VAE 及启用的 Omega 权重。

## 5. 启动训练

```bash
conda activate wsa15
cd /path/to/InternVLA-A-series
source weights/wsa1_5_weights.env
export DATA_ROOT=/data/Egodex-LerobotV3
export OUTPUT_DIR="$PWD/outputs/wsa1_5/run_001"

# 只打印启动命令，不加载模型。
WSA_DRY_RUN=1 bash launch/wsa_1_5_pretrain.sh

# 正式启动；默认单机双卡，每卡 batch=1。
bash launch/wsa_1_5_pretrain.sh
```

只需设置 `DATA_ROOT=父目录`。启动脚本用 `find -L` 递归查找其中所有同时包含 `meta/info.json`、`data/` 和 `videos/` 的数据集，打乱后以空格分隔的相对路径传给训练入口，无需填写子目录列表。若 `DATA_ROOT` 本身就是一个数据集，其 repo_id 为 `.`。不再通过 `DATASET_REPO_ID` / `DATASET_REPO_IDS` 环境变量传 JSON 列表，因此不会将 `['.']` 拼成目录名。父目录可以含空格；数据集的相对路径不能含空白字符，遇到时明确报错。多数据集均须提前生成自己的文本缓存。

启动脚本保留当前已设置的超参数：

| 参数 | 启动脚本默认值 |
| --- | --- |
| RGB_LR / MOTION_LR / GOAL_LR / HEADS_LR | 均为 `5e-5` |
| LAMBDA_RGB / LAMBDA_MOTION / LAMBDA_3D | `1.0` / `1.0` / `0.01` |
| FEATURE_MSE_WEIGHT / FEATURE_COSINE_WEIGHT | `1.0` / `0.0` |
| SCHEDULER_WARMUP_STEPS / SCHEDULER_DECAY_STEPS | `2000` / `600000` |
| PROC_PER_NODE / NODE_COUNT | `2` / `1` |
| BATCH_SIZE / NUM_WORKERS / GRAD_ACCUM_STEPS | `1` / `4` / `1` |
| STEPS / SAVE_FREQ / LOG_FREQ | `600000` / `10000` / `10` |
| WANDB_ENABLE / DIST_LOADING | `false` / `false` |
| 权重与计算精度 | BF16，`trainable_weights_fp32=false` |
| 文本编码 | `load_text_encoder=false`，必须有离线文本缓存 |

各组余弦学习率最终降到峰值的 `0.1`；短实验不会自动缩放 warmup/decay 步数。可通过环境变量覆盖表内大写参数，或把 policy/dataset 参数接在启动命令后；例如：

```bash
PROC_PER_NODE=1 BATCH_SIZE=1 bash launch/wsa_1_5_pretrain.sh

# 改为单机四卡，梯度累积两次。
PROC_PER_NODE=4 GRAD_ACCUM_STEPS=2 bash launch/wsa_1_5_pretrain.sh

# 按需改变衰减终点；默认值仍为 0.1。
bash launch/wsa_1_5_pretrain.sh --policy.scheduler_decay_lr_ratio=0.05
```

有效 batch 为 `BATCH_SIZE × 总进程数 × GRAD_ACCUM_STEPS`，step 按 optimizer update 计数。JSON 模板见 [configs/wsa1_5_pretrain.json](configs/wsa1_5_pretrain.json)，其中原有学习率、损失权重与 launcher 不完全相同，须按实验选择；JSON 不展开 shell 变量。

`dataset.streaming=false` 是当前随机访问适配器的要求：需要访问未来窗口和子任务终点，现有流式接口不能直接替换。数据按需解码，false 不表示一次加载所有视频；设置 true 会报错。

多节点在每个节点设置相同模型、数据和输出路径，并设置不同 NODE_RANK：

```bash
export NODE_COUNT=2
export PROC_PER_NODE=4
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500
export NODE_RANK=0   # 另一个节点为 1
bash launch/wsa_1_5_pretrain.sh
```

### 3D 消融与损失

G 使用 WSA 默认 slot_noise：819 个槽位，每视角 1 camera、16 register、256 patch。输入为 `0.5 × 正弦位置编码 + 0.25 × 标准正态噪声`；G 时间步固定为 `0.5 × num_train_timesteps`，当前为 500。训练时每个样本重新采样，推理/no_grad 使用 seed=0 的固定噪声。Query 自身没有可学习参数，骨干和预测头参与学习。

Omega 的每个有效目标视角都从原始 `[0,1]` 浮点张量直接双线性缩放至 `256×256`（`mode="bilinear"`、`align_corners=false`），不裁剪、不补白，也不转成 uint8/PIL。完整画面映射到正方形，非正方形原图会改变宽高比。Omega 内部自行做 ImageNet 标准化，外部不重复归一化。

教师每层直接输出 `[1,V,273,2048]`，按原顺序与学生的 `1 camera + 16 register + 16×16 patch` 槽位对应；已删除 `FeatureAligner3D`，不再对教师特征插值或重排。有效视角的全部 273 个 token 都参与监督，不再生成空间 padding mask；缺失视角直接跳过教师前向，整组 token 屏蔽。训练时仍要求对应的当前观测视角有效，缺少子任务标注时仍关闭该样本的 3D 监督。

例如 `480×640`（高×宽）直接缩放到 `256×256` 后输入教师，得到固定 273 个 token，按相同索引监督学生。该流程改变了 3D 标签的空间对应关系，不应把旧特征插值或补白方案下的实验当作同一配置直接续训。

| 设置 | G 联合计算 | Omega / 终点图像 / 3D loss |
| --- | --- | --- |
| ENABLE_3D_QUERIES=false | 不创建 | 关闭 |
| queries=true，LAMBDA_3D=0 | 保留 | 关闭 |
| queries=true，LAMBDA_3D>0 | 保留 | 开启 |

两项 feature 权重同时为零也关闭 3D 监督。无监督 G 冻结并移出优化器；R/M 不读取 G，其 loss 无法训练无监督 G。已删除 `enable_3d_supervision`，用 `lambda_3d=0` 关闭监督。

训练只做一次 O/R/M/G 联合前向：R/M 是 Flow Matching 带噪 latent，G 读取同次隐藏特征；不额外采样未来，真实终点和 teacher 仅作为损失标签。推理每个去噪步骤也包含 G，返回最后一次联合前向的 3D 特征。

3D 各层损失现在与 WSA 公式对齐：

```text
MSE = mean((LayerNorm(student.float()) - LayerNorm(teacher.float()))²)
cosine = mean(1 - cosine_similarity(student.float(), teacher.float()))
loss_3d_feature = 各层有效槽位损失的平均
               = mean_layers(FEATURE_MSE_WEIGHT × MSE + FEATURE_COSINE_WEIGHT × cosine)
total_loss = LAMBDA_RGB × loss_rgb_flow + LAMBDA_MOTION × loss_motion_flow
           + LAMBDA_3D × loss_3d_feature
```

旧公式对 L2 单位向量逐维求 MSE，2048 维随机预测也只有 `2/2048≈0.001`，量级被维数压低。新公式使用 LayerNorm MSE，独立随机特征的基线约为 2；真实初始化结果取决于特征相关性，不强行设定初始 loss。日志中的 `loss_3d_feature` 是乘 `lambda_3d` 前的值；当前启动配置 MSE=1、cosine=0、lambda_3d=0.01，因此 raw loss 约 2 时对总损失的贡献约 0.02。缺失视角和缺失标注对应的 3D 特征不参与归约，teacher 不接收梯度。

### 参数与耗时日志

加载时仅报告权重校验结果；训练前按 RGB/Observation、Motion、Future3D、VAE、UMT5、Omega 汇总参数量，单位 B。默认 UMT5 为 disabled；有 3D 监督时，首次更新前会看到 Omega 路径、加载结果和 ready 日志。

WSA 日志保留 step、sample、loss、grad norm、lr、吞吐、data_s 和总 updt_s，移除 episode/epoch；增加以下秒数，按 log_freq 区间平均：

| 字段 | 测量范围 |
| --- | --- |
| input_s | 当前/未来图像 VAE 编码、Omega 标签提取、读取后的文本准备、Flow Matching 状态及 MoT 输入编码 |
| mot_s | 一次 O/R/M/G MoT 骨干前向 |
| pred_s | RGB/Motion 速度预测头及 3D 特征预测头 |
| backward_s | 反向、梯度同步及裁剪；checkpointing 重算包含在这里 |
| optimizer_s | 本 micro-batch 的 optimizer/scheduler 和更新收尾；累积时未更新的 micro-batch 会很短 |

CUDA 阶段使用 event 计时，在读取统计时同步；输入耗时不含 DataLoader 读取缓存的时间，后者属于 data_s。pred_s 是训练这一次 FM 速度预测，不会为计时额外运行多步去噪。updt_s 还包含检查、loss 计算和日志取值，阶段之和不要求严格相等。

## 6. 断点恢复与推理

```bash
export RESUME_CONFIG="$PWD/outputs/wsa1_5/run_001/checkpoints/last/pretrained_model/train_config.json"
bash launch/wsa_1_5_pretrain.sh
unset RESUME_CONFIG
```

使用本版本保存的 train_config.json，恢复 optimizer、scheduler、RNG 和 step。恢复模式沿用保存的配置，不重新应用 launcher 的默认值。文本缓存继续保留在各数据集；启用 3D 监督时仍需独立的 Omega 文件。默认完整 policy 含 Wan/VAE，不含 UMT5 或 teacher。

旧版本带 UMT5 参数、通用缓存配置、FP32 optimizer 状态或可学习 query 的 checkpoint 不能直接当作本版继续训练；本轮变更应从 Wan 权重新开实验，不修改旧配置强行恢复。

推理不加载 Omega。当前 RGB 输入也须先执行 `img * 2 - 1`；RGB/motion 预测结果经过 VAE 解码、有限值检查和 `[-1,1]` 截断后，用 `(img + 1) / 2` 返回 `[0,1]`。默认无文本编码器时，从 `lerobot.policies.wsa1_5.text_embeddings.TextEmbeddingCache` 读取对应文本的 `(embedding, valid)`，加 batch 维后传给 `predict_future(..., text_embedding=embedding[None], text_valid=valid[None])`；输入仍只来自当前 RGB 和文本。`load_text_encoder=true` 保留给显式在线冻结编码的实验，其完整 checkpoint 会包含 UMT5；不能仅给默认离线训练所得 checkpoint 改这个开关，严格加载会拒绝缺失的文本编码器参数。

此前使用 ImageNet 输入归一化训练的 checkpoint 与当前输入分布不同，不应直接恢复训练并视为同一实验。

## 7. 本地检查范围

```bash
# 小型随机模块和合成数据；不加载真实权重、不更新 optimizer。
python -m pytest tests/test_wsa1_5*.py -q --basetemp outputs/pytest_wsa15
```

已删除旧 WSA VAE 验证、训练/update 验证和数据 inspect 工具；保留 ModelScope 下载脚本及离线文本预计算脚本。检查覆盖缓存与实际提示词一致性、缺失/非法 embedding 拒绝、BF16 模块构造、WSA 3D 损失公式、mask/梯度及启动参数。

本地不验证真实模型执行、CUDA 显存峰值、完整 DDP 训练或 checkpoint 续训；这些须在服务器验证。Motion 物理单位和视频压缩误差仍由数据制作侧确认。
