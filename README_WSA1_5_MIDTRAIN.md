# WSA1.5 统一动作空间中训练

本次适配覆盖原生机器人、Human Retarget、UMI Retarget 混合训练。沿用现有 WSA1.5 专家和并行 MoT；通过 schema、动作窗口读取和 prompt 区分本体。

## 1. 统一空间

开启 `policy.action_space=unified72`，同时设置 `state_dim=72`、`action_dim=72`。下标从 0 开始，区间右端不包含。

| 槽位 | 含义 | 容量 |
|---|---|---:|
| `[0:7]` | 左 EEF：xyz + quaternion xyzw | 7 |
| `[7:14]` | 右 EEF：xyz + quaternion xyzw | 7 |
| `[14:16]` | 左夹爪 | 2 |
| `[16:18]` | 右夹爪 | 2 |
| `[18:38]` | 左灵巧手 | 20 |
| `[38:58]` | 右灵巧手 | 20 |
| `[58:65]` | 左臂关节 | 7 |
| `[65:72]` | 右臂关节 | 7 |

灵巧手按 thumb/index/middle/ring/pinky 排列，每指预留 4 维。源数据有什么就由 schema 填什么；缺失维度置零并从 Action loss 中屏蔽，不凭空生成 EEF 或关节角。

- 原生 ALOHA joint 14 维：左臂 6 维填 `[58:64]`，左夹爪填 14；右臂 6 维填 `[65:71]`，右夹爪填 16。
- 重定向数据：选中本体的 `action.<name>` 提供关节、IK 夹爪或手部；`human_eef` 提供该本体对应的 EEF。其 grasp 字段不会覆盖 IK 夹爪。
- PiPER Inspire：保留双臂 12 维和双手各 12 维；组合标注中的旧平行夹爪不再填入夹爪槽位。每手 12 个记录关节包括独立和 mimic 关节，未记录的手部槽位屏蔽。
- UR7e Wuji：双臂各 6 维、双手各 20 维。

schema 新增 `unified_action_reorder`、`unified_state_reorder`，格式仍为 `[源起点, 源终点, 目标起点, 目标终点]`。旧 `action_reorder/state_reorder` 保留，默认 `action_space=legacy`，原有 14/16 维训练不受新布局影响。

本次补齐了现有 A1 joint 系列、ALOHA/PiPER RoboTwin、已有 quaternion EEF 等明确布局的映射，以及两套 retarget schema。未声明统一映射的 schema 会报错，避免将 Euler、全身等其他布局直接误填。

## 2. 数据读取与本体采样

继续使用 `dataset.type=wsa1_5`，每个数据集根目录独立判断：

| 数据集内容 | State / Action 来源 | 归一化 |
|---|---|---|
| 没有 `meta/annotations/action.parquet` | 原生 schema 对应的 state/action 字段 | 按 `robot_type` 查找，或指定 `external_stats_path` |
| 有 `meta/annotations/action.parquet` | 标注表中的当前姿态和未来姿态 | `retarget_stats_dir/<完整 schema 名>/stats.json` |

RGB、历史帧、motion、subtask、goal 仍走原有 WSA dataset。图像映射仍由源数据集 `robot_type` 对应的 schema 管理。动作 schema 按 `meta/info.json` 的 `robot_type` 选择前缀，再匹配标注列名：`egodex_v` 的 `action.piper` 对应 `egodex_synthetic_piper`，`egoverse_human_bimanual` 的同名列对应 `egoverse_synthetic_piper`。

UMI 沿用相同读取逻辑，缺少灵巧手列不会被要求补齐。新增源 `robot_type` 需在已有 `EMBODIMENT_SCHEMA_PREFIXES` 中登记实际前缀，并提供对应动作 schema 和图像映射；不会回退使用 EgoDex 的动作定义。

每次 `__getitem__` 都从该窗口的有效 IK 本体中均匀随机选一个；整个 chunk、state、prompt 使用同一个本体。`dataset.embodiments=[]` 使用全部可用本体，非空列表为白名单，支持列名短名（如 `piper`）或完整 schema 名。短名会在各源分别匹配，完整名称可精确限定来源。某个数据源没有白名单中的部分本体时取交集；没有任何候选时明确报错。

窗口规则与统计共用：

| 类型 | State | Action，H=chunk，s=stride |
|---|---|---|
| 原生机器人控制指令 | `state[t]` | `action[t], action[t+s], …, action[t+(H-1)s]` |
| 重定向逐帧姿态 | `pose[t]` | `pose[t+s], pose[t+2s], …, pose[t+Hs]` |

重定向窗口要求当前姿态、全部采样目标和共享 EEF 均有效，且不跨 episode。没有任何有效本体的 anchor 在初始化时剔除，视频时间和历史读取仍保持原始帧下标。原生数据沿用尾部 padding + loss mask；统计只计入实际有效目标。

先在源坐标中做 abs/delta 和归一化，再填统一槽位。delta 相对当前 state，具体哪些坐标取差由 schema 的 `action_mask_spec` 决定：EEF 位置取差，quaternion 和夹爪保持 schema 声明的绝对量。

## 3. 计算归一化

### 重定向数据

准备 `retarget_roots.txt`，每行一个含 `meta/info.json`、`data/` 和动作标注表的 LeRobot 根目录。可以同时列 Human 和 UMI 数据；相对路径以该文本所在目录为基准。

```bash
python util_scripts/compute_synthetic_embodiment_stats.py \
  --dataset-roots-file retarget_roots.txt \
  --output-dir norm_stats/midtrain_delta_h32_s1 \
  --action-space unified72 \
  --action-mode delta \
  --chunk-size 32 \
  --stride 1 \
  --num-workers 8
```

只计算指定本体时追加，例如：

```bash
  --embodiment piper --embodiment piper_inspire --embodiment ur7e_wuji
```

统一72模式按完整 schema 名输出每本体一份 `stats.json`，例如：

```text
norm_stats/midtrain_delta_h32_s1/egodex_synthetic_piper/stats.json
norm_stats/midtrain_delta_h32_s1/egoverse_synthetic_piper/stats.json
```

只有完整 schema 名相同的多个数据集根目录才合并统计。不同来源的同短名本体分开保存，源数据不必拥有相同的本体集合。可用 `--embodiment egodex_synthetic_piper --embodiment egoverse_synthetic_piper` 精确选择两类本体。PiPER 的统计仍是关节/夹爪 16 维加 EEF 14 维，共 30 维；模型输入仍为 72 维。

**旧版按短名合并的统一72统计需要重新计算，训练不回退读取旧的 `piper/stats.json`。** 其他策略的 legacy 统计格式保持原样。

统计文件记录 `action_space/action_mode/chunk_size/stride/action_mask_spec`；训练读取时核对这些字段。切换 abs/delta、chunk 或 stride 后请重算，并换用对应的统计目录。stride 默认 1。

### 原生机器人数据

同一次计算中的数据须有相同 `robot_type` 和动作字段布局。以 joint ALOHA 为例：

```bash
export HF_LEROBOT_HOME="$PWD/data"
python util_scripts/compute_norm_stats_multi.py \
  --repo_ids /absolute/RoboTwin/task_a/aloha-agilex_clean_50 \
             /absolute/RoboTwin/task_b/aloha-agilex_clean_50 \
  --action_mode delta --chunk_size 32 --stride 1 --num_workers 8 \
  --output_file "$HF_LEROBOT_HOME/stats/aloha/delta/stats.json"
```

这里 `aloha` 必须替换为 `meta/info.json` 中的真实 `robot_type`。统计在原始坐标中计算，legacy/unified72 均可使用。训练的 mode/chunk/stride 应与计算参数一致；原生旧格式统计文件不强制校验窗口元数据。

混合多个原生本体时，分别计算并放到 `data/stats/<robot_type>/<abs|delta>/stats.json`。单一本体也可用 `EXTERNAL_STATS_PATH=/absolute/stats.json` 显式指定；混合不同原始维度时应使用按本体查找方式。

## 4. 启动中训练

新增 `launch/wsa_1_5_midtrain.sh`，参数直接写在启动脚本里，通过环境变量覆盖，不额外读取训练配置 JSON。`INIT_CHECKPOINT_PATH` 指 Stage-1 `pretrained_model` 目录，优化器、scheduler 和 step 重新开始。

```bash
export INIT_CHECKPOINT_PATH=/absolute/stage1/checkpoints/520000/pretrained_model
export RETARGET_STATS_DIR=/absolute/norm_stats/midtrain_delta_h32_s1
export DATASET_REPO_ID="/absolute/simulation/task_a /absolute/human/task_b /absolute/umi/task_c"
export WAN_MODEL_PATH=/absolute/Wan
# WAN_MODEL_PATH 下应有 models_t5_umt5-xxl-enc-bf16.pth、google/umt5-xxl。
export OMEGA_CHECKPOINT_PATH=/absolute/vggt_omega_1b_256_text.pt

ACTION_MODE=delta ACTION_CHUNK_SIZE=32 ACTION_FRAME_STRIDE=1 \
NUM_HISTORY_FRAME=4 HISTORY_FRAME_STRIDE=4 \
USE_SUBTASK=false ALLOW_ENDFRAME_3D=true \
STEPS=100000 BATCH_SIZE=8 PROC_PER_NODE=8 \
bash launch/wsa_1_5_midtrain.sh
```

可用 `TEXT_ENCODER_PATH`、`TOKENIZER_PATH` 分别覆盖文本模型路径。统一72、Action 开启和在线 UMT5 在该脚本中明确设置；原有 configuration 的默认值保持不变。

省略 `DATASET_REPO_ID` 时会在 `DATA_ROOT` 下寻找含 `meta/info.json` 和 `videos/` 的 LeRobot 根目录；混合训练建议显式列出数据源，避免纳入无关数据。源之间的采样沿用现有 dataset 权重逻辑，可传 `WEIGHT_RULES_PATH`。本体采样仍是每窗口随机一个。

限定重定向本体：

```bash
EMBODIMENTS='["piper","ur7e_wuji"]' bash launch/wsa_1_5_midtrain.sh
```

该列表不改变原生机器人数据的本体。统计目录须包含所选有效本体的文件。

`RGB_FINETUNE_MODE=full|lora|frozen` 继续可用；学习率、四项 loss 权重、历史帧、chunk、stride、future frames、scheduler、保存频率等参数见脚本。Stage-1 的专家宽度、层数、输入分辨率等必须与加载权重相容；非默认结构可通过脚本末尾透传的 `--policy.xxx=...`、`--dataset.xxx=...` 参数设置。

只检查命令：`WSA_DRY_RUN=1 bash launch/wsa_1_5_midtrain.sh`。精确续训：`RESUME_CONFIG=/absolute/checkpoint/train_config.json bash launch/wsa_1_5_midtrain.sh`，续训分支使用保存的配置，不注入新训练参数。

## 5. Prompt、缓存与后训练

不使用 subtask 时，统一空间 prompt 为：

```text
You are a robot control agent,follow the instruction.
Task: knit a scarf
Robot Embodiment: egodex_synthetic_piper
Control Mode: joint
Action Mode: delta
```

`USE_SUBTASK=true` 时保留已有 Subtask/Hand/Scene 字段，再追加本体和控制字段。原始任务优先读取 sidecar；缺失时从 LeRobot episode tasks 回退。

中训练在线编码文本，本体抽样后才构造最终 prompt。有重定向标注的 Action 数据禁止离线文本缓存；预计算入口也会跳过这类根目录。

标准机器人数据后训练可以预计算，必须明确相同的空间和动作模式：

```bash
python util_scripts/precompute_wsa1_5_text_embeddings.py \
  --data-root /absolute/RoboTwin \
  --text-encoder-path "$WAN_MODEL_PATH/models_t5_umt5-xxl-enc-bf16.pth" \
  --tokenizer-path "$WAN_MODEL_PATH/google/umt5-xxl" \
  --enable-action --action-space unified72 --action-mode delta --no-use-subtask
```

后训练设置 `load_text_encoder=false`，从中训练 checkpoint 初始化时仍保留 `enable_action=true/action_space=unified72/state_dim=72/action_dim=72`。`init_checkpoint_path` 现在支持两种来源：

1. Stage-1：加载世界专家，再从该 RGB 权重插值初始化 Action 骨干。
2. 匹配动作空间的中训练/Action checkpoint：恢复已训练的 Action 和 Proprio，不重新插值。源 RGB LoRA 会先合入骨干，再按目标策略配置；在线源转离线目标时省略冻结 UMT5。

两者都是重新开始优化器的权重初始化。精确续训继续使用 resume。旧 legacy Action checkpoint 不直接转为统一72；它的输入/输出层维度与槽位语义不同。

例如从中训练转 RoboTwin Clean 后训练，缓存和原生归一化准备完成后：

```bash
INIT_CHECKPOINT_PATH=/absolute/midtrain/checkpoints/100000/pretrained_model \
DATA_ROOT=/absolute/RoboTwin \
EXTERNAL_STATS_PATH=/absolute/aloha_delta_h32_s1/stats.json \
STATE_DIM=72 ACTION_DIM=72 LOAD_TEXT_ENCODER=false \
bash launch/wsa_1_5_posttrain_robotwin_clean.sh --policy.action_space=unified72
```

训练、标准数据预计算、离线动作推理和 RoboTwin WSA1.5 适配器均使用同一本体 prompt 规则。RoboTwin 仍在输出时还原为环境需要的 joint 14 维，不会把 72 维直接交给环境。原仓库的 RoboTwin inference 流程未改。

## 6. 代码改动位置

| 位置 | 职责 |
|---|---|
| `dataset_schemas/schema.py`、现有 YAML | 选择 legacy/统一映射，记录本体名和源维度语义 |
| `datasets/retarget_actions.py` | 按数据源加载 schema，以完整身份隔离本体、prompt 和统计；共用有效窗口、帧对齐及 EEF 读取 |
| `datasets/synthetic_embodiment.py` | 复用已有按 row group 读取；在采样后校验有限值 |
| `policies/wsa1_5/dataset_wsa1_5.py` | 按来源分派、随机一个本体、归一化填槽、prompt 和 loss mask |
| `datasets/factory.py` | 重定向数据绕过原生动作 stats，保留现有混合数据流 |
| `configuration_wsa1_5.py` | 统一空间、白名单和统计目录参数 |
| `initialization.py`、`modeling_wsa1_5.py` | Stage-1 初始化与保留已训练 Action 的加载路径 |
| 两个 `compute_*stats*.py` | 匹配训练的 abs/delta、chunk、stride；可选本体统计 |
| `precompute_wsa1_5_text_embeddings.py` | 仅标准机器人 Action 缓存，支持统一本体 prompt |
| `infer_wsa1_5_action.py`、`evaluation/RoboTwin/wsa1_5/policy.py` | 统一空间推理 prompt 与原始动作还原 |
| `launch/wsa_1_5_midtrain.sh` | 独立中训练启动入口 |
| `tests/test_wsa1_5_midtrain.py`、launcher 测试 | 窗口/统计一致性、映射、混合反传、缓存和 checkpoint 加载 |

生产读取逻辑只新增一个共享模块；没有为 Human/UMI 分别复制 dataset，也没有改 MoT attention 或历史帧编码结构。

## 7. 验证与边界

相关回归结果为 **324 passed，5 skipped**，覆盖跨数据源同短名统计隔离、各源独立映射、完整名称白名单及串行/多进程统计。5 项跳过需要额外提供官方历史 RoboTwin checkout；其余数据、动作、缓存、增强、初始化和当前启动入口测试均通过。已将历史帧测试的间隔改为显式 fixture 参数，并移除引用已删除 launcher 的重复测试，没有修改训练默认配置。相关 Python 文件的 F 类静态检查、启动脚本语法检查及 `git diff --check` 通过。

已用小模型验证混合 batch 的 Action/Proprio 梯度，以及原生、重定向、缓存和初始化回归。实际 EgoDex `knit_scarf` 的 31,023 帧已按 delta/H32/stride1 计算全部 8 种本体统计：27,663 个 anchor 至少有一个有效本体，新版按完整 schema 名保存的统计文件位于本地 `outputs/wsa1_5_midtrain_demo_stats_by_schema/`。也已通过实际 LeRobotDataset 加载、RGB/历史/goal 视频解码和两条样本 collate，得到 state `[2,72]`、action `[2,32,72]`，prompt 正确包含抽样本体。

本地没有真实 Egoverse 和 UMI 样例；“数据源缺少灵巧手列”的情形通过合成 parquet 验证。未运行完整 GPU 中训练或真实机器人闭环评测。只有原始关节数据的仿真样本会屏蔽 EEF 槽位；若要同时监督 EEF，需源数据提供对应字段并在 schema 中声明。
