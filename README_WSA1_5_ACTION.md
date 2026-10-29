# WSA1.5 Action Post-training

## 模型结构

O/R/M/G/A 在同一次 MoT 中逐层并行计算。训练只做一次联合前向；推理的每个采样步同时预测并更新 RGB、Motion、Action，G 读取同一步的 world 表示。没有额外 Goal rollout、detached O/G cache 或 A-only 第二次前向。

| Query / Key | O | R | M | G | A |
|---|---:|---:|---:|---:|---:|
| O | 1 | 0 | 0 | 0 | 0 |
| R | 1 | 1 | 0 | 0 | 0 |
| M | 1 | 0 | 1 | 0 | 0 |
| G | 1 | 1 | 1 | 1 | 0 |
| A | 1 | 0 | 0 | 1 | 1 |

Action Expert 使用连续动作 projection、位置/RoPE、timestep modulation、Wan blocks 和 velocity head。state 经 projection 放进 A 的 cross-attention context；文本继续使用 UMT5。默认 Action hidden=1024、FFN=4096，深度和 attention head 几何与其他专家一致。

训练的 R/M 是标准 FM noisy target，因此 A 会经 G 受到未来 RGB/Motion 标签影响。这是当前并行训练语义，不能宣称动作分支独立于这些标签。真实 goal 图像和 Omega features 仍只作为损失标签。Action loss 可回传到未冻结的 G/O/world 参数。

## 历史编码：sparse_frame_encoding

编码方式通过 `policy.sparse_frame_encoding` 选择，并保存到 checkpoint 配置。默认 `joint`；训练和推理共用同一入口。

| 模式 | 对历史＋当前 RGB 的处理 | 4 张历史＋当前，共 5 帧 |
|---|---|---|
| `joint`（默认） | 整段调用冻结视频 VAE | 1 次调用，2 个 latent 时间步 |
| `independent` | 每张观测分别调用冻结 VAE，再按时间拼接 latent | 5 次单帧调用，5 个 latent 时间步 |

参考依据：[PrismWAM 常规 RoboTwin 的联合编码](https://github.com/THUSI-Lab/PrismWAM/blob/main/src/prismwam/models/wan22/rgb_depth_action.py#L513-L543)；[RoboDojo sparse-memory 的逐帧编码](https://github.com/THUSI-Lab/PrismWAM/blob/main/src/prismwam/models/wan22/rgb_memory.py#L74-L102)。两者是不同分支。

- `joint` 要求历史帧数量为 4 的倍数，使历史＋当前满足 Wan 的 `4n+1` 输入契约；非法长度直接报错。
- `independent` 支持任意非负历史帧数量，不做额外时间 padding 来满足 VAE 长度。
- `num_history_frame=0` 时，两种模式都沿用原单帧观测路径。
- 两种模式都使用连续 latent 序号的 temporal RoPE。5 帧联合编码时 O 位置为 `[0,1]`；独立编码时为 `[0,1,2,3,4]`，未来 latent 从对应 O 的下一个时间位置开始。不添加额外时间正弦向量。
- 此开关只选择编码方式；原始历史采样仍由 `num_history_frame/history_frame_stride` 控制
- 原“当前＋future_frames”的 RGB/Motion 监督 VAE 路径不变；历史分支只接收当前及过去图像。

例如 `num_history_frame=4, history_frame_stride=2` 仍采样 `[t-8,t-6,t-4,t-2,t]`。越过 episode 开头时复制首帧并标记无效。`independent` 保留每帧 mask；`joint` 按原四帧 latent 分组规则归约 mask，组内全无效时才屏蔽。缺失相机 mask 始终保留。

## Schema、动作与归一化

`dataset_schemas` 负责 image/state/action 源字段映射、重排和 delta mask。当前 schema 的 `action_mode=joint/end_effector` 表示控制空间，prompt 对应 `joint/endpose_effector`。abs/delta 是启动参数：launcher 的 `ACTION_MODE` 同时设置 dataset 和 policy 配置，二者不一致时直接报错。

归一化复用原 `datasets.factory._apply_dataset_stats_overrides()`：

1. `dataset.use_external_stats=true` 启用外部统计量。
2. 设置 `dataset.external_stats_path` 时直接读取该文件，适合单一数据集。
3. 否则读取 `$HF_LEROBOT_HOME/stats/{robot_type}/{abs|delta}/stats.json`。

使用原仓库普通 per-feature `stats.json`，不增加 robot/mode 包装层。复用 `DeltaActionTransformFn` 和 `NormalizeTransformFn`：先按 schema mask 做 delta，再归一化、重排；反归一化后加回相同的当前 state offset，返回物理单位绝对目标。保存 checkpoint 时保留实际使用的 stats。

RoboTwin AgileX Aloha 的注册名为 `aloha_robotwin`：每臂 xyz(3)＋quaternion(4)＋gripper(1)，共 16 维。delta mask 为 `[3,-4,-1,3,-4,-1]`，只转换 xyz。

| 数据 | 默认 batch shape / mask |
|---|---|
| 当前 RGB | `[B,3,384,320]`；mask `[B,1,384,320]` |
| 原始历史 RGB | `[B,4,3,384,320]`；对应逐帧像素 mask |
| O latent，`joint` | `[B,48,2,24,20]`，240 tokens |
| O latent，`independent` | `[B,48,5,24,20]`，600 tokens |
| state | `[B,16]`；`state_valid=[B]` |
| action | `[B,32,16]`；`action_valid_mask=[B,32,16]` |
| future RGB / motion | `[B,32,3,384,320]` / `[B,32,3,192,160]`，各自独立 mask |
| goal 视图 | 三张原始尺寸图像及 `[B,3]` 视角 mask，仅用于 teacher |
| 多层 3D features | 每层 `[B,3,273,2048]` |

动作窗口从 t 开始，episode 尾部补最后动作并屏蔽 loss，不跨 episode。混合数据缺少 action 时保留其他监督；全部训练源都没有 action 时提前报错。

## Prompt 与日志

固定前缀为用户指定的：`You are a robot control agent,follow the instruction.`，后接 Task；Action 模式再附加 Control Mode 和 Action Mode。`use_subtask=false` 不读取 sidecar 子任务内容；开启时仅追加真实 Subtask/Hand/Scene。有效子任务以其终点监督 G，否则只有 `allow_endframe_3d=true` 才使用 episode 终点。

前缀变化需要重算对应文本缓存。单独切换历史编码方式不改变 prompt，不需要重算文本 embedding。

保留四种 loss、action LR 和既有 `step_s/input_s/mot_s/pred_s/backward_s/optimizer_s`。不新增 Action 梯度范数或单独动作阶段耗时日志。

## Clean → Random 图像增广

复用本工作区 `lingbotvla/lingbotvla/data/vla_data/visual_augmentation.py` 的实现与配方，放在 `wsa1_5/utils/visual_augmentation.py`。训练时先在各相机原始 `[0,1]` RGB 上增广，再拼接 WSA 画布并归一化至 `[-1,1]`。Stage-1 和 Action posttrain 都可使用；默认两个开关均关闭。

| dataset 参数 | 默认值 | 作用 |
|---|---|---|
| `image_augment` | `false` | 色彩、色温、仿射、噪声、模糊、JPEG 和矩形遮挡 |
| `object_paste` | `false` | 独立控制物体截图粘贴，可在关闭其他增广时单独开启 |
| `image_augmentation_config` | `configs/augmentation/robotwin_conservative.yaml` | 复用 LingBot 的温和配方；完整配方为同目录 `robotwin_full.yaml` |

两个开关分别约束 YAML 中的图像操作概率与 `paste_prob`，因此只开启 `image_augment` 不会粘贴。两个开关均关闭时完全跳过 YAML、素材读取和增广随机采样。开启任一开关后，保留配方的 10% 原图分支；剩余 90% 样本进入组合分支，遮挡/粘贴的条件概率分别为 95%/50%。温和配方的遮挡面积为所选视角的 1%–4%，每张 paste 为 0.5%–2.5%；每次均匀抽取 1–2 张不同截图，保持宽高比。

同一样本只抽一次参数，历史、当前、未来 RGB 和 Omega 终点目标图像均重放这组参数，包括单独解码的终点图像。常规图像操作在全部有效视角间共享参数；遮挡和 paste 各自独立按头部/左腕/右腕 50%/25%/25% 选择一个视角。相机角色由 WSA 的 top/left/right 槽位确定，不依赖源数据的相机 key；缺失相机不参与选择，剩余视角重新归一化权重。缺失视角保持无效 mask 和归一化后的 `-1` 占位。

state、action、文本、时间戳和 mask 不变；已有 motion 三通道标签也不做颜色或空间增广。仿射属于 RGB 观测扰动，不代表真实三维相机移动，不重新计算 motion。遮挡/paste 不保护物体或夹爪，RGBA 会 alpha 混合，RGB 截图自带的背景也会一起粘贴。关闭这些操作可直接将对应 YAML 中的 `*_prob` 置 0。

45 张截图、来源索引和原始说明已复制到 `assets/RoboTwin_Clean_Object_Crops/`，无须安装或保留 LingBot 仓库。YAML 和 `crops_dir` 的相对路径均以 InternVLA 项目根目录为基准，也支持绝对路径；启用 paste 会提前检查素材，进程内缓存图片。迁移训练环境时需同步配方与素材；JPEG 的 CPU 编解码及多帧增广可能增加 DataLoader 开销。

```bash
# Action posttrain；Stage-1 launcher 使用相同的三个环境变量。
IMAGE_AUGMENT=true OBJECT_PASTE=true \
  bash launch/wsa_1_5_action_posttrain.sh

# 只开启 paste。
IMAGE_AUGMENT=false OBJECT_PASTE=true \
  bash launch/wsa_1_5_action_posttrain.sh

# 切换完整配方，也可指向自定义 YAML。
IMAGE_AUGMENT=true OBJECT_PASTE=true \
  IMAGE_AUGMENTATION_CONFIG=configs/augmentation/robotwin_full.yaml \
  bash launch/wsa_1_5_action_posttrain.sh

# 直接 CLI 参数；已有 launcher 会转发末尾参数。
bash launch/wsa_1_5_action_posttrain.sh \
  --dataset.image_augment=true --dataset.object_paste=true \
  --dataset.image_augmentation_config=configs/augmentation/robotwin_conservative.yaml
```

这些字段随训练配置保存；resume 沿用保存的配置，launcher 的新训练默认值不会覆盖它。在线 policy 推理不执行 dataset 增广，不读取配方或素材；离线构建评测 dataset 时显式传 `WSA1_5Dataset(..., training=False)`。Clean → Random 的成功率收益需实际闭环评测确认。

## 文件组织

| 已有文件 | 职责 |
|---|---|
| `experts.py` | RGB/Motion/Action experts、联合 MoT forward 和采样 |
| `modeling_wsa1_5.py` | policy API、联合训练、历史编码模式、在线动作队列 |
| `dataset_wsa1_5.py` | 数据采样、prompt、schema state/action 适配与归一化 |
| `utils/` | 时间布局、张量校验、计时、视频解码、可视化及 RGB 增广辅助代码 |
| `initialization.py` | 权重初始化、Stage-1 加载和 RGB LoRA |
| `mot.py` / `attention.py` | 五组专家前向与可见性 |

辅助代码统一收进 `utils/`，按职责保留 `temporal.py`、`validation.py`、`timing.py`、`video.py`、`visualization.py` 和 `visual_augmentation.py`。包入口不聚合导入，避免配置解析时加载可视化依赖。原根目录位置不再保留兼容文件，外部引用需改为 `lerobot.policies.wsa1_5.utils.<模块>`。

不新增编码模块或 Action 核心文件。原 `action.py/action_data.py/lora.py` 已合并删除；只保留必要的 launcher、配置、推理 CLI、测试和本文档。

## 服务器使用

以下仅供 Linux GPU 服务器执行；本地检查不加载完整模型权重。

```bash
cd /path/to/InternVLA-A-series
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export DATA_ROOT=/data/robotwin_lerobot
export DATASET_REPO_ID=aloha/task_name
export INIT_CHECKPOINT_PATH=/models/stage1/checkpoints/last/pretrained_model
export OMEGA_CHECKPOINT_PATH=/models/vggt_omega_1b_256_text.pt
export ACTION_MODE=delta
export EXTERNAL_STATS_PATH=/data/normalization/aloha_delta/stats.json

# 默认整段联合编码；可先 WSA_DRY_RUN=1 检查启动命令。
PROC_PER_NODE=2 bash launch/wsa_1_5_action_posttrain.sh

# 切换为逐帧独立编码；使用新的输出目录开启新实验。
SPARSE_FRAME_ENCODING=independent OUTPUT_DIR=outputs/action_independent \
  bash launch/wsa_1_5_action_posttrain.sh

# 配置文件 / 直接 CLI 对应字段：--policy.sparse_frame_encoding=joint|independent
# 恢复沿用保存的编码模式，不重新应用 launcher 默认值。
RESUME_CONFIG=/path/to/stage2/checkpoints/last/pretrained_model/train_config.json \
  bash launch/wsa_1_5_action_posttrain.sh

python util_scripts/infer_wsa1_5_action.py \
  --checkpoint /path/to/stage2/checkpoints/last/pretrained_model \
  --dataset-root "$DATA_ROOT/$DATASET_REPO_ID" \
  --input observations.pt --output actions.pt --seed 0
```

不设置 `EXTERNAL_STATS_PATH` 时采用本体/abs-delta 自动路径。文本缓存继续使用 `util_scripts/precompute_wsa1_5_text_embeddings.py --action-mode delta`，并提供实际 UMT5/tokenizer 路径。

推理输入包含当前 RGB/mask、原始历史 RGB/mask、task 列表和按 schema 源 key 保存的 `raw_state`。CLI 负责 prompt 和 state 归一化，并自动使用 checkpoint 中保存的编码模式。`return_world=true` 可返回同次联合采样的 world 输出；默认不解码视频。episode 或任务切换时调用 `reset()`。

Stage-1 初始化要求原参数 key/shape 匹配，只允许缺少新的 Action/LoRA 参数。Stage-2 resume 沿用完整保存配置及原 optimizer/scheduler/RNG/step 体系。改变编码模式会改变 O 的表示与 token 数，应作为新实验，不在 resume 时无声切换。

## 验证边界

CPU 测试检查两种 VAE 调用方式、时间长度、mask、RoPE、配置序列化、launcher 默认值/覆盖、单次并行 MoT、Action 梯度、普通 stats 路径、abs/delta、checkpoint 和在线历史。增广测试覆盖独立开关、Stage-1/Action 多帧重放、独立解码的终点、单视角 paste、透明通道、触发概率、缺失相机、评测关闭，以及 state/action/motion 标签保持原样。测试使用小尺寸随机模块、合成数据和真实截图素材。

增广接入后回归：**457 passed, 11 deselected**（86.90 秒）。新增代码 Ruff、修改生产代码的导入检查、Python 编译及两个 launcher 的 shell syntax 检查均通过；45 张素材及随附索引与 LingBot 原文件逐一校验一致，像素变换应用函数也保持原实现。测试报告见 [JUnit XML](outputs/wsa1_5_action_cpu_results.xml)。

旧 pretrain launcher 的 PET 环境及默认参数与 11 项原测试不匹配，仍单独列出。未加载完整 Wan/Omega/WSA 权重，未运行 GPU、正式训练或真实 DDP；独立编码会增加 O tokens，实际显存、延迟及任务效果需服务器验证。
