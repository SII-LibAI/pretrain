# 合成人类具身数据预训练：数据流、Dataset Schema 与输入审计

本文对应 `dataset.type=synthetic_embodiment`，说明 EgoDex（`robot_type=egodex_v`）和
Egoverse（`robot_type=egoverse_human_bimanual`）从 LeRobotV3、`action.parquet`、schema、归一化文件到
InternVLA-A1.5 forward 的真实数据流。

本文只描述和审计现有代码，不修改 policy、Dataset 或训练逻辑。行号对应当前版本；后续代码变化时应同时按函数名定位。

## 1. 结论

姿态主链路的顺序是：

```text
parquet 原始顺序（右手在前，左手在后）
  -> 按 action_mask_spec 计算 delta/absolute 混合 action
  -> 使用原始顺序的本体统计量做 Normalize
  -> Pad 到 policy 指定维度
  -> 按 schema Reorder 为左手在前、右手在后
  -> 使用处理后的 state 构造 prompt
  -> 图像与文本处理
  -> collate 成 batch
  -> policy.forward
```

因此，当前统计和 reorder 的相对顺序是正确的：

- 归一化文件应按 parquet 的原始“右手在前、左手在后”顺序计算。
- 不要把统计文件里的左右手 mean/std 预先交换。
- 只修改 reorder 不需要重新计算统计量。
- state 与 action 必须使用相同的 reorder，否则 prompt state 与动作目标的左右手语义会错位；当前两者使用相同 schema 配置。

静态审计同时发现几个需要注意的现状：

- 通用 `LeRobotDataset` 在视频解码失败时会记录错误，然后返回全零图像；当前合成人类 adapter 没有关闭这个 fallback。
- 当前没有逐 batch 自动检查器。训练代码只在每 200 step 打印主进程当前 batch 的第一个样本。
- prompt 的 state 离散化没有先 clip；标准化 state 小于 `-3` 时，`np.digitize(...)-1` 可能产生 `-1` 桶。
- prompt 的 `action layout` 写的是语义块大小，不是精确模型索引。xArm7、RM75、UR7e 中间有保留零槽，prompt 不会显示这些空槽。
- `annotation_row_group_cache_size` 已在配置中声明，但当前 factory 没有把它传给 `SyntheticEmbodimentDataset`，因此启动参数暂时不会生效。

这些项目不代表已经观察到实际数据错误，但意味着仅靠当前代码不能保证每一个真实 batch 都没有黑图、离群值或 prompt 异常。

## 2. 完整数据流

```mermaid
flowchart TD
    A[启动脚本扫描 DATASET_ROOT] --> B[dataset.repo_id 列表]
    B --> C[factory 为每个 repo 创建 LeRobotDataset]
    C --> D[读取 info.json 的 source robot_type]
    D --> E[选择 source schema 前缀和本体 schema]
    C --> F[读取当前视频帧和 task]
    E --> G[读取 action.parquet 与 per-body stats]
    G --> H[预扫描完整有效窗口]
    H --> I[__getitem__ 随机选择有效本体]
    I --> J[state=q[t], action=q[t+1:t+H+1]]
    J --> K[Delta -> Normalize -> Pad -> Reorder]
    F --> L[Resize -> Camera Remap]
    K --> M[加入本体名与 action layout]
    L --> N[ChatProcessor: 图像+task+state]
    M --> N
    N --> O[Unify -> collate]
    O --> P[InternVLA-A1.5 forward/loss]
```

### 2.1 启动脚本和 factory

- [`launch/internvla_a1.5_pretrain_egodex.sh`](../launch/internvla_a1.5_pretrain_egodex.sh) 在启动阶段扫描父目录，并把相对路径列表传入 `dataset.repo_id`。Dataset 内部不递归发现子数据集。
- [`make_dataset`](../src/lerobot/datasets/factory.py#L495) 解析 repo 列表、按 rank 分配子数据集，然后逐个调用 `_build_single_dataset`。
- [`_build_single_dataset`](../src/lerobot/datasets/factory.py#L328) 先创建 `LeRobotDataset`，再包装为 `SyntheticEmbodimentDataset`。
- synthetic adapter 自己构造 action，所以 factory 会从基础 Dataset 的 `delta_timestamps` 中删除 `ACTION`；图像仍按 policy 的 `image_delta_indices` 加载。
- 当前 `action_loss_only=true` 时 `image_delta_indices=[0]`，只读取当前图像帧。

### 2.2 有效帧和随机本体

[`_build_streaming_index`](../src/lerobot/datasets/synthetic_embodiment.py) 为每个本体检查完整窗口：

```text
valid[t] AND valid[t+1] AND ... AND valid[t+H]
```

同时要求：

- `t..t+H` 属于同一 episode；
- `frame_index` 连续；
- annotation 的 `index/episode_index/frame_index` 与基础 LeRobot 数据逐行对应；
- `human_eef` 左右手 valid 都为真；其它本体使用自己的 `action.<name>_valid`。

没有任何本体拥有完整有效窗口的 anchor 会在 Dataset 建索引时直接删除，而不是送入 loss 后做时间 mask。

[`SyntheticEmbodimentDataset.__getitem__`](../src/lerobot/datasets/synthetic_embodiment.py) 在当前 anchor 的候选本体中随机选一个，然后读取：

```python
poses = annotation[t : t + H + 1]
state = poses[0]
action = poses[1:]
```

即：

```text
state[t]  = q[t]
action[t] = [q[t+1], q[t+2], ..., q[t+H]]
```

同一视频帧在不同 epoch 可以随机抽到不同本体，所以 adapter 会把本体名加入 task：

```text
Current robot embodiment: <name>; task: <原任务>; action layout: ...
```

## 3. dataset_schema 的作用

### 3.1 schema 选择

每个子数据集先从 `meta/info.json` 读取 source `robot_type`，再通过
[`EMBODIMENT_SCHEMA_PREFIXES`](../src/lerobot/datasets/synthetic_embodiment.py) 选择前缀：

```text
egodex_v                 -> egodex_synthetic_
egoverse_human_bimanual  -> egoverse_synthetic_
```

若当前样本随机抽到 `franka`：

- EgoDex 使用 `egodex_synthetic_franka`；
- Egoverse 使用 `egoverse_synthetic_franka`。

schema 文件分别是：

- [`egodex_v.yaml`](../src/lerobot/dataset_schemas/configs/egodex_v.yaml)
- [`egoverse_human_bimanual.yaml`](../src/lerobot/dataset_schemas/configs/egoverse_human_bimanual.yaml)

### 3.2 各字段控制的行为

| schema 字段 | 作用位置 | 实际作用 |
|---|---|---|
| `feature_mapping` | Delta、Normalize、Compose | 指定标准 state/action 由哪些 key 构成 |
| `image_mapping` | Resize、Remap | 把数据源真实相机 key 映射到统一 image key |
| `action_mask_spec` | DeltaActionTransformFn | 正数段做 delta，负数段保留 absolute |
| `action_reorder` | ReorderStateActionTransform | action 从原始右先顺序复制到模型左先位置 |
| `state_reorder` | ReorderStateActionTransform | state 执行相同左右手交换 |
| `action_mode` | ChatProcessor | 在 prompt 中写 `<joint>` 或 `<end_effector>` |
| `action_layout` | SyntheticEmbodimentDataset | 在 prompt 中写动作语义块名称和大小 |
| `state_dim/action_dim` | adapter 初始化 | 与 annotation 原始有效维数对应 |

`action_mode: joint` 本身不决定是否计算 delta。是否存在 Delta transform 由 Dataset 配置控制；具体哪些维计算差分由 `action_mask_spec` 控制。

### 3.3 相机 key

两类 source schema 的主要差异是相机名：

```text
EgoDex:   observation.image
           -> observation.images.image0

Egoverse: observation.images.front
           -> observation.images.image0
```

本体 schema 也分别重复各自 source 的相机映射，因为图像 transform 是按随机抽到的本体 schema hydrate 的。

### 3.4 左右手 reorder

原始 annotation 都是右侧在前、左侧在后；模型输入统一为左侧在前、右侧在后：

| 本体 | 原始有效维 | 原始布局 | Reorder 后有效索引 |
|---|---:|---|---|
| `human_eef` | 16 | right 8 + left 8 | left `[0:8]`, right `[8:16]` |
| `piper` | 16 | right 8 + left 8 | left `[0:8]`, right `[8:16]` |
| `piper_inspire` | 40 | right 20 + left 20 | left `[0:20]`, right `[20:40]` |
| `franka` | 18 | right 9 + left 9 | left `[0:9]`, right `[9:18]` |
| `arx_l5` | 16 | right 8 + left 8 | left `[0:8]`, right `[8:16]` |
| `xarm7` | 14 | right 7 + left 7 | left `[0:7]`, right `[7:14]` |
| `rm75` | 14 | right 7 + left 7 | left `[0:7]`, right `[7:14]` |
| `ur7e` | 12 | right 6 + left 6 | left `[0:6]`, right `[6:12]` |
| `ur7e_wuji` | 52 | right 26 + left 26 | left `[0:26]`, right `[26:52]` |

[`test_source_schemas_share_action_layout_and_reorder`](../tests/test_synthetic_embodiment_dataset.py) 用递增向量检查 EgoDex/Egoverse 的同名本体 mask、layout、state reorder 和 action reorder 是否一致，并检查左半/右半目标位置。

## 4. 归一化统计脚本

入口是
[`compute_synthetic_embodiment_stats.py`](../util_scripts/compute_synthetic_embodiment_stats.py)。

### 4.1 输入和并行

- [`load_dataset_roots`](../util_scripts/compute_synthetic_embodiment_stats.py#L30) 读取 TXT，一行一个 LeRobotV3 根目录。
- [`compute_embodiment_stats`](../util_scripts/compute_synthetic_embodiment_stats.py#L162) 把每个 root 作为一个 job；`num_workers>1` 时使用 `spawn` 多进程。
- [`_compute_one_root`](../util_scripts/compute_synthetic_embodiment_stats.py#L115) 读取该 root 的 source `robot_type`，选择对应前缀和本体 schema。
- worker 使用 Welford 累积 `count/mean/M2/min/max`；主进程合并累积量，不拼接全部数据。
- [`write_embodiment_stats`](../util_scripts/compute_synthetic_embodiment_stats.py#L213) 写入 `<stats_dir>/<embodiment>/stats.json`。

### 4.2 state 和 action 分别统计什么

对每个本体，统计脚本先使用与训练相同的完整有效窗口。然后：

- state 统计加入每个有效 anchor 的绝对 `q[t]`；
- action 统计加入每个 chunk 中实际用于训练的 `H` 个目标；
- `--action-mode delta` 时 action 先调用仓库原始 `DeltaActionTransformFn`；`--action-mode abs` 时保持原值；
- 哪些维做 `q[t+k]-q[t]` 由本体 schema 的 `action_mask_spec` 决定；
- 负数段保留 `q[t+k]` 绝对值；
- `human_eef` 使用 `[-16]`，16 维全部保持 absolute；
- 统计阶段不执行 Normalize、Pad 或 Reorder。

所以统计文件的维度顺序仍是原始右先顺序。这与训练时“先 Normalize、后 Reorder”完全对应。

`--action-mode` 和 `--chunk-size` 都必须与训练配置相同。否则动作表示或无效帧附近能够进入统计的 anchor 集合不同，最终 mean/std 会错误。

### 4.3 多数据源合并统计

EgoDex 和 Egoverse 可以写入同一套 `<embodiment>/stats.json`，因为脚本会根据每个 root 的 source
`robot_type` 选择各自 schema，再合并同名本体累积量。

代码不会自动变换不同数据源的单位或坐标系。只有在同名本体的关节含义、单位、正方向和坐标约定完全一致时，合并统计才有物理意义。

## 5. 训练时加载归一化文件

加载过程是：

1. [`make_dataset`](../src/lerobot/datasets/factory.py#L495) 确定 `stats_dir`。
2. adapter 初始化时调用
   [`load_embodiment_stats`](../src/lerobot/datasets/synthetic_embodiment.py)，读取
   `<stats_dir>/<name>/stats.json` 的 state/action mean/std。
3. adapter 为每个本体创建独立 metadata view，其中 `robot_type` 是具体本体 schema 名，stats 是该本体统计量。
4. `DeltaActionTransformFn.hydrate(view)` 取得该本体 delta mask。
5. `NormalizeTransformFn.hydrate(view)` 取得该本体 state/action key 和 mean/std。
6. 每次随机抽到哪个本体，就运行哪个本体已 hydrate 的 transform 组合。

stats 文件只按本体名组织；加载代码不使用额外 manifest 或 schema 名称校验来源。

### 5.1 真实 transform 顺序

通用 InternVLA-A1.5 transform 列表见
[`configuration_internvla_a1_5.py`](../src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py)。

专用 adapter 会从列表中取出四个仓库原 transform，并重新组合为：

```text
DeltaActionTransformFn
NormalizeTransformFn
PadStateAndActionTransformFn
ReorderStateActionTransform
```

对应代码位于
[`SyntheticEmbodimentDataset.__init__`](../src/lerobot/datasets/synthetic_embodiment.py)。原 transform 实现位于：

- [`DeltaActionTransformFn`](../src/lerobot/transforms/core.py#L412)
- [`NormalizeTransformFn`](../src/lerobot/transforms/core.py#L273)
- [`PadStateAndActionTransformFn`](../src/lerobot/transforms/core.py#L97)
- [`ReorderStateActionTransform`](../src/lerobot/transforms/core.py#L453)

假设原始第 `j` 维统计量为 `μj, σj`：

```text
state_norm[j]  = (q[t,j] - state_mean[j]) / (state_std[j] + 1e-6)
action_norm[j] = (delta_or_absolute[j] - action_mean[j]) / (action_std[j] + 1e-6)
```

之后 Reorder 创建与 Pad 后 tensor 同形状的零张量，再执行 schema 的 `src -> dst` 复制。

## 6. 图像处理与图像归一化

### 6.1 图像数据流

```text
mp4
 -> LeRobotDataset._query_videos
 -> source camera tensor [3,H,W], 范围 [0,1]
 -> ResizeImagesWithPadFn
 -> RemapImageKeyTransformFn
 -> observation.images.image0, mask=True
 -> InternVLAA15ChatProcessorTransformFn
 -> observation.pixel_values + image_grid_thw
```

- [`ResizeImagesWithPadFn`](../src/lerobot/transforms/core.py#L131) 保持宽高比缩放，用 0 填充边缘并 clamp 到 `[0,1]`。
- [`RemapImageKeyTransformFn`](../src/lerobot/transforms/core.py#L239) 把真实相机统一为 `image0`，并创建 mask 为 false 的 `image1/image2` 占位图。
- [`InternVLAA15ChatProcessorTransformFn`](../src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py#L55) 只收集 mask 为真的图像，false mask 占位图不会送给 Qwen processor。
- processor 使用 `do_rescale=False`，因为视频解码结果已经是 `[0,1]`。

### 6.2 为什么统计脚本不计算图像 mean/std

合成本体 stats 只计算 state/action 是正确的。图像使用 Qwen3.5 processor 和视觉模型自带的固定预处理，不使用机器人本体的 `stats.json`。不应把 EgoDex/Egoverse 像素统计写入本体统计文件。

### 6.3 当前黑图风险

[`LeRobotDataset._query_videos`](../src/lerobot/datasets/lerobot_dataset.py#L991) 捕获任意视频解码异常后，会：

1. 记录 `[video_decode_error]`；
2. 创建 `torch.zeros(...)` 图像；
3. 记录 `[video_decode_fallback]`；
4. 继续训练。

因此当前确实存在“视频解码失败后整张输入变为黑图”的兜底路径。它不是 schema remap 造成的；相机 key 错误通常会直接 KeyError。检查服务器日志时必须搜索：

```bash
grep -R "video_decode_error\|video_decode_fallback" <训练日志目录>
```

另外要区分以下正常零值：

- Resize 产生的 letterbox 黑边；
- false mask 的占位相机；
- `action_loss_only=true` 时 Unify 创建的零 `observation.video_frames`；
- state/action Pad 后的尾部零维。

这些都不等于 VLM 当前帧是黑图。

## 7. prompt 处理

adapter 在姿态 transform 之后修改 task，所以 ChatProcessor 看到的数值 state 已经 Normalize、Pad、Reorder。prompt 结构包含：

```text
Task: Current robot embodiment: <name>;
      task: <原任务>;
      action layout: <语义块名称>[<大小>], ...;
Control Mode: <joint/end_effector>;
State: <离散 state>
```

当前 action layout 只表示语义块大小。例如 xArm7 prompt 是：

```text
left_arm[7], right_arm[7]
```

而真实 tensor 中右臂从索引 8 开始，中间索引 7 是零槽。机器人本体名仍能区分不同动作空间，但仅从 prompt 文本无法恢复所有 padding/gap 位置。

[`InternVLAA15ChatProcessorTransformFn._encode_state`](../src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py) 的当前算法是：

```python
state_np = normalized_state / 3
bins = np.digitize(state_np, np.linspace(-1, 1, 257)[:-1]) - 1
```

其意图是把大约 `[-3σ,3σ]` 映射到离散桶。但没有 clip 时：

- 小于 `-3σ` 的值可能得到 `-1`；
- 大于上界的值进入最高桶；
- state 中的 NaN/Inf 没有在 Dataset 层显式拦截。

现有代码会把这些结果作为文本写入 prompt，并不会自动检查。

## 8. 最终 batch 和 forward

[`UnifyInternVLAA15InputsTransformFn`](../src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py) 最终保留：

```text
observation.state
action
observation.pixel_values
observation.image_grid_thw
observation.input_ids
observation.attention_mask
observation.fast_token_mask
VQA.labels
vqa_type
label_mode
observation.video_frames
```

[`_multimodal_collate`](../src/lerobot/datasets/factory.py#L64) 对普通张量使用 `default_collate`，对视觉 patch 和 grid 使用 `torch.cat`。

典型形状是：

```text
state:       [B, max_state_dim]
action:      [B, H, max_action_dim]
input_ids:   [B, max_prompt_length]
pixel_values:[所有样本视觉 patch 的总数, patch_dim]
grid_thw:    [图像总数, 3]
```

[`InternVLAA15Policy.forward`](../src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py#L1572) 读取这些键。内部训练 forward 位于同文件约第 1099 行：

```text
x_t = t * noise + (1 - t) * action
u_t = noise - action
```

图像与 prompt 构成 VLM prefix，state、带噪 action 和时间构成 expert suffix，动作头预测 `v_t`，最终计算
`MSE(u_t, v_t)`。

当前 Pad 后未使用的动作维也是零，并且没有逐本体的维度 loss mask；这些 padding 维同样参与 MSE，模型会学习输出零。这不是左右手错位，但属于当前多本体共享固定维度的训练行为。

## 9. 当前能够检查到什么

### 9.1 已有每 200 step 日志

[`log_synthetic_input`](../src/lerobot/scripts/lerobot_train.py) 每 200 个更新在主进程打印当前 batch 第一个样本的：

- token 解码后的 prompt；
- Normalize、Pad、Reorder 后的 state；
- Delta/absolute、Normalize、Pad、Reorder 后的完整 action chunk。

它不能覆盖：

- 每一个 batch；
- 一个 batch 中除第一个以外的样本；
- 其它训练 rank；
- 图像像素内容。

所以目前不能声称代码已经检查了“每一个 batch 的图像、state、action、prompt”。README 能完成静态链路审计，实际全量检查仍需要后续单独实现或在服务器数据上运行检查脚本。

### 9.2 服务器日志人工检查

每次出现 `[synthetic input]` 时，应检查：

1. prompt 是否包含正确 `Current robot embodiment`。
2. `Control Mode` 是否与本体 schema 一致。
3. prompt state 是否出现 `-1`、异常长数字或明显截断。
4. state/action 是否包含 `nan`、`inf` 或极端大值。
5. 左侧段是否在前、右侧段是否在后。
6. action 尾部 padding 是否为 0。
7. 视频日志是否出现 `video_decode_fallback`。

可以搜索：

```bash
grep -n "\[synthetic input\]\|video_decode_error\|video_decode_fallback" train.log
```

## 10. 不启动训练的检查

```bash
cd /path/to/InternVLA-A-series
export PYTHONPATH="$PWD/src"

bash -n launch/internvla_a1.5_pretrain_egodex.sh

python -m pytest \
  tests/test_synthetic_embodiment_dataset.py \
  tests/test_synthetic_embodiment_stats_paths.py -q
```

这些测试覆盖：

- 完整有效 chunk 过滤；
- `action[t]=state[t+1]`；
- `human_eef` 双手 valid；
- delta mask；
- Normalize/Pad/Reorder 顺序；
- 左右手 reorder；
- EgoDex/Egoverse schema 一致性；
- 共享 stats；
- 多进程与串行统计一致性。

测试不会加载训练模型或启动训练。

## 11. 审计边界

静态代码和单元测试可以验证索引、transform 顺序、schema mask 和左右手复制位置，但不能仅靠代码证明两个数据源的物理坐标定义确实相同。

例如两个数据源都标为 `franka`，仍可能存在不同角度单位、关节正方向、世界/相机坐标系或手部镜像约定。这种错误通常不会产生 NaN，合并统计文件也能正常计算，但训练语义是错的。最终仍需抽取真实样本，把原始右/左向量、Normalize 后向量、Reorder 后向量与合成工具定义逐维核对。
