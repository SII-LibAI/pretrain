# EgoDex / Egoverse 合成本体数据接入 InternVLA-A1.5

本文说明 `synthetic_embodiment` Dataset 类型的目录发现、本体选择、delta 动作、归一化、schema 和检查方法。EgoDex 与 Egoverse 可以混合加载，共用按本体计算的统计文件。本实现没有修改 policy、模型 forward 或 loss。

## 1. EgoDex + Egoverse：按顺序启动

以下命令在训练服务器、仓库根目录执行；把示例路径换成实际路径。最后一步会真正启动训练，当前不能训练的机器不要执行。

### 1.1 确认数据和 schema

两种 LeRobotV3 子数据集都应有 `data/`、`videos/`、`meta/info.json` 和 `meta/annotations/action.parquet`。EgoDex 的 `meta/info.json` 应写 `robot_type: egodex_v`，Egoverse 应写 `robot_type: egoverse_human_bimanual`；其源图像键分别是 `observation.image` 与 `observation.images.front`。训练所选的每个本体必须同时出现在两种数据的 `action.<本体>` 和 `action.<本体>_valid` 列中。当前 EgoDex schema 还包含 Egoverse schema 没有的 `piper`，因此混合训练不要选 `piper`，除非也为 Egoverse 补齐标注和 schema。

先确认服务器上两份 YAML 都已注册。注册器遇到 YAML 错误会跳过整份文件；例如重复的 `&..._reorder` 锚点会让其下所有本体都显示为“schema 未注册”。

```bash
cd /path/to/InternVLA-A-series
export PYTHONPATH="$PWD/src"
python - <<'PY'
from lerobot.dataset_schemas import get_registry, get_schema
names = set(get_registry().list_available())
expected = {
    "egodex_v", "egoverse_human_bimanual",
    "egodex_synthetic_franka", "egoverse_synthetic_franka",
    "egodex_synthetic_ur7e", "egoverse_synthetic_ur7e",
}
missing = expected - names
assert not missing, f"Missing schemas: {sorted(missing)}"
for body in ("franka", "ur7e"):
    a = get_schema(f"egodex_synthetic_{body}")
    b = get_schema(f"egoverse_synthetic_{body}")
    assert (a.state_dim, a.action_dim, a.action_mask_spec) == (
        b.state_dim, b.action_dim, b.action_mask_spec
    )
print("Source and embodiment schemas registered")
PY
```

### 1.2 准备共同父目录

启动脚本只扫描一个 `DATASET_ROOT`，用目录软链接把两个原始父目录放在其下；不必复制视频和 parquet。`find -L` 会跟随链接。以下是示例：

```bash
EGODEX_PARENT=/path/to/Egodex-LerobotV3/extra
EGOVERSE_PARENT=/path/to/Egoverse-LerobotV3/extra
MIX_ROOT=/data/synthetic_pretrain
mkdir -p "$MIX_ROOT"
ln -s "$EGODEX_PARENT" "$MIX_ROOT/egodex"
ln -s "$EGOVERSE_PARENT" "$MIX_ROOT/egoverse"
find -L "/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series/data/ego" -type f -path '*/meta/annotations/action.parquet' | head
```

预期相对路径类似 `egodex/knit_scarf`、`egoverse/task_b`。脚本会将它们作为 `dataset.repo_id`，Dataset 本身不递归扫描。当前脚本用空格分隔 repo_id，因此子数据集的相对路径不要包含空格。不要把同一个真实 LeRobot 根目录同时以原路径和链接路径重复加入训练集合。

### 1.3 准备合并的归一化文件

已经算好两数据源合并统计量的，可以跳过计算，直接令 `STATS_DIR` 指向该目录。目录至少应有每个所选本体的 `<本体>/stats.json`；不需要 `manifest.json`。如果尚未计算，先分别对两个**真实父目录**运行 `discover_lerobot_root.py`，再合并两个 TXT；不要依赖该 Python 发现脚本跟随软链接。

```bash

python /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series/util_scripts/compute_synthetic_embodiment_stats.py \
  --dataset-roots-file /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series/outputs/repos/human.txt \
  --output-dir /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series/stats/humandata  \
  --chunk-size 50 \
  --num-workers 55 \
  --embodiment arx_l5 \
  --embodiment franka \
  --embodiment human_eef \
  --embodiment ur7e \
  --embodiment rm75 \
  --embodiment piper
```
### 1.4 核对启动参数并训练


确认 `VLM_MODEL_PATH` 有效；另外 ChatProcessor 的 `pretrained_model_name_or_path` 在其 transform 配置中有独立默认值，若使用非默认 Qwen3.5 路径，也要确保该处理器配置指向可用的同款模型。确认本体统计文件存在后再启动：

```bash
test -f "$MIX_ROOT/synthetic_embodiment_stats/franka/stats.json"
test -f "$MIX_ROOT/synthetic_embodiment_stats/ur7e/stats.json"

DATASET_ROOT="$MIX_ROOT" \
STATS_DIR="$MIX_ROOT/synthetic_embodiment_stats" \
EMBODIMENTS='["franka","ur7e"]' \
CHUNK_SIZE=50 \
VLM_MODEL_PATH=/path/to/qwen3.5-2b \
PROC_PER_NODE=2 \
WANDB_MODE=offline \
bash launch/internvla_a1.5_pretrain_egodex.sh
```

脚本打印的 `DATASET_REPO_ID` 应同时包含 `egodex/...` 与 `egoverse/...`。`SMOKE_NUM_DATASETS` 会在随机打乱后截取若干数据集，不能保证同时抽到两种来源；检查混合训练时不要仅凭它判断两个来源都已加载。代码没有显式配置两来源 50:50，实际比例受各进程分配的数据集与可训练样本数影响。

## 2. 数据流

运行时的数据流是：

```text
启动脚本扫描父目录
  -> 将相对路径列表传给 dataset.repo_id
  -> factory 为每个 repo_id 创建 LeRobotDataset
  -> SyntheticEmbodimentDataset 读取该子数据集的 action.parquet
  -> 当前样本的有效本体中随机选择一个
  -> 构造 delta action chunk 并使用该本体的统计量归一化
  -> padding 到用户指定的 policy 维度
  -> 按该本体 schema 执行 reorder，再进入提示词处理和模型
```

`SyntheticEmbodimentDataset` 自身不会扫描父目录。启动脚本的 `find -L` 可以跟随目录软链接，从同一父目录发现两个数据源；每个子数据集按自身的 `robot_type` 选择 schema。

## 3. 启动脚本发现子数据集

专用脚本是 `launch/internvla_a1.5_pretrain_egodex.sh`，它参考 `launch/internvla_a1.5_pretrain_interndata.sh`，搜索 `DATASET_ROOT` 下的 `data` 目录，并要求同一数据根目录同时存在：

```text
<child_dataset>/data/
<child_dataset>/meta/
<child_dataset>/videos/
<child_dataset>/meta/annotations/action.parquet
```

脚本把匹配到的子数据集转换成相对 `DATASET_ROOT` 的路径，用空格拼接后传给：

```bash
--dataset.root="${DATASET_ROOT}"
--dataset.repo_id="${DATASET_REPO_ID}"
```

例如父目录包含 `knit_scarf/` 和 `task_b/` 两个数据集时，传入 Dataset 的 `repo_id` 是 `knit_scarf task_b`，而不是由 Dataset 再递归搜索。

## 4. 样本、本体和有效帧

annotation 第 `t` 行的合成本体姿态记为 `q[t]`，action chunk 长度为 `H`：

```text
state_raw        = q[t]
action_chunk_raw = [q[t+1], q[t+2], ..., q[t+H]]
```

因此原始标注满足 `action[t] = state[t+1]`，窗口不会跨 episode。对于每个 anchor，Dataset 分别检查每个选定本体：

```text
valid(q[t]) AND valid(q[t+1]) ... AND valid(q[t+H])
```

然后从拥有完整有效窗口的本体中均匀随机选择一个。没有任何本体拥有完整有效窗口时，才从 Dataset 索引中删除该 anchor。这里通过过滤完整窗口来屏蔽无效 action 帧，因为当前 policy/loss 没有消费逐时间步 loss mask；不需要修改 policy。

选定本体后，adapter 会把本体名称放在 `task` 文本最前面，例如 `Current robot embodiment: franka; task: fold the cloth; action layout: ...`。现有 ChatProcessor 使用 `task` 构造用户提示词，所以同一视频帧即使抽到不同本体，模型看到的提示词也不同；本体名靠前可降低被长提示词截断的风险。

训练时可用 `dataset.embodiments` 限定本体：

```bash
--dataset.embodiments='["franka","rm75","ur7e"]'
```

传入空列表 `[]` 时使用标注 parquet 中的全部本体。

## 5. Delta 动作与归一化

`synthetic_embodiment` DatasetConfig 默认 `action_mode=delta`，启动脚本无需重复传参。每个 schema 的 `action_mask_spec` 与仓库其他机器人含义相同：正数段使用 delta，负数段保留绝对目标。例如 Franka：

```yaml
action_mask_spec: [7, -2, 7, -2]
```

表示左右臂各 7 维计算 `q[t+k] - q[t]`，两个夹爪各 2 维仍使用 `q[t+k]`。处理顺序是：

```text
原始 q -> 按 action_mask_spec 计算 delta -> 按本体归一化 -> padding -> reorder
```

adapter 不再手写 delta、归一化或 reorder。它使用仓库原有的 `DeltaActionTransformFn`、`NormalizeTransformFn`、`PadStateAndActionTransformFn` 和 `ReorderStateActionTransform`，按此顺序、按选中本体的 schema 与统计量执行一次，再进入提示词处理。离线统计也使用原有 `DeltaActionTransformFn` 计算 action 统计量。

归一化文件中的 state 和 action 统计量不是同一份：

- state：所有可训练 anchor 的绝对 `q[t]`；
- action：所有可训练 action chunk 中实际送入模型的 delta/absolute 混合向量；
- 只统计完整有效窗口，与训练 Dataset 的过滤条件一致。

预计算分两步。以下命令是**仅 EgoDex** 的示例；两数据源混合统计应按 1.3 节分别发现并合并 TXT。先递归扫描父目录，生成一行一个 LeRobotV3 子数据集**绝对路径**的 TXT；可以手工删行来控制参与统计的数据集：

```bash
python util_scripts/discover_lerobot_root.py \
  --parent-dir /data/Egodex-LerobotV3/extra \
  --output-file /data/Egodex-LerobotV3/extra/dataset_roots.txt
```

再检查 TXT 内容并计算归一化统计：

```bash
export PYTHONPATH="$PWD/src"

python util_scripts/compute_synthetic_embodiment_stats.py \
  --dataset-roots-file /data/Egodex-LerobotV3/extra/dataset_roots.txt \
  --chunk-size 50 \
  --num-workers 8
```

统计工具只读取 TXT，不递归发现目录。TXT 可包含空行和 `#` 注释；相对路径以 TXT 所在目录为基准，同一数据集目录不能重复。统计工具参考 `util_scripts/compute_norm_stats_multi.py`，采用 `spawn` 多进程、每个子数据集一个任务、主进程合并 Welford 累积量；`--num-workers 1` 可做串行检查，`--batch-size` 控制每批 anchor 数量。混合 EgoDex/Egoverse 时，当前两份 schema 的本体集合不同，必须用重复的 `--embodiment` 只选择两边共有的本体：

```bash
python util_scripts/compute_synthetic_embodiment_stats.py \
  --dataset-roots-file /data/Egodex-LerobotV3/extra/dataset_roots.txt \
  --chunk-size 50 \
  --embodiment franka \
  --embodiment rm75
```

默认输出到 TXT 旁的 `synthetic_embodiment_stats/`，也可通过 `--output-dir` 指定；只生成每个本体的 `<embodiment>/stats.json`，不生成或校验 manifest。TXT 可以同时列出 EgoDex 和 Egoverse 子数据集，统计脚本根据各自 `meta/info.json` 的 `robot_type` 选择 schema，并合并同名本体的统计量。计算与训练需自行保持相同的 `chunk_size`、delta 规则和向量坐标约定。

## 6. 每个本体的 schema 与 reorder

schema 位于 `src/lerobot/dataset_schemas/configs/egodex_v.yaml`。`egodex_v` 只描述源视频数据，当前 EgoDex 的合成本体各自拥有独立 schema：

| annotation key | schema | 原始维度 | action layout |
|---|---|---:|---|
| `piper` | `egodex_synthetic_piper` | 16 | left arm 6 + left gripper 2 + right arm 6 + right gripper 2 |
| `piper_inspire` | `egodex_synthetic_piper_inspire` | 40 | left arm 6 + left hand 14 + right arm 6 + right hand 14 |
| `franka` | `egodex_synthetic_franka` | 18 | left arm 7 + left gripper 2 + right arm 7 + right gripper 2 |
| `arx_l5` | `egodex_synthetic_arx_l5` | 16 | left arm 6 + left gripper 2 + right arm 6 + right gripper 2 |
| `xarm7` | `egodex_synthetic_xarm7` | 14 | left arm 7 + right arm 7 |
| `rm75` | `egodex_synthetic_rm75` | 14 | left arm 7 + right arm 7 |
| `ur7e` | `egodex_synthetic_ur7e` | 12 | left arm 6 + right arm 6 |
| `ur7e_wuji` | `egodex_synthetic_ur7e_wuji` | 52 | left arm 6 + left hand 20 + right arm 6 + right hand 20 |

Egoverse 使用独立的完整 schema 文件 `src/lerobot/dataset_schemas/configs/egoverse_human_bimanual.yaml`。源 `robot_type` 是 `egoverse_human_bimanual`，相机键 `observation.images.front` 映射到 `observation.images.image0`，本体 schema 名称是 `egoverse_synthetic_<embodiment>`。当前两份 schema 对共有的同名本体采用相同的维度、delta mask 和左右手 reorder，但相机映射各自独立。增加同格式数据源时，新增一份完整 schema，并在 `EMBODIMENT_SCHEMA_PREFIXES` 中登记源 `robot_type` 与本体 schema 前缀。

两个数据源共用同一份 `<embodiment>/stats.json`。前提是合成后的同名本体向量已统一维度含义、单位、关节正方向及坐标约定；代码不会自动变换坐标系。将合并统计目录放在共同父目录下，或显式设置 `STATS_DIR`。

每个本体都显式定义了 `action_reorder` 和 `state_reorder`，格式与仓库已有 schema 相同：

```yaml
action_reorder:
  - [src_start, src_end, dst_start, dst_end]
state_reorder:
  - [src_start, src_end, dst_start, dst_end]
```

当前 EgoDex parquet 的原始 state/action 向量均为右侧在前、左侧在后；训练时统一为左侧在前、右侧在后。每个 schema 的 `state_reorder` 和 `action_reorder` 都交换左右两半，同时保留各侧内部的“arm → hand/gripper”顺序。例如 Franka 原始 `[right_arm(7), right_gripper(2), left_arm(7), left_gripper(2)]` 会变成 `[left_arm(7), left_gripper(2), right_arm(7), right_gripper(2)]`。原有 transform 在原始顺序上完成 delta 和归一化，再用原有 `ReorderStateActionTransform` 交换左右；它被提前到提示词处理之前，保证数值 state 与提示词里的 state token 顺序一致。仅改变左右顺序不需要重算已有统计文件。以后某个本体的源向量顺序变化时，可修改对应 schema 的 src/dst 区间，不需要修改 Dataset 或 policy。

## 7. 模型维度由用户指定

代码不会根据本体 schema 或统计文件自动修改模型维度。policy 默认 state/action 维度都是 32；选中的本体超过 32 维时，需在启动脚本的 `ARGS` 中显式加入，例如：

```bash
--policy.max_state_dim=52
--policy.max_action_dim=52
```

这两个值必须不小于所选本体需要的维度。设置过小时 adapter 会直接报错，并显示数据维度和配置维度。

完整的混合数据源启动命令见 1.4 节。首次只跑一个子数据集可设置 `SMOKE_NUM_DATASETS=1`，但它不能验证两个来源都已加载。启动脚本会真正启动训练；在当前不支持训练的机器上只做下面的检查，不要执行启动示例。

## 8. 一次训练前向：从索引到 loss

以下描述对应上面启动脚本的配置：`action_loss_only=true`、`enable_vqa_loss=false`、`tokenize_state=true`、`use_fast_action_tokens=false`。这里的“单次”指 DataLoader 取出一个 batch 后调用一次 `policy.forward(batch)`；batch 中每个样本仍会各自随机选择有效本体。

```text
启动时（每个进程一次）
  DATASET_ROOT --find -L--> 相对 repo_id 列表
  repo_id --> LeRobotDatasetMetadata(robot_type) --> EgoDex / Egoverse 源 schema
          --> LeRobotDataset(视频、任务、索引)
          --> SyntheticEmbodimentDataset(action.parquet、共享 stats.json、本体 schema)
          --> 预计算每个本体的完整有效窗口、base 行与 annotation 行映射
          --> MultiLeRobotDataset --> DataLoader

取一个样本
  全局索引 --> 子数据集及帧 t --> 可用本体中随机选一个
  原视频帧和 task + 合成 q[t]、q[t+1:t+H]
  --> Delta --> Normalize --> Pad --> Reorder
  --> 图像 Resize --> 相机键 Remap --> ChatProcessor --> Unify
  --> collate 成 batch

一次 forward
  图像 + task/本体名/数值 state 的文本 --> Qwen3.5 VLM prefix
  归一化 action chunk + 随机噪声/时间 --> action expert suffix
  prefix + suffix --> 共享注意力层 --> 速度预测 v_t
  v_t 与目标 (noise - action) --> flow-matching MSE --> loss
```

1. `make_dataset` 先按 `dataset.repo_id` 为每个子数据集创建 `LeRobotDataset`。源 `robot_type` 决定相机键以及本体 schema 前缀；两个来源的同名本体读取同一份 `STATS_DIR/<本体>/stats.json`。初始化 adapter 时读取标注，校验与 LeRobot 行索引的对应关系，并预计算候选窗口。`dataset.dist_loading=true` 时，不同训练进程分配子数据集；每个进程再由 DataLoader 抽样。
2. `__getitem__` 对帧 `t` 只保留满足 `valid(q[t]) ∧ ... ∧ valid(q[t+H])` 的本体，从中均匀随机选一个。`base_dataset[base_row]` 提供源图像和 task；其原始 `observation.state`、`action` 随即被合成标注的 `q[t]`、`[q[t+1],...,q[t+H]]` 覆盖。因此训练端没有逐步无效帧 mask：不完整的 chunk 在取样前已被排除。
3. 数值姿态先经过仓库原有的 `DeltaActionTransformFn`：schema mask 为真的关节维度做 `q[t+k]-q[t]`，其余维度保留 `q[t+k]`。`NormalizeTransformFn` 分别用该本体的 state/action `mean/std` 做标准化，再由 `PadStateAndActionTransformFn` 补到用户设置的模型维度，最后 `ReorderStateActionTransform` 将原始右手在前的布局换成训练使用的左手在前。归一化发生在 reorder **之前**，因此统计文件必须按原始向量顺序计算。schema 的 `action_mode: joint` 只决定提示词里的控制模式文字，delta 与否由 `dataset.action_mode` 和 `action_mask_spec` 决定。
4. 余下的图像/文本 transform 才运行：`ResizeImagesWithPadFn` 按所选本体 schema 中的源相机键缩放图像；`RemapImageKeyTransformFn` 将 EgoDex 的 `observation.image` 或 Egoverse 的 `observation.images.front` 统一成 `observation.images.image0`，其余两个视角用 false mask 占位。adapter 把 `Current robot embodiment: <name>` 和 action layout 放到 task 前部。`InternVLAA15ChatProcessorTransformFn` 把单帧图像、task、`Control Mode: <joint>` 和离散化的归一化 state 组织成 Qwen3.5 输入。当前是 action-only 配置，`image_delta_indices=[0]`，不读取用于视频损失的未来帧；`UnifyInternVLAA15InputsTransformFn` 最终输出统一的 state、action、图像/文本 token、mask 等键，供 batch collate 使用。
5. `policy.forward(batch)` 中，`action` 形状是 `[B,H,max_action_dim]`。模型采样噪声 `ε` 和时间 `t`，构造 `x_t=t·ε+(1−t)·action`，目标速度 `u_t=ε−action`。图像/提示词进入 VLM prefix，带噪 action 与时间进入 expert suffix，并通过共享层得到 action 速度预测。由于 `policy.tokenize_state=true`，state 已作为文本 token 输入，suffix 不再额外放 state projection token。
6. action expert 对预测速度与 `u_t` 计算逐元素 MSE，再聚合成 `loss_action`。完整无效 chunk 已过滤，但 padding 到 `max_action_dim` 的尾部零维没有单独的维度 loss mask，当前也参与 MSE。`enable_vqa_loss=false` 时 VQA 标签不参与损失；`action_loss_only=true` 时不构造 WAN 视频模型，视频损失为 0，因此本配置的总损失就是 action flow-matching loss。`lerobot_train.py` 随后执行反向传播、梯度裁剪、优化器和学习率调度器更新；这属于 forward 之后的训练步骤。

## 9. 不启动训练的检查

```bash
bash -n launch/internvla_a1.5_pretrain_egodex.sh

PYTHONPATH=src python -m pytest \
  tests/test_synthetic_embodiment_dataset.py -q
```

重点人工检查：

1. 计算统计量与训练使用相同的 `chunk_size`、delta 规则和向量坐标约定。
2. 每个 `stats.json` 的 `observation.state` 和 `action` 统计是分开计算的。
3. `dataset.repo_id` 由启动脚本生成，不是 Dataset 内部发现。
4. 所选本体的维度不超过 policy 默认的 32；若超过，则在脚本中显式设置 `policy.max_state_dim` 和 `policy.max_action_dim`。
5. schema 中每个本体均有 `action_mask_spec`、`action_reorder` 和 `state_reorder`。
6. policy 目录没有本次功能修改。

## 10. 代码位置

- `src/lerobot/datasets/synthetic_embodiment.py`：训练时的 adapter、有效窗口、原有变换的调度和统计文件读取；
- `src/lerobot/datasets/configuration_synthetic_embodiment.py`：独立 `synthetic_embodiment` DatasetConfig；
- `src/lerobot/datasets/factory.py`：接收脚本提供的 repo_id 并创建 adapter；
- `src/lerobot/dataset_schemas/configs/egodex_v.yaml`、`egoverse_human_bimanual.yaml`：两个源数据的完整逐本体 schema；
- `util_scripts/discover_lerobot_root.py`：递归发现子数据集并生成目录 TXT；
- `util_scripts/compute_synthetic_embodiment_stats.py`：读取目录 TXT，离线计算 delta 归一化并写入文件；
- `launch/internvla_a1.5_pretrain_egodex.sh`：目录发现和训练参数入口；
- `tests/test_synthetic_embodiment_dataset.py`：不加载模型的独立测试；
- `tests/test_synthetic_embodiment_stats_paths.py`：目录发现与 TXT 清单读取测试。
