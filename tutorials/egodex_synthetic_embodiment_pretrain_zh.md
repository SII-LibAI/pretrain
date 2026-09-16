# EgoDex 合成本体数据接入 InternVLA-A1.5

本文说明 `synthetic_embodiment` Dataset 类型的目录发现、本体选择、delta 动作、归一化、schema 和检查方法。本实现没有修改 policy、模型 forward 或 loss。

## 1. 数据流

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

`SyntheticEmbodimentDataset` 自身不会扫描父目录。这样目录发现方式、抽样数量和数据源组合都留在启动脚本中，后续接入其他数据源时不需要修改 Dataset。

## 2. 启动脚本发现子数据集

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

## 3. 样本、本体和有效帧

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

传入空列表 `[]` 时使用统计文件中列出的全部本体。

## 4. Delta 动作与归一化

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

预计算命令：

```bash
export PYTHONPATH="$PWD/src"

python tools/compute_synthetic_embodiment_stats.py \
  --dataset-root /data/Egodex-LerobotV3/extra \
  --chunk-size 50 \
  --num-workers 8
```

统计工具参考 `util_scripts/compute_norm_stats_multi.py`，采用 `spawn` 多进程、每个子数据集一个任务、主进程合并 Welford 累积量；`--num-workers 1` 可做串行检查，`--batch-size` 控制每批 anchor 数量。离线工具可以递归扫描父目录；这个扫描只发生在预处理脚本中，不发生在训练 Dataset 中。只计算部分本体时重复传入 `--embodiment`：

```bash
python tools/compute_synthetic_embodiment_stats.py \
  --dataset-root /data/Egodex-LerobotV3/extra \
  --chunk-size 50 \
  --embodiment franka \
  --embodiment rm75
```

默认输出到 `<dataset-root>/synthetic_embodiment_stats/`，包含 `manifest.json` 和每个本体的 `<embodiment>/stats.json`。`manifest.json` 记录 `action_mode=delta` 和 `chunk_size`。训练启动时会检查两者是否与当前配置一致；改变 chunk size 后需要重新计算统计量。

## 5. 每个本体的 schema 与 reorder

schema 位于 `src/lerobot/dataset_schemas/configs/egodex_v.yaml`。`egodex_v` 只描述源视频数据，七个合成本体各自拥有独立 schema：

| annotation key | schema | 原始维度 | action layout |
|---|---|---:|---|
| `piper_inspire` | `egodex_synthetic_piper_inspire` | 40 | left arm 6 + left hand 14 + right arm 6 + right hand 14 |
| `franka` | `egodex_synthetic_franka` | 18 | left arm 7 + left gripper 2 + right arm 7 + right gripper 2 |
| `arx_l5` | `egodex_synthetic_arx_l5` | 16 | left arm 6 + left gripper 2 + right arm 6 + right gripper 2 |
| `xarm7` | `egodex_synthetic_xarm7` | 14 | left arm 7 + right arm 7 |
| `rm75` | `egodex_synthetic_rm75` | 14 | left arm 7 + right arm 7 |
| `ur7e` | `egodex_synthetic_ur7e` | 12 | left arm 6 + right arm 6 |
| `ur7e_wuji` | `egodex_synthetic_ur7e_wuji` | 52 | left arm 6 + left hand 20 + right arm 6 + right hand 20 |

每个本体都显式定义了 `action_reorder` 和 `state_reorder`，格式与仓库已有 schema 相同：

```yaml
action_reorder:
  - [src_start, src_end, dst_start, dst_end]
state_reorder:
  - [src_start, src_end, dst_start, dst_end]
```

当前 EgoDex parquet 的原始 state/action 向量均为右侧在前、左侧在后；训练时统一为左侧在前、右侧在后。每个 schema 的 `state_reorder` 和 `action_reorder` 都交换左右两半，同时保留各侧内部的“arm → hand/gripper”顺序。例如 Franka 原始 `[right_arm(7), right_gripper(2), left_arm(7), left_gripper(2)]` 会变成 `[left_arm(7), left_gripper(2), right_arm(7), right_gripper(2)]`。原有 transform 在原始顺序上完成 delta 和归一化，再用原有 `ReorderStateActionTransform` 交换左右；它被提前到提示词处理之前，保证数值 state 与提示词里的 state token 顺序一致。仅改变左右顺序不需要重算已有统计文件。以后某个本体的源向量顺序变化时，可修改对应 schema 的 src/dst 区间，不需要修改 Dataset 或 policy。

## 6. 模型维度由用户指定

代码不会根据本体 schema 或统计文件自动修改模型维度。启动时显式传入：

```bash
--policy.max_state_dim=52
--policy.max_action_dim=52
```

这两个值必须不小于所选本体需要的维度。设置过小时 adapter 会直接报错，并显示数据维度和配置维度。

专用启动脚本使用环境变量传递：

```bash
MAX_STATE_DIM=52 MAX_ACTION_DIM=52 \
EMBODIMENTS='["franka","rm75"]' \
DATASET_ROOT=/data/Egodex-LerobotV3/extra \
bash launch/internvla_a1.5_pretrain_egodex.sh
```

首次只跑一个子数据集可设置 `SMOKE_NUM_DATASETS=1`。启动脚本会真正启动训练；在当前不支持训练的机器上只做下面的检查，不要执行启动示例。

## 7. 不启动训练的检查

```bash
bash -n launch/internvla_a1.5_pretrain_egodex.sh

PYTHONPATH=src python -m pytest \
  tests/test_synthetic_embodiment_dataset.py -q
```

重点人工检查：

1. `manifest.json` 中 `action_mode` 是 `delta`，`chunk_size` 与训练一致。
2. 每个 `stats.json` 的 `observation.state` 和 `action` 统计是分开计算的。
3. `dataset.repo_id` 由启动脚本生成，不是 Dataset 内部发现。
4. `policy.max_state_dim` 和 `policy.max_action_dim` 是显式启动参数。
5. schema 中每个本体均有 `action_mask_spec`、`action_reorder` 和 `state_reorder`。
6. policy 目录没有本次功能修改。

## 8. 代码位置

- `src/lerobot/datasets/synthetic_embodiment.py`：训练时的 adapter、有效窗口、原有变换的调度和统计文件读取；
- `src/lerobot/datasets/configuration_synthetic_embodiment.py`：独立 `synthetic_embodiment` DatasetConfig；
- `src/lerobot/datasets/factory.py`：接收脚本提供的 repo_id 并创建 adapter；
- `src/lerobot/dataset_schemas/configs/egodex_v.yaml`：源数据和逐本体 schema；
- `tools/compute_synthetic_embodiment_stats.py`：独立的离线 delta 归一化计算与文件写入；
- `launch/internvla_a1.5_pretrain_egodex.sh`：目录发现和训练参数入口；
- `tests/test_synthetic_embodiment_dataset.py`：不加载模型的独立测试。
