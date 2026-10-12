# WSA1.5 数据流说明

本文说明当前 WSA1.5 的 LeRobotV3 数据入口、机器人/人类数据分流、补充标注格式、有效帧规则以及最终监督字段。当前阶段只整理数据流；`policy` 的网络实现不属于本次改动。

## 0. 如何阅读代码

建议从单个样本开始，再看多数据集训练。阅读顺序如下；先跟随表中的入口函数，遇到具体数据问题时再进入底层辅助函数。

| 顺序 | 文件与入口 | 需要确认的事情 |
|---|---|---|
| 1 | [`inspect_wsa1_5_dataflow.py`](../../../../util_scripts/inspect_wsa1_5_dataflow.py)：`main` → `build_dataflow_dataset` → `summarize_sample` | 一个数据集如何创建、一个随机样本最终有哪些字段与 shape |
| 2 | [`datasets/wsa1_5.py`](../../datasets/wsa1_5.py)：`is_wsa15_human_robot_type` → `resolve_wsa15_delta_timestamps` → `resolve_wsa15_video_key_mapping` | 如何分流 robot/human；哪些时间偏移被查询；逻辑视频键如何找到存储文件 |
| 3 | 同文件：`wrap_wsa15_dataset` | robot 包装为 `TransformedLeRobotDataset`；human 包装为 `SyntheticEmbodimentDataset` |
| 4 | [`synthetic_embodiment.py`](../../datasets/synthetic_embodiment.py)：`SyntheticEmbodimentDataset.__init__` → `__getitem__` | human 如何筛选有效窗口、随机选择 EEF/本体、替换 state/action |
| 5 | [`datasets/wsa1_5.py`](../../datasets/wsa1_5.py)：`build_wsa15_data_transforms` | 所有样本经过哪些转换，以及转换的严格执行顺序 |
| 6 | 同文件：`CanonicalStateActionTransformFn`、`LoadWSA15PhysicalAnnotationsFn`、`PrepareWSA15MotionTargetsFn` | 动作槽如何填写、subtask 如何按帧匹配、缺失目标如何产生有效位 |
| 7 | [`factory.py`](../../datasets/factory.py)：`make_dataset` → `_build_single_dataset` → `make_dataloader` | 多数据集如何复用以上链路、合并样本、随机抽帧和组成 batch |

### 0.1 先看两条调用链

纯数据审阅入口：

```text
launch/wsa1_5_test_dataflow.sh
  -> inspect_wsa1_5_dataflow.main
     -> build_dataflow_dataset
        -> 读取 LeRobotDatasetMetadata
        -> resolve_wsa15_delta_timestamps / resolve_wsa15_video_key_mapping
        -> 创建 LeRobotDataset
        -> build_wsa15_data_transforms
        -> wrap_wsa15_dataset
     -> 随机有效 sample
     -> summarize_sample
```

训练入口：

```text
factory.make_dataset
  -> 每个 repo: _build_single_dataset
     -> 同一套时间查询 / 视频映射 / wrap_wsa15_dataset
     -> 同一套 build_wsa15_data_transforms
        + 文本与图像 processor
        + UnifyWSA15InputsFn
  -> 多 repo: MultiLeRobotDataset
  -> make_dataloader
  -> batch
```

共享的基础转换与来源包装都定义在 `datasets/wsa1_5.py`。检查脚本保留解码后的 RGB、原始 subtask 文本和监督有效位；训练配置在这些转换之后添加 tokenizer 和最终 batch 字段整理。

### 0.2 区分“创建数据集”和“取一个样本”

这两部分职责不同，阅读时不要把它们混在一起：

- **创建时**：读 metadata，建立时间查询，扫描 human validity，加载 schema/stats，并调用各 transform 的 `hydrate(dataset)`，将数据源配置绑定到转换器。
- **取样时**：`__getitem__` 获取当前帧和动作目标；human 随机选择一种有效动作表示；随后依次执行各 transform 的 `__call__(sample)`。

`TransformedLeRobotDataset.from_base` 在创建时 hydrate，`TransformedLeRobotDataset.__getitem__` 在取样时执行转换。Human 则在创建时为每种动作表示分别 hydrate，在取样时执行所选表示的转换链。

### 0.3 Human 分支需要继续往下读的函数

在 `synthetic_embodiment.py` 中按以下依赖继续读：

```text
SyntheticEmbodimentDataset.__init__
  -> SyntheticActionTable.from_parquet
     -> discover_embodiments            # 找本体动作列和 human_eef 原生列
  -> load_embodiment_schemas / load_embodiment_stats
  -> _build_valid_frame_index
     -> _scan_candidate_masks           # 校验对齐、跨 episode、完整有效窗口
  -> 为每个本体 hydrate transforms

SyntheticEmbodimentDataset.__getitem__
  -> _rows[idx]                          # dataset 序号映射到源数据行
  -> 当前帧的 candidate embodiments
  -> 随机选择 body_id
  -> SyntheticActionTable.read_values    # 只读取所选表示的 t..t+H
  -> 当前 image，state=q[t]，action=q[t+1:t+H+1]
  -> 所选表示的 transforms
```

`_rows` 保存有效源帧；`_candidate_masks` 保存每帧可选动作表示的位集合。读取训练数据时只扫描索引/valid 列来建索引，动作向量在真正取样时按 parquet row group 读取。

### 0.4 跟踪一个样本时看哪些值

以 human `franka` 为例，先观察 `index/episode_index/frame_index` 与 `synthetic_embodiment`，再观察 schema 的 `action_reorder` 和 `action_layout`。其原始 18D 双臂动作会写入左右 joint/gripper 槽，得到 `[H,32]`，EEF 空槽的 `wsa.action_valid_mask` 为 false。

随后确认 subtask valid 与两组 motion valid，最后检查 batch 是否仍有这些有效位。归一化的数学细节集中在 `CanonicalStateActionTransformFn.__call__`；最初阅读只需掌握“原始值 → schema 重排/归一化 → 固定语义槽 + mask”即可。

当前监督来自 subtask 与两个 motion 视频流，已移除单独的 pose 读取模块。原始数据中的 `pose.parquet` 可以保留，它不参与这条加载链路。

## 1. 训练样本契约

每个样本必须提供：

| 用途 | 字段 | 是否必需 | 说明 |
|---|---|---:|---|
| 输入 | 当前 RGB image | 是 | 由 source schema 的 `image_mapping` 映射 |
| 输入 | `observation.state` | 是 | robot 来自主 parquet；human 由补充动作表示在时刻 `t` 的值替换 |
| 监督 | `action` | 是 | robot 来自主 parquet；human 来自补充 `action.parquet` |
| 监督 | subtask | 否 | 来自 `meta/sidecar.json`；缺失时关闭 subtask loss |
| 监督 | `motion_mask` | 否 | 未来 2D 图；缺失时关闭该 loss |
| 监督 | `motion_mask_ids` | 否 | 未来 3D/ID 图；缺失时关闭该 loss |

LeRobotV3 原生的 image、state、action 是数据集必须项。subtask、`motion_mask` 和 `motion_mask_ids` 是补充标注，三者分别带独立有效位，互不依赖。

数据层不会因为某个可选标注缺失而伪造有效目标，也不会用最后一帧填满未来监督。

## 2. 数据源分类

唯一分类依据是 `meta/info.json` 中的 `robot_type`：

```text
"ego" in robot_type.lower()  -> human
其他                            -> robot
```

入口实现在 `src/lerobot/datasets/wsa1_5.py`：

- `is_wsa15_human_robot_type`
- `wsa15_source_kind`
- `resolve_wsa15_delta_timestamps`

`factory.py` 通过 dataset 配置的 `dataflow="wsa1_5"` 启用该规则，不再 import policy 类来识别数据流。

## 3. 两条基础加载路径

### 3.1 Robot：标准 LeRobotV3

机器人数据完全使用标准 `LeRobotDataset`：

```text
主 parquet: observation.state[t]
主 parquet: action[t : t + H]
当前 RGB:   image[t]
```

其中 `H=chunk_size`。动作查询索引为 `0..H-1`，因此 action tensor 为 `[H, D_raw]`。

如果 `action_mode=delta`，是否求差以及哪些维度保持绝对量，只由该 `robot_type` 的 dataset schema 中 `action_mask_spec` 决定。数据层不会根据字段名自行转换 EEF，也不会把四元数改成欧拉角或旋转向量。

### 3.2 Human：LeRobotV3 媒体 + 补充动作表

人类数据仍用标准 `LeRobotDataset` 解码当前 RGB、索引和可选 motion 视频，但不使用主 parquet 中的 state/action 数值。`SyntheticEmbodimentDataset` 从配置指定的补充动作表读取：

```text
state  = q[t]
action = q[t+1 : t+H+1]
```

默认补充动作路径为：

```text
meta/annotations/action.parquet
```

demo 使用的是这个路径。若实际数据放在 `meta/action.parquet`，必须显式设置：

```text
dataset.human_action_file=meta/action.parquet
```

代码不进行第二路径猜测，避免误读旧文件。

## 4. Human 补充动作格式

### 4.1 本体合成动作

每种本体由一对列描述：

```text
action.<embodiment>        fixed-size vector [D]
action.<embodiment>_valid  bool
```

动作表还必须包含逐行对齐字段：

```text
index
episode_index
frame_index
```

加载器会校验这些索引与 LeRobotV3 主数据逐行一致。

### 4.2 原生 human EEF 动作

`human_eef` 不是从 `pose.parquet` 推导，而是只从 action parquet 的以下列读取：

```text
human_eef.right.pos       [3]
human_eef.right.rot       [4], quaternion xyzw
human_eef.right.grasp     [1]
human_eef.right.valid     bool
human_eef.left.pos        [3]
human_eef.left.rot        [4], quaternion xyzw
human_eef.left.grasp      [1]
human_eef.left.valid      bool
```

左右两侧都有效时该帧的 `human_eef` 才有效。任意一列出现但格式不完整会直接报错；不会退化成其它表示。

## 5. 有效帧与随机抽样

对 human 数据，anchor `t` 只有满足以下条件才进入 dataset 索引：

1. `t..t+H` 位于同一 episode；
2. `frame_index` 连续；
3. 至少一种动作表示在整个 `t..t+H` 窗口全部有效。

因此无效 anchor 在构建索引时就被移除，不进入 loss。`__getitem__` 会在当前 anchor 的所有有效动作表示中随机选择一种：可以是 `human_eef`，也可以是某个合成本体。

训练时 `DataLoader(shuffle=True)` 再从全部有效 anchor 中随机取帧。两层随机性的含义不同：

- shuffle 随机选择有效时间帧；
- dataset 随机选择该帧使用的有效 EEF/本体动作表示。

## 6. dataset schema 的职责

source schema 描述原始相机字段；具体动作表示的 schema 描述动作语义。Human 在随机选择动作表示后，会使用对应的 `egodex_synthetic_*` 或 `egoverse_synthetic_*` schema。

| schema 字段 | 职责 |
|---|---|
| `feature_mapping` | 将原始 state/action 字段组合为标准字段 |
| `image_mapping` | 将真实相机名映射到统一图像槽 |
| `action_mask_spec` | `delta` 模式下决定哪些维求差、哪些维保持 absolute |
| `state_reorder` / `action_reorder` | 从数据原始顺序重排到语义顺序 |
| `action_layout` | 将各语义块写入固定动作槽 |

### 6.1 WSA1.5 的 32D 语义槽

WSA1.5 不按原始向量长度做尾部 padding，而是写入固定语义位置：

| 范围 | 语义 |
|---|---|
| `[0,7)` | left arm joints |
| `[7,10)` | left EEF xyz |
| `[10,14)` | left EEF quaternion xyzw |
| `[14,16)` | left gripper，最多 2D |
| `[16,23)` | right arm joints |
| `[23,26)` | right EEF xyz |
| `[26,30)` | right EEF quaternion xyzw |
| `[30,32)` | right gripper，最多 2D |

输出同时包含：

```text
wsa.state_valid_mask       [32]
wsa.action_valid_mask      [32]
wsa.action_decode_matrix   [32, 32]
wsa.action_decode_bias     [32]
wsa.action_raw_dim         scalar
```

空槽永远不会作为有效动作监督。灵巧手 schema 会被 WSA1.5 明确排除，不会静默截断。

### 6.2 abs 与 delta

模型只有 `abs` 和 `delta` 两种数据策略：

- `abs`：不运行 `DeltaActionTransformFn`；
- `delta`：运行该 transform，并严格服从 schema 的 `action_mask_spec`。

例如单臂 7D EEF schema 可以声明 `[-7]`，表示 7 个值都保留原始 absolute 定义。双手 EEF 加夹爪是 16D，因此当前 human schema 使用 `[-16]`。固定槽编码只负责映射、归一化和生成 mask，不负责改变 EEF 的物理定义。

## 7. 可选 subtask 标注

默认从 `meta/sidecar.json` 读取，接受顶层 episode 列表或：

```json
{
  "episodes": [
    {
      "episode_index": 0,
      "task_instruction": "fold the cloth",
      "subtasks": [
        {
          "start_frame": 0,
          "end_frame": 40,
          "subtask_text": "pick up the left corner"
        }
      ]
    }
  ]
}
```

当前帧位于闭区间 `[start_frame, end_frame]` 时输出：

```text
wsa.subtask_text   string
wsa.subtask_valid  bool
```

进入 tokenizer 后，policy-facing batch 使用固定长度的 `wsa.subtask_token_ids` 和 `wsa.subtask_token_mask`。没有 sidecar 或当前帧没有 subtask 时，valid 为 false，不计算 subtask loss。

## 8. 未来 2D/3D 图

补充视频字段仍登记在 LeRobotV3 的 `meta/info.json` features 中：

```text
observation.motion_mask
observation.motion_mask_ids
```

EgoDex demo 的逻辑 feature 名带 `observation.`，而 episode 列和视频目录使用 `motion_mask` / `motion_mask_ids`。数据流会对这两组已声明名称建立显式映射；标准 LeRobot 的逻辑名与存储名相同时不映射。除此之外不猜测第三种路径。

当前 RGB 只查询 `t`。未来目标在 action horizon 内均匀查询。默认 `H=50`、`F=4` 时偏移为：

```text
[12, 25, 37, 50]
```

输出为：

| 字段 | shape | 插值 | 无标注时 |
|---|---|---|---|
| `wsa.future_motion_mask` | `[F,3,224,224]` | bilinear | 值为 `-1`，valid=false |
| `wsa.future_motion_mask_ids` | `[F,3,224,224]` | nearest | 值为 `-1`，valid=false |
| `wsa.future_motion_mask_valid` | scalar bool | - | false |
| `wsa.future_motion_mask_ids_valid` | scalar bool | - | false |

目标像素从 `[0,1]` 变换到 `[-1,1]`。两种目标分别判断是否存在，因此可以只监督其中一种。当前数据流不从 `pose.parquet` 合成这两个目标。

## 9. 归一化统计

robot 数据继续使用 LeRobotV3 metadata 中的统计量。human 的每种动作表示使用独立文件：

```text
<human_stats_dir>/<embodiment>/stats.json
```

统计文件包含 `observation.state` 和 `action` 的 mean/std。可用现有脚本计算：

```bash
python util_scripts/compute_synthetic_embodiment_stats.py \
  --dataset-roots-file /path/to/human_roots.txt \
  --output-dir /path/to/human_stats \
  --chunk-size 50 \
  --action-mode abs
```

`chunk-size` 和 `action-mode` 必须与训练一致。Human 数据存在时 `human_stats_dir` 是显式必填项，不会回退到 robot stats。

## 10. 完整执行顺序

```text
meta/info.json.robot_type
  |
  +-- robot
  |    LeRobotV3 image/state/action
  |
  +-- human
       LeRobotV3 image/index/optional motion
       -> action.parquet 有效窗口索引
       -> 随机有效 EEF/本体
       -> state=q[t], action=q[t+1:t+H+1]

两条路径汇合
  -> optional Delta（仅 action_mode=delta）
  -> future motion targets + 独立 valid
  -> RGB resize/remap
  -> Compose state/action
  -> 32D semantic slots + valid/decode metadata
  -> sidecar task/subtask + valid
  -> 后续 tokenizer/processor
```

## 11. 数据流启动脚本

`launch/wsa1_5_test_dataflow.sh` 是纯数据 smoke test，不加载 VLM 或 policy。它会递归发现数据集、按 `robot_type` 分类、随机解码有效帧，并打印 state/action、subtask、未来目标和所有有效位。

```bash
DATASET_ROOT=/path/to/collection \
HUMAN_STATS_DIR=/path/to/human_stats \
ACTION_MODE=abs \
CHUNK_SIZE=50 \
NUM_FUTURE_FRAMES=4 \
bash launch/wsa1_5_test_dataflow.sh
```

只检查指定数据集和本体：

```bash
REPO_IDS="robot/task_a human/task_b" \
HUMAN_EMBODIMENTS="human_eef franka" \
DATASET_ROOT=/path/to/collection \
HUMAN_STATS_DIR=/path/to/human_stats \
bash launch/wsa1_5_test_dataflow.sh
```

若数据实际位于 `meta/action.parquet`：

```bash
HUMAN_ACTION_FILE=meta/action.parquet \
DATASET_ROOT=/path/to/collection \
HUMAN_STATS_DIR=/path/to/human_stats \
bash launch/wsa1_5_test_dataflow.sh
```

对应 Python 入口是 `util_scripts/inspect_wsa1_5_dataflow.py`。

## 12. 关键实现文件

- `src/lerobot/datasets/factory.py`：多数据集创建和 robot/human 分流。
- `src/lerobot/datasets/wsa1_5.py`：时间查询、可选标注、motion target、固定语义槽。
- `src/lerobot/datasets/synthetic_embodiment.py`：补充动作表、有效窗口、随机动作表示。
- `src/lerobot/dataset_schemas/configs/egodex_v.yaml`：EgoDex source/embodiment schema。
- `src/lerobot/dataset_schemas/configs/egoverse_human_bimanual.yaml`：Egoverse source/embodiment schema。
- `util_scripts/inspect_wsa1_5_dataflow.py`：纯数据审阅入口。
- `launch/wsa1_5_test_dataflow.sh`：launch 包装脚本。
