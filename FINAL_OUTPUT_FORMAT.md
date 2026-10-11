# 最终标注产物格式

最终数据位于每个 shard 的 `data/`、`videos/` 和 `meta/` 目录中。

```text
shard_XXX/
├── data/
├── videos/
│   ├── observation.images.front/
│   ├── motion_mask/
│   └── motion_mask_ids/
└── meta/
    ├── info.json
    ├── episodes/
    └── annotations/
        ├── action.parquet
        └── pose.parquet
```

## 公共字段

`action.parquet` 和 `pose.parquet` 每行对应一帧，均包含：

| 字段 | 类型 | 维度 |
|---|---|---:|
| `timestamp` | float32 | 1 |
| `frame_index` | int64 | 1 |
| `episode_index` | int64 | 1 |
| `index` | int64 | 1 |
| `task_index` | int64 | 1 |

## action.parquet

### 机器人关节 action

所有 action 都是 `list<float64>`，顺序为右臂到左臂。

| 字段 | 维度 | 组成 |
|---|---:|---|
| `action.piper` | 16 | Piper 右 8 + 左 8 |
| `action.piper_inspire` | 40 | Piper 右 8 + Inspire 右 12 + Piper 左 8 + Inspire 左 12 |
| `action.franka` | 18 | Franka 右 9 + 左 9 |
| `action.arx_l5` | 16 | ARX-L5 右 8 + 左 8 |
| `action.xarm7` | 14 | xArm7 右 7 + 左 7 |
| `action.rm75` | 14 | RM75 右 7 + 左 7 |
| `action.ur7e` | 12 | UR7e 右 6 + 左 6 |
| `action.ur7e_wuji` | 52 | UR7e 右 6 + Wuji 右 20 + UR7e 左 6 + Wuji 左 20 |

每个 action 都有一个对应的 `*_valid` 布尔标量字段。

### 人手 EEF

| 字段 | 维度 | 说明 |
|---|---:|---|
| `human_eef.right.pos` | 3 | 右手位置 `[x,y,z]` |
| `human_eef.right.rot` | 4 | 右手四元数 `[qx,qy,qz,qw]` |
| `human_eef.right.grasp` | 1 | 右手 grasp |
| `human_eef.right.valid` | 1 | 布尔值 |
| `human_eef.left.pos` | 3 | 左手位置 `[x,y,z]` |
| `human_eef.left.rot` | 4 | 左手四元数 `[qx,qy,qz,qw]` |
| `human_eef.left.grasp` | 1 | 左手 grasp |
| `human_eef.left.valid` | 1 | 布尔值 |

## pose.parquet

### 人手姿态

| 字段 | 维度 | 说明 |
|---|---:|---|
| `hand.right.pos` | 3 | 右手位置 |
| `hand.right.rot` | 9 | 右手 `3×3` 旋转矩阵展开 |
| `hand.right.valid` | 1 | 布尔值 |
| `hand.left.pos` | 3 | 左手位置 |
| `hand.left.rot` | 9 | 左手 `3×3` 旋转矩阵展开 |
| `hand.left.valid` | 1 | 布尔值 |

### 物体姿态

设当前帧有 `N` 个交互物体：

| 字段 | 单帧维度 |
|---|---:|
| `object.count` | 1 |
| `object.label` | N |
| `object.pos` | N × 3 |
| `object.rot` | N × 9 |
| `object.valid` | N |

`object.rot` 是每个物体的 `3×3` 旋转矩阵展开。

### 手-物体相对位姿

| 字段 | 单帧维度 |
|---|---:|
| `rel.left.pose` | N × 16 |
| `rel.left.valid` | N |
| `rel.right.pose` | N × 16 |
| `rel.right.valid` | N |

`rel.*.pose` 是 `4×4` 齐次变换矩阵展开为 16 维。

## motion_mask 视频

路径：

```text
videos/motion_mask/chunk-XXX/file-YYY.mp4
```

每帧是 `H×W×3` 的 MP4/H.264 RGB 视频，三个通道编码物体运动 `ΔX、ΔY、ΔZ`。

## motion_mask_ids 视频

路径：

```text
videos/motion_mask_ids/chunk-XXX/file-YYY.mp4
```

每帧是 `H×W×3`，三个 RGB 通道复制同一个物体 ID。

两个视频目录按照源视频的 `(chunk_index, file_index)` 共享存储，不是每个 episode 一个视频。

## mask 索引

写入 `meta/episodes/*.parquet`：

```text
videos/motion_mask/chunk_index
videos/motion_mask/file_index
videos/motion_mask/from_timestamp
videos/motion_mask/to_timestamp

videos/motion_mask_ids/chunk_index
videos/motion_mask_ids/file_index
videos/motion_mask_ids/from_timestamp
videos/motion_mask_ids/to_timestamp
```

其中 `chunk_index`、`file_index` 为 int64 标量，时间字段为 float 标量。
