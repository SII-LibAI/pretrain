现在开始实现 `wsa1_5`，只开发数据部分，不修改已有模型结构。注意你只负责写代码和简单调试，不能加载模型，训练模型，这些是在服务器上进行的。

### 数据来源

数据采用 LeRobot V3，每个数据集有：
- 标准三视角 RGB 视频 features
- `observation.motion` 三通道视频 feature
- `meta/sidecar.json`，保存丰富的子任务标注信息
- 标准 LeRobot episode、frame、timestamp 元数据
E:\VLA相关\wam_code\demo\inspire\qb-ilm\project\embodied-basic-model\zhangjianing-253108140206\HUMANDATASET\Egodex-LerobotV3\extra\knit_scarf   
这里有一份demo的数据集你可以看一下。sidecar中
hand_used:0左手 1右手 2双手
scene：场景
object_ragidity暂时不用。

### 训练样本定义

每个样本必须严格属于一个 episode 和一个 subtask。

当前观测是子任务内合法的起始观测；目标状态是此子任务真实的结束时刻。

输出训练字段：
- `condition_text`
- `observed_rgb`
- `future_rgb`
- `future_motion`
- `goal_rgb_views`
- `sample_id`
- `episode_index`
- `subtask_id`
- `start_timestamp`
- `goal_timestamp`
- `frame_timestamps`

第一版 observed RGB 为当前一个时间点的三视角 RGB。

future RGB 为未来 32 帧，future motion 为相同时间点或明确定义的时间区间对应的 32 帧 motion。

目标 3D teacher 的图像来自 subtask 的真实结束时刻，保留三路原始 RGB 图像，不使用拼接图作为 Omega 输入。


### 文本构造

构造固定、可复现的提示词模板，包含：
- 全局任务
- 当前子任务
- 执行手部
- 场景描述
所有物理属性只进入文本编码器。不要新增独立的 physics-value embedding。


### RGB 处理

复用 WSA-Large 的三视角拼接顺序和布局。
最终输出 384×320 的 RGB 画布。
三个相机的视角映射顺序需要配置和断言检查。
映射可复用internvla-series的schema

### Motion 处理

读取 `observation.motion`。
RGB 为真实 delta XYZ 坐标。
默认使用双线性插值 resize 到 192×160。



### 时间采样

时间采样与lerobot的规则不变

### 训练目标隔离

`future_motion` 只能出现在 targets 中。
模型 inference_condition 构造接口不得接收 motion。
`goal_rgb_views` 只能用于训练 teacher，不能进入模型条件。

### 兼容 InternVLA

复用现有数据集注册、预训练采样、数据混合与 DataLoader 机制。
尽量只新增自定义 dataset adapter，而不是复制整个 LeRobot Dataset。

### 兼容性

预训练数据集中如果缺少sidecar.json，或sidecar.json中缺少信息，则自动抹去当前输入。
数据集视频不满3视角或没有motion，参考WSA的占位机制，并标记mask不参与监督。

### 验收测试

1. 真实数据集能够返回一个样本。
2. 每个样本恰好 32 个 future RGB 和 motion 时间步。
3. episode 和 subtask 边界测试通过。
4. RGB 与 motion timestamp 对齐。
5. 三视角拼接顺序正确。
6. motion 的数值范围与解码误差有统计日志。
7. 在不提供 motion 的情况下仍能构造推理条件。
8. 多 worker 和多数据集加载能够正常运行。
