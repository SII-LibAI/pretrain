# WSA1.5 模型架构与接口说明

本文对应 `src/lerobot/policies/wsa1_5` 的当前实现，以实际代码路径为准，供模型结构和数据接口审阅。

## 1. 设计依据与结论

WSA1.5 保留 InternVLA-A1.5 的 LeRobot Policy 接口，内部参考三部分实现：

- [Qwen-RobotManip 技术报告](https://arxiv.org/abs/2606.17846)及[官方代码](https://github.com/QwenLM/Qwen-RobotManip)：采用“不同本体写入固定语义位置 + 逐维有效掩码”的原则，而不是按原始维度做尾部 padding。本项目根据自己的本体改成 32 维，不照搬其 80 维表。
- [WSA 技术报告](https://arxiv.org/abs/2607.03941)及[官方代码](https://github.com/zaleni/WSA)：参考 WSA-B 的理解/规划/动作三专家结构，以及 WSA-L 的未来视频和 Future-3D query 设计。
- 本仓库 `internvla_a1_5`：复用 Qwen3.5-2B 多模态上下文、WAN2.2 世界解码器、flow-matching 动作生成和已有 Policy API。

当前有两个明确边界：

1. memory 机制暂时留空。代码会移除 `language_memory`，不把未定义的记忆内容送入模型。
2. 当前数据只有 `motion_mask` 和 `motion_mask_ids`，没有 WSA-L 的 DA3 多层 3D teacher 特征。因此目前是两组 query 联合条件化 WAN，而不是完整复刻 WSA-L 的独立 3D 蒸馏头。

## 2. 总体模型架构

```text
当前多视角 RGB + 任务指令
             |
             v
 Qwen3.5-2B 多模态上下文专家 ---- context tokens [B,L,Hq]
             |                              |
             | full-attention 层联合 Q/K/V  |
             |                              |
  +----------+-------------------+----------+
  |                              |          |
  v                              v          v
规划专家（全深度）                 动作专家（全深度）
[16 subtask queries]             [2 state tokens]
[64 future-2D queries]           [2T noisy-action tokens + time]
[144 future-3D queries]                    |
  |              |                         v
  |              |                 flow velocity [B,T,32]
  |              |
  |              +--> 208 future outputs --> 分支适配器 --> WAN2.2
  |                                                     |
  v                                                     v
subtask logits [B,16,V]                 motion canvas [B,Tv,3,224,448]
```

### 2.1 三个专家

- **上下文专家**：预训练 Qwen3.5-2B，编码当前图像和任务文本；视觉编码使用 Qwen3.5-2B 自身的多模态视觉路径。
- **规划专家**：与 action expert 同配置、同层数的 `Qwen3_5TextModel`，拥有独立 Transformer 参数。它没有词嵌入层，直接接收可学习 query。
- **动作专家**：复用 InternVLA-A1.5 action expert 骨干，输入当前状态、带噪动作和 flow 时间。

在 Qwen3.5 的 `full_attention` 层，三个专家的 Q/K/V 会拼接后按显式依赖图联合计算。`linear_attention` 层具有各自的递归状态，三个分支分别计算，并在下一个 full-attention 层再次交换信息。这是全深度 Mixture-of-Transformers，不是在 Qwen 后附加两层浅 Transformer。

### 2.2 token 序列和默认长度

```text
context : [image/text context]
planning: [subtask x16 | future-2D x64 | future-3D x144]
action  : [left state | right state | (t0,left) | (t0,right) | ...]
```

默认动作 horizon `T=50`，action branch 长度为 `2 + 2T = 102`。planning branch 长度为 `16 + 64 + 144 = 224`，其中送给 WAN 的未来 query 为 `64 + 144 = 208`。

## 3. 32 维跨本体状态/动作空间

### 3.1 固定语义槽位

平坦向量为 `[left 16 | right 16]`，左右完全对称。左右夹爪均为 2 维，EEF 朝向均使用数据中的 `xyzw` 四元数。

| 索引（左闭右开） | 维数 | 语义 |
|---|---:|---|
| `[0,7)` | 7 | 左臂关节 |
| `[7,10)` | 3 | 左 EEF `xyz` |
| `[10,14)` | 4 | 左 EEF 四元数 `xyzw` |
| `[14,16)` | 2 | 左夹爪 |
| `[16,23)` | 7 | 右臂关节 |
| `[23,26)` | 3 | 右 EEF `xyz` |
| `[26,30)` | 4 | 右 EEF 四元数 `xyzw` |
| `[30,32)` | 2 | 右夹爪 |

不再保留灵巧手槽位、reserved 槽位或隐式扩展区。schema 中出现 `hand` 语义块会直接报错，避免手指维度被静默截断或误映射。

### 3.2 数值语义

- 关节和夹爪状态是当前绝对值；关节和夹爪动作是未来绝对目标。
- EEF 状态是当前绝对 `xyz + quaternion_xyzw`。四元数会归一化，并统一到 `w >= 0` 的半球，减少 `q` 和 `-q` 双覆盖造成的不连续。
- EEF 动作是相对当前状态的 `delta_xyz + delta_quaternion_xyzw`。编码使用 `R_delta = R_target @ R_current.T`，解码使用 `R_target = R_delta @ R_current`。
- 有 `observation.camera_extrinsics` 时，EEF delta 转到参考相机坐标系；无标定时退化为基座坐标系，并输出 `wsa.camera_frame_valid=false`。

### 3.3 语义映射不是 padding

`CanonicalStateActionTransformFn` 根据 schema 的 `action_layout`、`action_reorder` 和 `state_reorder` 把语义块放入固定位置。未使用槽位虽然数值为 0，但 mask 为 `False`，不会进入动作损失。缺失 layout、未知语义或容量超限均直接报错。

| 本体类型 | 映射方式 |
|---|---|
| 双 Piper | 每侧 6 个臂关节写入 joint 前 6 维，2 维并联夹爪写入 gripper |
| 双 Franka | 每侧 7 个臂关节和 2 维夹爪写入对应槽位 |
| Human EEF | 每侧 `xyz + quaternion_xyzw + grasp`；单维 grasp 占夹爪第 1 维，第 2 维无效 |
| 灵巧手 | 当前不支持，显式报错而非截断或 padding |

数据变换输出：

```text
wsa.state_valid_mask  bool[32]
wsa.action_valid_mask bool[32]
```

模型将 `[B,32]` 状态变成左右两个 16D token，将 `[B,T,32]` 动作变成 `2T` 个 16D token。动作头再做严格逆变换，不会把整个 32D 向量压成一个 token。

## 4. 输入定义

### 4.1 推理输入

| 键 | 形状 | 含义 |
|---|---|---|
| `observation.images.pixel_values` | Qwen processor 格式 | 当前时刻多视角 RGB patch |
| `observation.images.image_grid_thw` | `[Nimg,3]` | Qwen 视觉网格 |
| `observation.images.input_ids` | `[B,L]` | 图像占位符和当前任务指令 |
| `observation.images.attention_mask` | `[B,L]` | 上下文有效 token |
| `observation.state` | `[B,32]` | 规范语义状态 |
| `wsa.state_valid_mask` | `[B,32]` | 状态有效维 |
| `wsa.action_valid_mask` | `[B,32]` | 当前本体可输出的动作维 |

“当前观测”分两路：RGB 进入 Qwen 上下文专家，本体状态进入动作专家；两路在 full-attention 层交互。

### 4.2 memory 当前状态

架构上保留“观测 + 指令 + memory”的扩展位置，但 memory 的数据定义、检索规则、时序编码和训练目标尚未确定。当前实现：

- 不创建静态 memory learnable token；
- 不读取 `pose.parquet` 并把 pose 串成文本；
- `WSA15ChatProcessorTransformFn` 删除 `language_memory`，不送入 Qwen。

所以本版真正的 context 是“当前图像 + 当前指令”。memory schema 定稿后可增加独立编码和注意力依赖，而不改变 Policy 外部接口。

### 4.3 训练附加输入

| 键 | 形状 | 含义 |
|---|---|---|
| `action` | `[B,T,32]` | 规范动作块，默认 `T=50` |
| `wsa.subtask_token_ids` | `[B,16]` | 离散子任务 BPE 目标，忽略位为 `-100` |
| `wsa.subtask_token_mask` | `[B,16]` | 子任务有效位置 |
| `observation.video_frames` | `[B,Tv,3,224,448]` | 左半 `motion_mask`，右半 `motion_mask_ids` |
| `world_target_valid` | `[B]` | 是否存在世界监督 |
| `VQA.labels` | `[B,L]` | 可选 VQA next-token 监督 |

`sidecar.json` 的 `subtask_text` 只构造子任务标签；chat processor 会在处理上下文前移除该字段，防止答案泄漏。

## 5. learnable token 的作用和数量

learnable token 是训练参数，不是预先写好的文本，也不是未来图像像素。它们在每个 batch 中以相同初值出现，通过注意力主动查询当前图像、指令、状态和动作；经过规划专家后，才成为与当前样本相关的规划表示。

| query 组 | 默认数量 | 作用 |
|---|---:|---|
| `subtask_queries` | 16 | 固定槽位的离散子任务预测 |
| `future_2d_queries` | 64 | 2D 物体、区域和跟踪规划瓶颈；数量可看作 `8x8` 容量，但代码不把它硬绑定到像素网格 |
| `future_3d_queries` | 144 | 3D 变化规划瓶颈；采用 WSA-L 默认的每视角 `144=12x12` query 数 |

总计 224 个 planning token。三组 query 拥有独立参数；2D/3D 还分别使用输入投影 `future_*_in_proj` 和 WAN 条件投影 `future_*_to_world`，不是只在名称上分组。`future_query_init_std` 默认为 `0.02`。

## 6. 2D/3D 未来预测如何生成

### 6.1 WSA-L 原实现的关键机制

WSA-L 的未来视频和未来 3D 是两类不同输出：

- **未来视频**：画面进入 WAN VAE latent 空间。训练时对未来 latent 加噪并做 flow matching，第 1 帧作为固定条件；视频专家处理密集时空 latent token，并非用少量 query 直接回归所有像素。
- **Future-3D**：默认每视角使用 144 个 3D query。WSA-L 默认 2 个视角，所以共有 288 个；query 在 MoT 中与视频和动作交互，再由 Perceiver 式输出 query 展开成每视角 1296 个特征，对齐 DA3 teacher 的多层 3D 特征，使用 cosine 与 layer-normalized MSE 监督。

因此 learnable query 是检索、压缩未来结构的中间载体，不是视频或 3D 点本身。

### 6.2 本仓库的适配实现

当前世界目标是一张左右拼接的 canvas：

```text
left  half: motion_mask      [B,Tv,3,224,224]  # 栅格化 Delta-X/Delta-Y/Delta-Z
right half: motion_mask_ids  [B,Tv,3,224,224]  # 2D 实例/跟踪 ID 图
packed    : motion_canvas    [B,Tv,3,224,448]
```

默认 `num_video_frames=4`，再加第 0 条件帧，因此 `Tv=5`。数据变换把第 0 帧设为空白固定条件，未来帧由 WAN 在 VAE latent 空间做 flow denoising 后解码。

```text
64 learnable 2D queries ---+
                            +--> planning expert --> [B,208,H]
144 learnable 3D queries --+             |
                                         v
                              2D/3D 独立 world adapter
                                         |
                                         v
                               WAN cross-attention context
                                         |
                  noisy future VAE latent + fixed condition frame
                                         |
                                         v
                         flow denoising --> VAE decode --> canvas
```

2D query 的设计目标是汇总物体身份、区域和像平面运动；3D query 的设计目标是汇总动作条件下的未来 `Delta-XYZ`。当前代码已经分离两组 query 的参数、投影和注意力路径，但最终仍由同一个 WAN 联合生成左右 canvas，像素损失也是联合的。因此“2D query 只决定右半、3D query 只决定左半”目前不是硬约束。后续若加入 DA3 teacher，应再增加独立 Future-3D 蒸馏头。

## 7. 显式注意力依赖

| Query 组 | 允许读取的 Key 组 |
|---|---|
| context | context，保持 Qwen 因果掩码 |
| subtask | context + subtask |
| future 2D | context + subtask + future 2D + future 3D |
| future 3D | context + subtask + future 3D + action |
| state | context + subtask + future 3D + state |
| action | context + subtask + future 3D + state + action |

子任务作为高层规划条件；2D 可以查询 3D；3D 与动作双向交换；动作不直接读取 future-2D；context 不读取未来或动作 token，防止目标反向泄漏。`knowledge_insulation=true` 时，规划/动作分支读到的 context K/V 会 `detach`。

## 8. 输出与损失

### 8.1 离散子任务

16 个 subtask 输出经 `subtask_to_qwen` 对齐到 Qwen hidden size，再复用 Qwen `lm_head`：

```text
subtask_logits [B,16,vocab_size]
```

这是固定槽位的并行离散预测，不是自回归文本生成。

### 8.2 动作 flow matching

```text
x_t = t * noise + (1-t) * action
target_velocity    = noise - action
predicted_velocity = model(...)             # [B,T,32]
```

flow MSE 只在 `wsa.action_valid_mask=True` 且属于机器人样本的维度上聚合。推理从高斯噪声开始，默认用 10 步 Euler 从 `t=1` 积分到 `t=0`，再由 codec 逆映射回环境的原始动作顺序和维度。EEF 路径结合当前位姿，把相对 `xyz + quaternion` 恢复成未来绝对位姿。

### 8.3 总损失

```text
L = L_action
  + lambda_vqa * L_vqa
  + subtask_loss_weight * L_subtask
  + video_loss_weight * L_world
```

- `L_action`：有效动作维的 flow-matching MSE。
- `L_vqa`：可选 Qwen next-token CE。
- `L_subtask`：有效子任务槽位的 CE。
- `L_world`：仅对 `world_target_valid=True` 的样本计算 WAN flow-matching loss。

## 9. Policy 公共接口

| 方法 | 返回值 |
|---|---|
| `select_action(batch)` | 单步原始本体动作 `[B,Draw]`，使用 action queue |
| `predict_action_chunk(batch)` | 原始本体动作块 `[B,T,Draw]` |
| `predict_action_chunk_with_video(batch, steps)` | `(actions, motion_canvas)` |
| `predict_action_chunk_with_world(batch, steps)` | `(actions, world_dict)` |
| `forward(batch)` | `(total_loss, loss_dict)` |

`world_dict` 内容：

```text
subtask_token_ids [B,16]
subtask_logits    [B,16,V]
future_2d_queries [B,64,H]
future_3d_queries [B,144,H]
motion_mask       [B,Tv,3,224,224]
motion_mask_ids   [B,Tv,3,224,224]
motion_canvas     [B,Tv,3,224,448]
```

## 10. 数据处理顺序

```text
PrepareWSA15MotionTargetsFn
-> Resize/remap current RGB
-> Compose raw state/action fields
-> CanonicalStateActionTransformFn (semantic 32D + masks + inverse codec)
-> LoadWSA15PhysicalAnnotationsFn (subtask only; memory remains empty)
-> WSA15ChatProcessorTransformFn (Qwen context + fixed subtask labels)
-> UnifyWSA15InputsFn
```

主路径不再使用通用 `PadStateAndActionTransformFn`，也不运行产生 gap padding 的通用 reorder。关节绝对/EEF 相对的混合语义由专用 codec 实现，因此不再叠加通用 `DeltaActionTransformFn`。

## 11. 关键配置

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `vlm_model_name_or_path` | Qwen3.5-2B 路径 | 上下文视觉语言主干 |
| `max_state_dim` | 32 | 固定语义状态维度 |
| `max_action_dim` | 32 | 固定语义动作维度 |
| `num_subtask_tokens` | 16 | 子任务 query 数 |
| `num_future_2d_tokens` | 64 | 未来 2D query 数 |
| `num_future_3d_tokens` | 144 | 单视角未来 3D query 数 |
| `future_query_init_std` | 0.02 | 2D/3D query 初始化标准差 |
| `chunk_size` | 50 | 动作 horizon |
| `num_inference_steps` | 10 | 动作 Euler 步数 |
| `num_video_frames` | 4 | 未来帧数；world tensor 另含第 0 条件帧 |
| `subtask_loss_weight` | 1.0 | 子任务 CE 权重 |
| `video_loss_weight` | 1.0 | 世界损失权重 |
| `lambda_vqa` | 1.0 | VQA 权重 |

`num_learnable_tokens` 自动设为 `64 + 144 = 208`，用于兼容 InternVLA-A1.5 的 WAN 构造接口；模型中已用显式 `future_2d_queries` 和 `future_3d_queries` 替代父类未分类的 query bank。

## 12. 代码位置

| 文件 | 责任 |
|---|---|
| `configuration_wsa1_5.py` | Policy/Dataset 配置和 transform pipeline |
| `modeling_wsa1_5.py` | 三专家模型、query 头、flow 动作和 Policy API |
| `mot.py` | 全深度三专家逐层执行 |
| `attention.py` | 可独立测试的显式注意力依赖图 |
| `transform_wsa1_5.py` | Qwen context processor 和子任务标签 |
| `../../datasets/wsa1_5.py` | 32D codec、sidecar 子任务、世界标注和 batch schema |

`attention.py` 已放入 `policies/wsa1_5/` 包内，不再在 `src/lerobot/` 根目录放 WSA1.5 专用文件。

## 13. 当前限制

1. memory 是空扩展点，当前没有记忆检索、记忆 token 或时序记忆损失。
2. EEF camera-frame delta 在没有外参时退化到 base frame；若要求严格相机系，应设置 `require_camera_calibration=True`。
3. 2D/3D query 当前联合条件化一个 WAN，没有分开的像素损失，也没有 WSA-L 的 DA3 多层 3D 蒸馏。
4. 灵巧手 schema 当前不支持；需要时必须设计新的显式槽位，不能借用保留维。
   使用 `synthetic_embodiment` 时应通过 `dataset.embodiments` 排除
   `piper_inspire`、`ur7e_wuji` 等灵巧手本体，否则 codec 会在加载样本时明确报错。
5. 旧版浅层 `mot.*`、80D 动作头和旧 query bank 参数与本结构不兼容。Qwen3.5 和 WAN 仍按 InternVLA-A1.5 路径加载，新规划专家、query 和 32D 动作头需要重新训练。
6. `inference_backend=optimized` 和 `compile_model=True` 对三专家路径会明确报错，避免静默退化为旧 action-only 语义。
