# Role: Senior Embodied AI Research Engineer

你是一名专门从事具身智能、视频世界模型、多模态 Transformer、Mixture-of-Transformers、Flow Matching 和大规模分布式训练的高级研究工程师。

我们正在 InternVLA-A-series 框架上开发自己的世界动作模型 **WSA1.5（World-Spatial-Action 1.5）**。

第一阶段已经完成世界模型预训练，实现了：

- RGB Expert：基于 Wan2.2-TI2V-5B 的未来 RGB 视频预测。
- Motion Expert：独立 Motion DiT，预测未来三通道 XYZ motion 视频。
- Future3D Expert：通过 VGGT-Omega 教师监督，预测子任务终点的 3D features。
- MoT 联合注意力及严格的物理因果 attention mask。
- LeRobot V3 数据适配、Accelerate 分布式训练、Flow Matching、checkpoint 和日志。

**本次任务是第二阶段 Action Post-training。**

我们需要为现有 WSA1.5 增加 Action Expert，使模型能够根据当前多视角观测、历史观测、语言指令、机器人当前本体状态，以及自身预测的目标 3D 信息，生成连续机器人动作序列。

本次是对已有预训练模型的增量扩展，**不是重新实现 WSA1.5**。

---

# 0. 开发边界与代码审查

项目地址：

InternVLA-A-series

参考实现：

1. WSA-Large：https://github.com/zaleni/WSA
2. InternVLA-A-series：https://github.com/InternRobotics/InternVLA-A-series
3. VPP2：https://github.com/roboterax/video-prediction-policy-2
4. Wan2.2-TI2V-5B 官方实现
5. VGGT-Omega 官方实现
6. PrismWam:https://prism-wam.github.io/
开始编写代码前，必须先检查**当前实际 checkout**，不能根据网上仓库的历史版本猜测 API。

优先阅读：

- `agents.md`、`rules.md`
- `src/lerobot/policies/wsa1_5/configuration_wsa1_5.py`
- `src/lerobot/policies/wsa1_5/dataset_wsa1_5.py`
- `src/lerobot/policies/wsa1_5/modeling_wsa1_5.py`
- `src/lerobot/policies/wsa1_5/text_embeddings.py`
- 当前训练入口、policy factory、optimizer、scheduler、checkpoint 相关文件
- WSA-Large Action Expert 和 state conditioning 的真实实现
- InternVLA-A1.5 的 action 数据处理、prefix 和训练流程

如果现有文件或 API 与预期不符，以当前代码为准。

## 强制约束

1. 只负责代码编写、配置、静态检查和轻量 CPU 测试。
2. 禁止加载 Wan、Omega 或完整 WSA1.5 权重。
3. 禁止运行正式训练、GPU forward 或大型模型性能测试。
4. 保留仓库现有 LeRobot、Accelerate、WandB、checkpoint、resume 体系。
5. 不修改第一阶段已有模型的核心计算语义。
6. 优先复用已有模块，不引入不必要的新框架。
7. 第一阶段 `enable_action=False` 时必须保持原有行为。
8. 不得无声修改已有 checkpoint 参数的 shape 或 key。
9. 不允许用 fake loss、伪造标签、隐式默认参数掩盖错误。
10. 不允许设置多于兜底，导致训练出现数值错误
11. 不允许因为新增代码导致训练可能出现严重通信阻塞，训练缓慢问题

开始修改前先输出：已有实现调查结果、拟复用模块、修改文件清单和技术方案。之后直接实施，无需等待一般性确认；遇到无法从代码确定的重要 schema 或架构冲突时，应明确记录事实和解决方案。

---

# 1. Action Expert 总体架构

## 1.1 新增 Action Expert

在现有 WSA1.5 MoT 中新增第四个预测专家：Action Expert。

参考 WSA-Large 的 ActionDiT，实现独立 Transformer Expert，使用 Flow Matching 预测连续 `action_chunk`。

模型的模态定义为：

- O：历史与当前 RGB Observation Tokens
- R：未来 RGB Prediction Tokens
- M：未来 Motion Prediction Tokens
- G：预测的 Goal 3D Tokens
- A：待生成的 Action Tokens

原有 RGB、Motion、Goal Expert 不重新实现。

Action Expert 包含：

- Action input projection
- Action timestep embedding
- Action positional embedding
- State conditioning
- Action Transformer blocks
- 与现有 MoT 兼容的 attention projections
- Action output projection
- Flow Matching 训练和采样接口

优先复用现有的Flow Matching 工具。

Action Expert 的 hidden dimension、层数和 FFN dimension 应可配置。默认实现参考 WSA-Large 的有效设计，但必须结合现有 WSA1.5 attention head 约束和 checkpoint 格式确定。

不要直接假定两个模型的模块命名或 shape 完全一致。

## 1.2 Action Expert 开关

新增配置：

```python
enable_action: bool = False
```

要求：

- False：完全保持第一阶段预训练行为。
- True：创建 Action Expert，并开启相应训练与推理。
- 禁用时不创建无用的 Action 参数，不要求 action/state 数据。
- 禁用时不改变 O/R/M/G attention 和 loss。
- 从第一阶段 checkpoint 初始化时，新增 Action Expert 使用独立初始化。

新增动作训练配置应具有单独的默认值，不能影响旧版配置反序列化。

## 1.3 Action Expert 的物理条件

Action Expert 的核心目标是学习：

```text
History RGB + Current RGB
             +
        Task / Prompt
             +
       Current State
             |
             v
       World Prediction/Goal3D
             |
             v
        Action Expert
             |
             v
       Future Action Chunk
```

动作生成必须真正依赖观察和预测的 Goal 3D

当前机器人 state 参考 WSA-Large和VPP2 的输入方式，用于动作生成条件化。

不新增独立物理语义特征编码器，物体、场景和交互物理属性仍然来自文本。

---

# 2. 数据流：历史观测

## 2.1 历史帧采样

新增两个参数：

```python
num_history_frame: int = 4
history_frame_stride: int = 2
```

定义：`num_history_frame` 表示当前帧之前的历史帧数量，当前帧不计入该数值。

当：

```python
num_history_frame = 4
history_frame_stride = 2
```

在 t 时刻输入：

```text
[t-8, t-6, t-4, t-2, t]
```

总共 5 个时间点。

通用公式：

```python
history_indices = [
    t - k * history_frame_stride
    for k in range(num_history_frame, 0, -1)
] + [t]
```

支持 `num_history_frame=0`，此时退化为单帧观测。

## 2.2 历史帧边界

必须满足：

- 不允许跨 episode。
- episode 开头不足的历史帧复制 episode 首帧补齐。
- 补齐帧需要独立 valid mask。
- 时间顺序严格保持从旧到新。
- 训练和推理使用完全相同的采样约定。
- 保持当前三视角 RGB 拼接及缺失相机 mask 规则。
- 不读取当前时刻之后的 RGB 作为 observation。

## 2.3 历史帧编码

当前 WSA1.5 的视频 VAE 时间布局基于单帧当前观测和未来帧。

不能简单将历史帧拼接到现有 `1 + future_frames` VAE 输入中。

请设计尽量复用原有编码器的 History Observation 路径。

- 历史帧增加可区分的时间位置或相对时间信息。
- 历史与当前观测共同组成 O tokens。
- 未来 RGB/Motion 的监督路径不变。
- 历史观测能够影响未来预测和 Action Expert。

具体采用何种设计，需要先核对当前 VAE 和 MoT 实现，也请参考PrismWam的实现方式

**验收要求：history=0 时能够退化为旧模型行为，不破坏第一阶段 checkpoint。**

---

# 3. Prompt 与文本条件

Prompt 参考 WSA-Large 和 InternVLA-A1.5 的设计，但必须兼容 WSA1.5 现有 Wan UMT5 cross-attention。

不允许直接照搬不兼容的 tokenizer 或 VLM 前缀接口。

## 3.1 Prompt 格式

推荐语义结构：

```text
[Robot Control Instruction]

Task: {task_instruction}
Subtask: {subtask_instruction}
Hand: {hand_used}
Scene: {scene_description}

Control Mode: {control_mode}
Action Mode: {action_mode}


```

这只是推荐的逻辑结构，不要求照搬字符串。其中control_mode从dataset_schemas里面读取，必须为endpose_effector或joint,不存在会报错。

Agent 应检查参考实现中真实的 prefix，并选择兼容当前文本编码器的格式。

## 3.2 新增配置

```python
action_mode: str = "delta"   # abs | delta
use_subtask: bool = False
```

其中：

**`action_mode`** 控制动作数值表示。

- abs：绝对目标
- delta：相对增量


## 3.3 use_subtask 行为

当 `use_subtask=True`：

- 使用数据集真实 sidecar 标注。
- 在 prompt 中添加 subtask、hand、scene 和已有物理语义。
- 保持第一阶段 sidecar 文本构造语义。
- 不从 sidecar 中无根据地推断不存在的字段。

当 `use_subtask=False`：

- 不使用 sidecar 的子任务文本、物理描述和 hand 信息。
- 使用任务自身可提供的 task instruction。
- 保留固定 prefix、control_mode、action_mode 等默认 prompt 字段。

任务文本优先复用 LeRobot 的任务元数据接口，不创建第二套不兼容的任务系统。

该模式必须支持在线机器人评估：推理时只知道当前任务，而不知道完整 subtask 划分。

## 3.4 文本缓存兼容

WSA1.5 已经支持 SHA256 文本 embedding 缓存。

必须复用现有 TextEmbeddingCache 和 UMT5 编码路径。

新增 prefix 会改变最终文本，因此：

---

# 4. 当前机器人状态 State

## 4.1 State 输入

Action Expert 需要当前机器人 state。

参考 WSA-Large 的具体实现，对 `observation.state` 等实际 feature 做适配。

新增：

```python
state_dim: int
action_dim: int
```

实际 feature key 和维度必须来自 dataset schema 或明确配置，不得猜测。

对于不同 robot embodiment，应采用实际已存在的映射和统计规则。


## 4.2 状态与动作归一化

参考 WSA-Large / InternVLA 的归一化策略，优先复用现有工具。

必须支持：

- state normalization
- action normalization
- action denormalization
- abs/delta 转换
- 机器人类型对应的统计量选择


---

# 5. Action Chunk 数据集

## 5.1 动作窗口

新增：

```python
action_chunk_size: int = 32
```

默认从当前控制时刻 t 开始，监督：

```text
A[t], A[t+1], ...,
A[t+action_chunk_size-1]
```


## 5.2 边界与 Mask

episode 末尾不足一个 chunk：

- 使用约定值或最后有效动作补齐。
- 同时输出 `action_valid_mask`。
- 所有 padding action 不参与 loss。
- 不得将后一个 episode 的 action 读入当前样本。

Action 缺失时：

- 如果是混合监督数据集，允许该样本 action loss 为零。
- 如果整个训练数据源均无 action，而启用了 Action Expert 监督，应给出清晰错误。
- 缺失 action 不影响有效 RGB/Motion/3D loss。

## 5.3 LeRobot 集成

基于现有 `WSA1_5Dataset` 增量修改。

保留：

- 现有 episode indexing
- Segment 定义
- 多相机数据读取
- motion video 解码
- future RGB sampling
- goal frame 提取
- dataset factory 和 collate
- 多数据集采样逻辑

新增必要的历史帧、state、action chunk 及其 masks。

不要复制一份完整的新数据加载框架。

---

# 6. Goal 3D 的新数据策略

## 6.1 新增参数

```python
allow_endframe_3d: bool = False
```

## 6.2 Goal frame 选择

按以下优先级处理：

**情况 A：use_subtask=True 且有有效 sidecar**

继续使用当前 subtask 的结束帧作为 3D 监督目标，保持第一阶段逻辑。

**情况 B：没有有效 sidecar，或 use_subtask=False**

- `allow_endframe_3d=True`：使用当前 episode 最后一帧作为 3D 监督目标。
- `allow_endframe_3d=False`：该样本不计算 3D supervision loss。

不能静默切换为其他结束帧。

## 6.3 Teacher 与 Goal Expert

保留当前 VGGT-Omega 教师实现：

- 只在有有效 3D 监督时提取 teacher features。
- 保留当前目标图像预处理规则。
- 保留多个层级的 feature distillation。
- 保留相机缺失 mask。
- 保留当前 3D loss 定义。
- 教师不接受梯度。

**特别强调：目标图像和 teacher features 永远只是监督标签。**

Action Expert 只能利用模型自己预测出的 G tokens，绝不能读取真实 goal image 或 Omega teacher features。

即使 `allow_endframe_3d=True`，episode 终点也只是训练监督标签，不能出现在推理输入里。

3D queries 是否存在和是否有 3D 监督必须继续保持独立。

---

# 7. MoT Attention Mask

## 7.1 全部模态

定义：

```text
O = History + Current Observation
R = Future RGB
M = Future Motion
G = Predicted Goal 3D
A = Action
```

要求新增 Action Expert 后继续执行严格的依赖关系。

## 7.2 Attention Mask

Query 为行，Key 为列，1 表示允许 attention：

```text
          O   R   M   G   A
       +---------------------
O      | 1   0   0   0   0
R      | 1   1   0   0   0
M      | 1   0   1   0   0
G      | 1   1   1   1   0
A      | 1   0   0   1   1
```

含义：

- O 只读取当前及历史观测。
- R 读取 O 和自身。
- M 读取 O 和自身。
- R 与 M 互相不能直接 attention。
- G 可以读取 O、R、M、G。
- A 只能读取 O、G 和自身。
- O/R/M/G 均不能读取 A。
- State 通过 Action Expert 条件化输入，不改变上述 block 依赖。

文本继续通过现有 cross-attention 机制注入，不把文本强行拼进 self-attention 序列。

## 7.3 严禁未来监督标签泄漏

这里需要特别检查现有训练机制。

当前第一阶段的 R/M 使用 Flow Matching 的 noisy target latent。G 又能读取 R/M。

因此，即使 A 不能直接读取 R/M，A 仍然可能通过 G 间接获得训练时真实未来帧的信息。

这是第二阶段最重要的风险之一。

实现要求：

1. Action 所依赖的 G 必须是模型预测路径得到的结果。
2. 不允许 Action 直接或间接获得真实未来 RGB/Motion、真实 Goal 3D teacher features。
3. 保留第一阶段 O/R/M/G 世界模型联合训练逻辑。
4. 对 Action loss，设计独立且无标签泄漏的 Goal conditioning 路径。
5. 优先复用模型自身的预测/采样组件，避免完整复制世界模型。
6. 如果需要额外一次前向或使用 detached predicted features，应明确说明计算成本与梯度路径。
7. Action 训练条件和推理条件必须尽可能一致。

不要以 `detach()` 作为防泄漏的唯一手段，因为 detach 只切断梯度，不会删除隐藏特征中已经包含的真实未来信息。

在动手修改之前，明确说明采用何种无泄漏 conditioning 方案，以及它相对于第一阶段 joint MoT forward 的变化。

必须提供测试：

固定当前 observation、prompt、state 和动作噪声，改变训练样本中的未来 RGB/Motion 标签、终点目标图像和 teacher features，不应改变 Action conditioning 分支的预测输入。

---

# 8. Action Flow Matching

## 8.1 训练

参考 WSA-Large 的 Action Flow Matching：

给定归一化动作：

```text
x1 = normalized_action_chunk
x0 = Gaussian noise
```

用与参考实现一致的插值方向、时间分布和速度目标训练。

如果采用简单线性约定：

```text
x_t = (1 - t) * x0 + t * x1
v_target = x1 - x0
```

必须保证该约定与采样 ODE 的时间方向一致。

不要直接复制 RGB 专家中的 flow shift 或噪声调度而不做检查。

输出：

```text
predicted_velocity: [B, action_chunk_size, action_dim]
```

训练损失：

```text
loss_action_flow
```

按 `action_valid_mask` 在有效元素上归约。

## 8.2 推理

实现动作 chunk 生成接口，优先与当前 LeRobot Policy API 对齐：

```python
predict_action_chunk(...)
select_action(...)
```

具体签名以当前基类为准。

推理需要：

- 固定随机 seed 的可复现生成。
- 可配置采样步数。
- 正确的动作反归一化。
- 输出 `[B, action_chunk_size, action_dim]`。
- 不需要未来真实图像或真实 3D 标签。
- 可以由在线环境逐时刻调用。

新增：

```python
action_sampler_steps: int = 10
```

可以提供独立于 RGB sampler 的采样步数。

推理时不得创建 VGGT-Omega teacher。

对于完整世界预测和快速 action-only 推理，如果共享前向与分离前向有不同成本，应提供明确的调用路径和说明，避免对同一条件重复运行无必要的模型计算。

---

# 9. LoRA 与渐进式后训练

## 9.1 WAN 微调策略

参考 VPP2 的分阶段训练思想，增加一个可配置的 RGB Wan Expert 微调策略：

```python
rgb_finetune_mode: str = "lora"
```

支持：

```text
frozen
lora
full
```

对应：

- frozen：冻结 RGB Wan Expert。
- lora：冻结 RGB 原始参数，仅训练所选模块的 LoRA。
- full：全参数微调 RGB Expert。

这三个模式应互斥。

只对 RGB Wan Expert 应用这项策略，不要意外将 LoRA 注入 Motion、Goal 或 Action Expert。

新增 LoRA 配置：

```python
rgb_lora_rank: int = 16
rgb_lora_alpha: int = 32
rgb_lora_dropout: float = 0.0
rgb_lora_target_modules: ...
```

Target modules 必须根据实际模型模块名确定，不能盲目使用通用列表。

默认优先考虑 attention projection，但要先检查当前 MoT 的共享 attention 实现是否允许安全注入。

确保 LoRA 权重被正确纳入训练、保存和恢复。

## 9.2 其余专家

继续保留或扩展模块冻结策略：

```python
freeze_motion: bool
freeze_3d: bool
freeze_heads: bool
```

Action Expert 具有独立学习率：

```python
action_lr: float
```

推荐策略：

- Action Expert 从头学习。
- RGB Wan 初期优先 frozen 或 LoRA。
- Motion / Goal 根据是否需要继续保留世界模型能力选择训练状态。
- 后期允许切换为 full 微调。


通过 config、launcher 和不同实验配置选择策略。


---

# 10. 多任务损失

模型总损失定义为：

```text
L_total =
    lambda_rgb    * L_rgb
  + lambda_motion * L_motion
  + lambda_3d     * L_3d
  + lambda_action * L_action
```

新增：

```python
lambda_action: float = 1.0
```

原有三个 loss 完全保留当前实现。

各模态都必须支持独立 mask。

例如：

- RGB 有效而 action 缺失：只训练可用目标。
- Action 有效但 sidecar 缺失：允许 action 正常监督。
- 3D teacher 不可用：只关闭对应 3D loss。
- 数据集没有 motion：不允许因为 motion 缺失而把 action 样本整体丢弃。

所有 loss 权重必须可以独立设置为零。

保留当前梯度累积、scheduler 和全无效监督处理语义。

新增日志：

```text
train/loss_action_flow
train/loss_rgb_flow
train/loss_motion_flow
train/loss_3d_feature
train/loss
train/action_grad_norm
train/action_lr
```

继续保留现有 step_s、input_s、mot_s、pred_s、backward_s 和 optimizer_s 计时口径。新增计时字段时注明其真实测量范围。

---

# 11. Checkpoint 与初始化

## 11.1 从第一阶段权重初始化

新增独立参数：

```python
init_checkpoint_path: str | None = None
```

用于初始化第二阶段模型。

流程：

1. 构造启用 Action Expert 的新模型。
2. 加载第一阶段 RGB、Motion、Future3D 及其他原有权重。
3. 初始化新增 Action Expert。
4. 如启用 LoRA，正确注入和初始化 adapter。
5. 输出详细的加载报告。


初始化第一阶段权重不等于 resume 训练。

---

# 12. 配置与训练入口

继续使用：

```text
policy.type = WSA_1_5
dataset.type = wsa1_5
```

不要因为增加 Action Expert 而创建一个完全独立的新 policy 类型。

新增一个第二阶段启动脚本，例如：

```text
launch/wsa_1_5_action_posttrain.sh
```

以及对应配置模板。

建议配置项如下，实际放在 policy 还是 dataset config 中，由现有代码职责决定：

```python
# Action
enable_action = True
action_chunk_size = 32
action_frame_stride = 1
action_sampler_steps = 10
lambda_action = 1.0

# History
num_history_frame = 4
history_frame_stride = 2

# Prompt
use_subtask = False
control_mode = "eef"
action_mode = "delta"

# Goal supervision
allow_endframe_3d = True

# Optimization
rgb_finetune_mode = "lora"
rgb_lora_rank = 16
rgb_lora_alpha = 32
rgb_lora_dropout = 0.0

# Checkpoint
init_checkpoint_path = None
```

这只是接口设计目标，不意味着所有参数都应该加入同一个配置类。


# 13. 数据条件与推理一致性

第二阶段的最终目标是在线机器人动作生成，因此必须特别检查训练和推理条件是否一致。

训练可使用：

- 历史 RGB
- 当前 RGB
- 当前 robot state
- 当前任务指令
- 可选 subtask 物理文本
- 真实未来 RGB/Motion 作为世界模型监督
- 真实目标 3D features 作为蒸馏监督
- 真实 action chunk 作为 Flow Matching 监督

推理仅可使用：

- 历史 RGB
- 当前 RGB
- 当前 robot state
- 当前任务指令
- control_mode/action_mode
- 模型内部预测的未来/Goal 表示
- 随机采样噪声

不能依赖：

- 未来真实 RGB
- 未来真实 Motion
- 未来真实 State
- 真实 Goal image
- Omega teacher features
- sidecar 标注的真实 subtask 终点
- 未来 action labels

对于 `use_subtask=False`，即使数据集磁盘存在 sidecar，也不能在 action inference 的文本或条件分支中间接使用该标注。

---

# 14. 测试与验收

只运行轻量测试，不加载完整预训练模型。

至少补充下列测试。

## A. Attention Mask

验证 5×5 block mask 与要求完全一致。

检查：

- A 可以读 O/G/A。
- A 不能直接读 R/M。
- O/R/M/G 不能读 A。
- R/M 保持互相隔离。
- 每个样本的 padding mask 生效。

## B. History Sampling

验证：

```text
num_history_frame=4
history_frame_stride=2
```

正确得到：

```text
t-8, t-6, t-4, t-2, t
```

覆盖 episode 开头、末尾、短 episode 和无历史情况。

## C. Prompt

验证 use_subtask=True/False。

在 False 模式中，无论是否存在 sidecar，最终 prompt 都不包含 sidecar-only 信息。

检查不同 prompt 的文本缓存 key 不会冲突。

## D. Goal Frame

测试：

- 有 sidecar 时的 subtask 终点。
- 无 sidecar 时的 episode 终点。
- use_subtask=False 时的 episode 终点。
- allow_endframe_3d=False 时的 mask。
- 3D supervision loss 关闭时不读取无效 goal。

## E. Action Dataset

检查：

- action chunk shape。
- state shape。
- chunk 首个动作的时间对齐。
- padding mask。
- 不跨 episode。
- abs/delta 模式。
- normalize/denormalize round-trip。

## F. Action Expert

使用小尺寸随机模块检查：

- action token projection
- timestep embedding
- state conditioning
- Flow Matching 输出 shape
- masked loss
- action sampler 的数值计算约定

不得通过加载完整 Wan/WSA 权重来测试。

## G. 梯度与 LoRA

利用轻量参数或 mock 模块检查：

- frozen RGB 无梯度。
- LoRA 模式只有目标 adapter 可训练。
- full 模式 base 参数可训练。
- Action Expert 正确接收梯度。
- 冻结参数没有进入 optimizer。
- LoRA target modules 实际匹配。

## H. Label Leakage

验证 Action conditioning 不受训练标签的直接影响。

使用受控 synthetic inputs，固定观测、任务、state 和随机噪声，替换未来监督标签。

Action 分支使用的预测特征不应因为读取了真实未来标签而改变。

## I. Checkpoint Compatibility

在不加载大型权重的前提下，通过配置、state_dict key 检查或小型模块模拟验证：

- enable_action=False 的旧路径。
- Stage-1 → Stage-2 初始化。
- 新增 Action 参数缺失的允许列表。
- 原有参数 shape mismatch 的明确报错。
- LoRA 参数保存和恢复约定。

不能把静态测试通过描述为真实大模型 checkpoint 已成功加载。

## J. 静态检查

运行当前项目已有、且无需 GPU 权重的适用检查：

- Python syntax/import
- Ruff 或现有 linter
- pytest 中相关 CPU 测试
- Shell syntax check
- Launcher dry-run
- 配置解析测试

---

# 15. 实施顺序

为了降低错误风险，按以下顺序工作。

### Phase A：现有实现审查

阅读当前代码，确认 Action Expert 与现有 O/R/M/G MoT 的集成方案。

特别检查：

- shared attention 的实际实现
- 3D goal conditioning
- Flow Matching 的时间定义
- 现有数据 adapter 的状态
- checkpoint 初始化和 resume
- WSA-Large 的动作维度、state encoder 与 Action Expert

先说明发现的问题。

### Phase B：数据与配置

实现：

- history sampling
- state/action chunk
- prompt 模式
- allow_endframe_3d
- action mask
- text cache 兼容

优先完成对应测试。

### Phase C：模型

实现：

- Action Expert
- state conditioning
- 5 模态 attention mask
- Action Flow Matching
- action sampling

验证无监督标签泄漏。

### Phase D：训练集成

实现：

- 四种 loss
- optimizer 参数组
- frozen/lora/full
- Stage-1 初始化
- Stage-2 checkpoint
- 日志和训练入口

### Phase E：推理和文档

提供：

- Action inference API
- 不依赖 sidecar 的推理方式
- 服务器训练启动命令
- checkpoint 初始化示例
- LoRA/frozen/full 配置示例
- 第二阶段使用说明

---

# 16. 最终交付要求

完成后必须输出：

1. **Architecture Summary**：说明新增的 Action Expert 如何与原世界模型交互。
2. **Modified Files**：列出所有修改和新增文件及原因。
3. **Data Schema**：以RoboTwin-Agilex-Aloha为例,列出 observation、history、state、action、goal 的 tensor shape 和 mask。
4. **Attention Dependencies**：给出最终 attention mask 和防标签泄漏方案。
5. **Training Strategy**：解释 RGB frozen/lora/full 的具体训练参数范围。
6. **Checkpoint Compatibility**：说明第一阶段初始化及第二阶段 resume。
7. **Test Results**：列出实际运行的测试、通过情况及未测试部分。
8. **Server Commands**：给出可复制的第二阶段训练、resume、action inference 命令。
9. **Known Risks**：包括训练/推理分布差异、额外前向开销、数据时间对齐以及缺失监督等问题。

不要只输出代码片段或设计建议。应直接修改当前仓库中的实际文件，完成必要的最小实现，并执行允许范围内的测试。


**最终目标：保留第一阶段 WSA1.5 的视频世界模型能力，新增可配置、可独立监督、可从已有预训练 checkpoint 初始化的 Action Expert，使 WSA1.5 成为一个具备历史观测、目标 3D 推演和连续动作生成能力的完整 World-Action Model。**