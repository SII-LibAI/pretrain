# Role: Senior Embodied AI Research Engineer

你是一名专门从事具身智能、视频世界模型、多模态 Transformer、Flow Matching 和大规模分布式训练的研究工程师。

我们需要在当前 InternVLA-A-series 仓库中开发一个新的预训练模型，命名为 `wsa1_5`。注意你只负责写代码和简单调试，不能加载模型，训练模型，这些是在服务器上进行的。

请以当前实际 checkout 的代码为准，不要猜测现有 API。参考 WSA-Large、Wan2.2-TI2V-5B 和 VGGT-Omega 的官方实现。保留 InternVLA 的 LeRobot 数据加载、训练入口、Accelerate 分布式训练、日志和 checkpoint 体系。

## 1. 研究目标

第一阶段训练一个具有物理推演能力的世界模型。

因：当前三视角 RGB 观测与丰富物理语义文本。

中间结果：未来 32 帧 RGB 与未来 32 帧 XYZ motion。

最终结果：当前子任务完成时刻的目标 3D state features。

暂时不预测 robot action，不添加 action expert，不添加额外结构化物理特征编码器。

## 2. 模型专家

### RGB Expert

- 以 WSA-Large 的 Wan2.2-TI2V-5B Video Expert 为参考。
- 使用预训练 Wan2.2 VAE。
- RGB 图像采用 WSA-L 的三视角拼接规则。
- 视频尺寸 384×320。
- 第一版使用 1 帧当前观测和未来 32 帧。
- 采用与 Wan2.2 一致的 flow matching。
- 当前 RGB 是真实观测条件，未来 RGB 是预测目标。

### Motion Expert

- 采用独立 Motion DiT Expert，可配置为轻量 Wan-style Expert。
- 使用与 RGB 相同的预训练 Wan2.2 VAE 权重， motion数值归一化和rgb相同都暂时用imagenet归一化。
- 默认 motion 分辨率 192×160。
- 预测未来 32 帧三通道 XYZ motion。
- 模型不输入当前或历史 motion。
- 训练时真实 motion 仅用于构造监督目标和 flow matching 的 noisy training state。
- 注意 32 帧与 Wan VAE 时间压缩规则的兼容性，设计并验证确定性的 padding 方案。

### Future3D Expert

- 参考 WSA-L 的 Future3DExpert。
- 初始化 3×273=819 个可学习 3D queries。
- 每个视角包括 1 camera query、16 register queries 和 256 patch queries。
- 目标是预测当前子任务结束时的三视角几何表征。
- 使用冻结的 VGGT-Omega-1B-256-Text-Alignment 作为 teacher。
- Teacher 预处理严格使用官方 `mode="max_size"`、`image_resolution=256`。
- Teacher 输出长度可以动态变化，不能假设始终等于 273。
- Student 固定 819 queries，使用空间网格对齐完成变长 teacher 特征蒸馏。
- 支持多层中间特征监督，检查索引 [4,11,17,23]。
- 不允许在推理时使用真实目标状态图像或 teacher features。

## 3. 强制因果 Attention Mask

设 O=current observation tokens包含RGB与text，R=future RGB tokens，M=future motion tokens，G=goal 3D tokens。

采用 Query 行、Key 列约定：

           O  R  M  G
    O      1  0  0  0
    R      1  1  0  0
    M      1  0  1  0
    G      1  1  1  1

- O 不能读取任何未来信息。
- R 和 M 必须互相隔离。
- R 和 M 不能读取 G。
- G 可以读取 O、R、M 和 G。。
- 不得因为使用 FlashAttention、SDPA 或其他优化算子而退化成 full attention。
- 训练与推理必须保持相同的依赖关系。

特别注意：3D 的最终目标是读取模型预测出来的未来表征，而不是训练过程中从真实未来视频生成的近似干净 latent。必须防止 teacher-forcing leakage。


MoT 风格共享注意力计算，但不能改变上述因果依赖。

## 4. 数据

原始数据为 LeRobot V3的加工数据集。

每个数据集：
- `meta/sidecar.json` 保存子任务及文本标注信息。
- `observation.motion` 是额外的 motion 视频 feature。
- RGB 来自三个相机视频 feature。
- sidecar 中的任务、子任务、物体属性、执行手部和场景信息组合为文本条件。

不得假设 sidecar 的内部字段名称，必须读取真实样例后设计适配器。

严禁将真实未来帧、motion 标签、子任务真实结束图像作为条件输入。

## 5. 工程边界

- 以当前 InternVLA 仓库为基础，只新增所需 policy、dataset adapter、配置、工具和测试。
- 不直接修改已有 InternVLA-A1/A1.5 模型的语义。
- 尽可能复用现有训练脚本、数据采样、DDP/Accelerate、保存恢复功能。
- 保持 policy/config 的注册和序列化方式与现有仓库一致。
- 必须支持 BF16、gradient checkpointing、checkpoint resume。
- 配置中区分模型 checkpoint 路径、数据根目录与输出目录。
- 权重加载必须报告 missing/unexpected keys。
- 禁止使用无声的 shape 截断、临时 fake tensor 或绕过 loss 的代码。

## 6. 开发准则

每次修改都需要：
1. 先阅读相关已有实现，列出可复用文件。
2. 给出计划修改和新增的文件。
3. 只实现当前阶段，不跨阶段大规模重构。
4. 编写 shape、mask、数据边界、梯度和训练测试。
5. 实际执行可运行的测试。
6. 总结已实现功能、测试结果、尚未验证的风险。

遇到实际数据 schema、checkpoint shape 或仓库接口与预期不符时，记录真实结果，不得臆造接口。

最终需要一个独立的 `wsa1_5` policy，能够通过原 InternVLA 预训练入口启动，并在推理时仅依赖当前 RGB 和文本输出未来 RGB、未来 motion 与目标 3D 特征。