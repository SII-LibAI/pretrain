现在实现 `wsa_1_5` 的 RGB Video Expert 和 XYZ Motion Expert。注意你只负责写代码和简单调试，不能加载模型，训练模型，这些是在服务器上进行的。

基于阶段 0 已确认的 WSA-Large 代码，复用 Wan2.2-TI2V-5B Video Expert、Wan2.2 VAE 和 UMT5 conditioning。

严格保持现有 causal attention 架构。

## A. RGB Expert

配置：
- image size: 384×320
- observed frames: 1
- future frames: 32
- total training video frames: 33
- video backbone: Wan2.2-TI2V-5B
- VAE: Wan2.2 VAE

训练过程：
1. 读取当前 RGB 和未来 RGB。
2. 通过 Wan VAE 编码 33 帧。
3. 使用 WSA-Large 同款 flow-matching objective。
4. 第一帧真实观测作为无噪条件，未来 latent 添加噪声。
5. RGB Expert 预测 future latent flow velocity。
6. 只在未来部分计算flow loss。
7. 当前条件的 VAE latent 必须只依赖当前已观测 RGB，不能依赖任何真实未来 RGB。

实现仅输入当前 RGB 和文本的 RGB sampler，生成未来 32 帧。

## B. Motion Expert

使用一个独立 MotionDiT，不与 Video Expert 共享可训练 Transformer 权重，这个权重可以从预训练的WAN_DiT模型中插值的来。

建议初始配置：
- hidden_dim: 1024
- layers: 30
- motion size: 192×160
- future motion frames: 32
- VAE: 同一套预训练 Wan2.2 VAE 权重
- flow matching

Motion Expert 接收当前 RGB 条件和文本条件。

严禁加入真实当前 motion 条件，也不能通过训练 batch 的其他隐藏字段获取当前 motion。


## C. Motion temporal handling

真实目标只有未来 32 帧。

请先用官方 Wan2.2 VAE 测试 32 帧直接编码、解码的真实行为。

如果该实现要求 4n+1 帧，优先实验：
- 在时间 t0 添加固定全零占位帧。
- 占位帧不来自真实 motion。
- 在训练目标和解码输出中正确处理占位。
- 真实未来 motion 仍对应 t1 至 t32。
- 确保生成时不需要任何 motion observation。

必须验证该方案不会造成未来 motion 通过 VAE 时间卷积泄漏到作为条件的占位 latent。

不要通过硬编码 latent slice 假装满足时间对齐。

## D. Expert 初始化

RGB Expert 从 Wan2.2-TI2V-5B 预训练权重初始化。

Motion Expert 采用可配置的初始化方式：
- Wan-style 缩宽初始化

优先复用 WSA-L 的专家权重预处理思路。
不同 hidden dimension 的权重转换必须有显式规则。
输出 missing/unexpected keys 和每层参数数量。

## E. Training

实现：
- RGB flow-matching loss
- Motion flow-matching loss
- 可选 motion 数值重建指标
- 模态独立 loss logging

RGB 和 Motion 不能互相 attend。
两者读取相同的当前 RGB 和文本条件。


## F. 验收

1. RGB Expert  forward/backward 正常。
2. Motion Expert  forward/backward 正常。
3. 两者联合 forward/backward 正常。
4. RGB 输出不依赖 motion 输入 tokens。
5. Motion 输出不依赖未来 RGB tokens。
6. Motion inference 函数只需要当前 RGB、文本。
