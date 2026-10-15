现在实现第三个专家 `Future3DExpert` 和冻结的 `VGGT-Omega Teacher`。

必须复用当前已经实现的 Causal MoT 规则，不允许修改 RGB/Motion 的因果关系。注意你只负责写代码和简单调试，不能加载模型，训练模型，这些是在服务器上进行的

## 1. Future3DExpert

参考 WSA-L 的 Future3DExpert。

随即噪声初始化 819 个可学习 tokens，分成三个视角，每个视角 273 个：
- 1 camera query
- 16 register queries
- 256 patch queries

建议初始配置：
- hidden_dim=768
- num_layers=30
- num_views=3
- queries_per_view=273
- output_dim=2048

支持从Wan 缩宽权重初始化，。

## 2. VGGT-Omega Teacher

使用官方 `VGGT-Omega-1B-256-Text-Alignment` checkpoint。

按照官方要求构建模型，正确启用对应的 checkpoint 选项。

输入是真实子任务结束时刻的三视角 RGB 原图。

预处理：
- image_resolution=256
- 完全遵循官方极端长宽比裁剪、resize 和多视角 padding 规则，而不是直接拉伸
 
要注意VGGT-Omega的输入处理方式，确认每个图像中间层的token数为273

冻结 teacher，使用 eval 和 inference_mode。

提取中间层 [4,11,17,23] 的 features。
在调用官方源码之后核对真实返回接口、维度和层索引。

优先对齐每一层 2048-D 的 hidden features。

## 3. Student/Teacher Alignment

Student 始终保持每视角 273 个 queries。



设计一个 `3DFeatureAligner`：
对齐这273个tokens
## 4. Multi-layer Feature Distillation

配置 teacher layers:
[4, 11, 17, 23]

Student 配置四个对应 feature heads。
先核对 Future3DExpert 层数和中间层输出接口，再确定 student layer mapping。

每个 head 将 student hidden_dim 投影到 teacher 的 2048 维。

实现：
- normalized feature MSE
- cosine feature loss
- loss 日志

## 5. 因果依赖

3D Expert 可以读取：
- 当前 RGB+TEXT observation
- 未来 RGB 的token
- 未来 motion的token
- 自身 3D token
- 文本条件

3D Expert 不能读取：
- 真实未来 RGB 的干净 latent
- 真实未来 motion 的干净 latent
- 真实 goal RGB 的视觉 embedding
- 真实 teacher features 作为 forward 条件


要区分 flow matching 的 noisy target 输入与模型真正生成的未来预测。



可以实现可配置的一步 future estimate 作为实验选项，但如果它是从混有真实未来 target 的 noisy latent 得到的，不得把这种训练称为严格无泄漏训练。

## 6. 3D Inference

实现只使用当前 RGB、文本和生成的 future RGB/Motion 表征进行 3D 预测。

推理时绝对不能加载真实 endpoint RGB 或 VGGT teacher。

最终能输出固定的 student 3D features
