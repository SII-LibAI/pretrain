现在把已经完成并通过单元测试的 RGB Expert、Motion Expert、Future3DExpert 整合成 `WSA1_5_Policy`，接入当前 InternVLA-A-series 的预训练系统。注意你只负责写代码和简单调试，不能加载模型，训练模型，这些是在服务器上进行的

## 1. 新增 Policy

按照实际仓库的 policy 注册机制新增：

policy type: `WSA_1_5`

复用已有的训练、保存、恢复、logging 和分布式入口。

暂时不需要action字段

## 2. Loss

实现总损失：

L = lambda_rgb * L_rgb_flow
  + lambda_motion * L_motion_flow
  + lambda_3d * L_3d_feature

每个 loss 独立返回并记录。

支持 enable_rgb、enable_motion、enable_3d。

优化器采用参数组，允许 Video Expert、Motion Expert、3D Expert 和 projection heads 使用独立学习率。

Video Expert 使用较保守的预训练权重学习率，新专家和 projection 可以使用不同学习率。

## 3. 训练与推理一致性

实现两个模式：

`training_forward(batch)`:
- 获取真实训练 targets。
- 构造 RGB/Motion flow-matching 训练状态。
- 计算 RGB/Motion flow loss。
- 按 3D conditioning 配置选择无泄漏的模型生成未来表征。
- 计算冻结 Omega teacher 的 feature targets。
- 计算 3D loss。

`predict_future(observed_rgb, text, seed)`:
- 编码当前 RGB 和文本。
- 从噪声采样未来 RGB。
- 从独立噪声采样未来 Motion。
- 3D Expert 读取生成结果并预测目标几何表征。
- 返回 RGB/Motion/3D 输出。

推理中不允许出现当前 motion、未来 GT 或 teacher feature。

## 4. Cache

为了减少训练开销，设计可选缓存：
- text embeddings
- frozen Wan VAE target latents
- Omega teacher endpoint features
训练时可选先离线计算缓存，或者在线学习

真实数据、text prompt 模板及 teacher preprocessing 之间必须保持缓存一致性。

## 5. Distributed Training

使用当前 InternVLA 已有的 Accelerate 入口。

必须支持：
- 多 GPU DDP/Accelerate
- BF16
- gradient accumulation
- gradient checkpointing
- dataset weighting
- checkpoint resume
- 按模块冻结
- 日志记录
- 混合精度与数值稳定性检查
- 模型初始化时日志记录模型各主要模块参数量
不得通过重写完整 trainer 来实现 policy 接入。

## 6. 启动脚本和配置

新增：
- `launch/wsa_1_5_pretrain.sh`

以当前项目真实支持的 CLI/config 体系为准，不能假设 YAML 配置会被自动解析。

先完成最小 synthetic batch 的 1 step 训练，再运行真实 LeRobot 样本的 1 step 训练。

随后执行小样本过拟合实验，确认 loss 能下降，并保存一个可以重新加载的 checkpoint。

## 7. 验收

需要验证：

1. policy.type 可以注册和实例化。
2. dataset.type 可以注册和实例化。
3. 无 action 数据可以训练。
4. 一个真实 batch 的前向和反向通过。
5. 训练可被中断和恢复。
6. checkpoint 保存并重新加载后输出一致。
7. inference 只需要当前 RGB 和文本。
8. 所有因果 mask 测试仍然通过。
9. 在训练结束后，能生成完整的未来 RGB、Motion 和 3D features。

最后输出：
- 所有新增和修改的文件
- 实际训练命令
- 当前尚未解决的问题

不要声称未运行的测试已通过。