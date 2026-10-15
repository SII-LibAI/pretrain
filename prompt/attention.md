现在单独开发 `wsa_1_5` 的 Causal MoT Attention 模块。注意你只负责写代码和简单调试，不能加载模型，训练模型，这些是在服务器上进行的。

先不加载 Wan2.2 权重。使用小型随机 Transformer 测试所有注意力规则。

### 输入 groups

O: current RGB observation tokens,text tokens
R: future RGB tokens
M: future motion tokens
G: goal 3D tokens

严格采用：

          O  R  M  G
O         1  0  0  0
R         1  1  0  0
M         1  0  1  0
G         1  1  1  1

实现 `build_causal_attention_mask()`。

要求：
1. 根据真实 token lengths 动态构造 mask。
2. 可兼容不同 expert hidden dimensions。
3. 保留各 expert 自己的 Q/K/V projection、FFN 和 modulation。
4. 优先复用 WSA-L MoT 的模块化思路。
5. 只允许在显式授权的分支间交换信息。
6. 处理 padding token 和无效视角的 key visibility。
7. 选择合适的PyTorch 版本金额可正确支持自定义 mask 的 attention backend。
8. 如果高性能 attention kernel 不支持这个 mask，使用正确的替代实现并记录性能开销，绝对不允许改成 full attention。
