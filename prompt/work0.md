请先对当前 InternVLA-A-series 仓库进行代码审计，暂时不要修改代码。

我准备新增一个 `wsa1_5` policy 和 dataset adapter，复用现有预训练训练框架。

请完成以下工作：

1. 找到 policy config 注册、policy factory、dataset factory、processor、DataLoader、collate 和 `lerobot_train.py` 之间的调用链。
2. 找到 InternVLA-A1.5 的 pretrain launcher、loss 返回形式、optimizer/scheduler、参数冻结、混合精度、分布式训练和 checkpoint 保存恢复实现。
3. 检查当前代码是否支持 LeRobot V3 的额外视频键 `observation.motion`，以及多数据集发现和权重混合。
4. 检查现有 WAN 视频分支的真实职责：它是完整生成式视频专家、冻结监督模型，还是 latent feature distillation 模块？不能混淆。
5. 阅读WSA-Base WSA-Large 官方源码，定位 Video Expert、Wan VAE、UMT5、MoT、attention mask、Future3DExpert、flow matching loss 及相关权重初始化模块。
6. 阅读 VGGT-Omega 官方代码，核实中间层特征、max_size 预处理、padding、token 排列及模型加载接口。
7. 确认运行环境的 Python、PyTorch、Transformers、Accelerate 版本，检查与 WSA 组件的依赖兼容性。
8. 检查一个真实数据集的 `meta/info.json`、`meta/sidecar.json` 和 `observation.motion` feature 描述。

输出：

- policy 注册及训练调用链
- 数据输入调用链
- WSA 可复用组件与需要移植的部分
- 具体文件改造表
- 潜在依赖冲突
- 推荐的最小侵入式开发方案
