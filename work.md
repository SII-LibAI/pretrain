我现在有如下的工作要做，在InternVLA-A-series的仓库下新增一个policy，名字为WSA1_5并打算做预训练。
数据格式：
预训练数据包含多种多源的数据集，他们由标准的LerobotV3数据集改造来，代码库里现有的dataset已经可以正常加载
除了标准的Lerobot数据信息，还包含meta/sidecar.json，代表分割的子任务。
对于特殊的人类数据，他们不但进行了机器人本体合成，还包含2D，3D的分割，跟踪信息等，具体标注格式见E:\VLA相关\wam_code\InternVLA-A-series\FINAL_OUTPUT_FORMAT.md。在之前的工作中，对于机器人本体合成的数据已经做了相应的dataset。要注意并非所有的数据集都包含补充的信息，但他们至少是一个完整的LerobotV3数据集，E:\VLA相关\wam_code\egodex-demo这个是一个demo数据集的实例。

我的Policy想要如下设计，模型是一个世界模型，设计一个多专家的MoT架构。
模型的输入是当前观测，指令，本体状态和历史记忆，记忆的形式还没想好，可以先留空。
模型的核心架构是多专家MoT混合Transformer架构,MoT的实现方式可以参考E:\VLA相关\wam_code\WSA。
模型预测输出：
第一层物理信息：当前子任务，3D Pose以文本的形式表达
第二层输出：未来的2D mask和3D变化，不显式预测，以图像的形式预测是未来的motion_mask和motion_mask_ids
第三层输出：动作块Action chunk。
模型作为WSA的1.5扩展工作，模型架构等细节方面的设计优先参考WSA。
请你为我撰写WSA1.5的模型代码，最终在InternVLA-A-series的仓库下，应该在src/lerobot/policies和src/lerobot/datasets下有WSA1.5的补充代码，其他部分尽可能不做改动，模型的预训练，微调启动接口也保持不变，只是替换了dataset和policy，模型仍在internvla-a-series的framework下训练与评估。