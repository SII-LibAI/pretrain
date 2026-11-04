# WSA1.5 RoboTwin 评测

使用 Stage-2 Action checkpoint，在 RoboTwin 上执行 ALOHA 双臂关节控制。
这是独立的 WSA 评测入口，沿用原评测 `../inference.py` 所对应的 RoboTwin 环境接口、
任务顺序和有效 seed 筛选流程。原有评测代码、脚本和 README 保持原样。
本目录中 `inference.py` 负责评测流程，`policy.py` 负责 WSA 观测和动作适配。

## 动作和归一化

默认 `ROBOT_TYPE=aloha`，控制模式为 `joint`，环境执行 `action_type="qpos"`。
原始 state 和环境 action 都是 **14 维**：

```text
[左臂关节 6 维, 左夹爪 1 维, 右臂关节 6 维, 右夹爪 1 维]
 0..5          6             7..12          13
```

输入读取 `observation["joint_action"]["vector"]`，使用训练时相同的
`ActionCodec` 和 `aloha` schema。归一化统计对应 `observation.state` 和 `action`，
两项的 mean/std 均为原始 14 维向量。默认读取 checkpoint 的 `stats.json["aloha"]`；
也可以显式设置 `STATS_PATH=/path/to/stats.json` 或 `EXTERNAL_STATS_PATH`，
指向训练时实际使用的普通 per-feature 统计文件。

仓库当前 `aloha.yaml` 定义了 14→16 的模型内部补位重排，夹爪被放到内部索引 7、15，
空位是内部索引 6、14。评测保持 checkpoint 的 `state_dim` / `action_dim`：
输入按 schema 重排；预测经过反归一化和 delta 还原后，再按 schema **逆重排回 14 维**。
模型内部的补位不能作为环境控制维度，也不表示末端位姿。
不要为了让环境接收 14 维动作而修改已训练 checkpoint 的模型维度。
如果训练使用了自定义 schema，评测环境也必须加载相同的定义。

`abs` / `delta` 从 checkpoint 读取。delta 模式只给 schema 指定的关节坐标加回
**本次规划时的原始 state**，夹爪保持绝对目标；执行缓存动作时不重复加当前 state。
输出仅将 14 维动作中的夹爪索引 6、13 截到 `[0,1]`，关节坐标保持预测值。

## 环境准备

复用原仓库 RoboTwin 评测使用的 **同一个版本、同一套安装**，目录仍为
`third_party/RoboTwin`。环境和资产按原评测的既有方式准备；本适配不指定新的
RoboTwin revision，不要求切换分支或 checkout，也不覆盖 RoboTwin 的任何源码。
使用已经装好 WSA1.5 和原评测仿真依赖的 Conda 环境。

## 单任务评测

```bash
TEXT_ENCODER_PATH=/path/to/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth \
TOKENIZER_PATH=/path/to/Wan2.2-TI2V-5B/google/umt5-xxl \
INFER_HORIZON=16 NUM_EPISODES=100 \
bash evaluation/RoboTwin/wsa1_5/eval.sh \
  /path/to/stage2/checkpoints/last/pretrained_model \
  outputs/robotwin/wsa1_5/demo_randomized/adjust_bottle \
  demo_randomized 0
```

`demo_clean` 是 Clean 评测，`demo_randomized` 是 Clean2Random 评测。
最后的 `0` 是任务索引，0–49 的顺序见 `inference.py` 中的 `TASK_NAMES`。
`INSTRUCTION_TYPE=unseen`、`SEED=42`、`FPS=30` 为默认值。

`INFER_HORIZON=0` 是 WSA 默认值，表示执行完整的 checkpoint 动作 chunk；
正数表示执行多少个动作后重新规划，不能超过 `action_chunk_size`。
checkpoint 必须使用 `action_frame_stride=1`。

## 50 任务多卡评测

```bash
GPUS=0,1 \
TASK_CONFIG=demo_clean \
OUTPUT_ROOT="/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series/evaluation/RoboTwin/output/eval_10000_demo_clean" \
TEXT_ENCODER_PATH="/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth" \
TOKENIZER_PATH="/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/Wan2.2-TI2V-5B/google/umt5-xxl" \
bash evaluation/RoboTwin/wsa1_5/eval_ngpu.sh
```

复用现有任务调度器，每个 worker 独立加载模型和仿真器。先使用每卡一个 worker。
参数依次为 checkpoint、输出根目录、task config、起始任务索引、结束任务索引、episode 数。

## 图像、历史和文本

- 图像使用训练时的 WSA 拼图：上方 2/3 是 head camera，下方左右分别是 wrist camera。
  保留缺失相机 mask；存在 `train_config.json` 时使用保存的 `view_keys` / `view_names`。
- 图像尺寸、future 帧数、历史数量与步长、joint/independent VAE 编码方式均沿用 checkpoint。
  每个控制步都记录图像，包括执行缓存动作的步骤；每个 episode 重置历史和动作队列。
- `ACTION_MODE=auto`、`DTYPE=auto` 为默认值；显式值必须与 checkpoint 一致。
  WSA 拼图尺寸完全由 checkpoint 决定。
- prompt 复用训练的 `action_prompt`，`Control Mode: joint`；使用环境任务指令，
  不提供真实子任务、未来图像、motion 或终点标签，也不应用训练图像增广。

离线文本 checkpoint 可能没有随机评测指令的缓存。提供示例中的 UMT5/tokenizer 路径，
可以按需编码新指令。外部 UMT5 冻结，默认 `TEXT_DEVICE=cpu`，每个 worker 加载一次，
需要足够主机内存；显存充足时可以设置 `TEXT_DEVICE=cuda`。
`TEXT_CACHE=/path/to/wsa1_5_text_embeddings` 可复用已有缓存；缓存缺失时使用外部 UMT5，
没有提供编码器路径则明确报错。`load_text_encoder=true` 的 checkpoint 使用自身保存的
UMT5 权重；原 tokenizer 路径失效时，通过 `TOKENIZER_PATH` 指定新位置。

## 输出与检查

视频记录 WSA 观测拼图，命名为 `success_<id>.mp4` / `failure_<id>.mp4`。
多卡日志位于 `OUTPUT_ROOT/logs/TASK_CONFIG/`，终端会输出每个任务的成功率。
调度器跳过已完成任务；WSA 不覆盖已有 episode 视频，未完成任务重跑时应使用新的输出目录。

只检查命令、不启动仿真时设置 `ROBOTWIN_DRY_RUN=1`。汇总多任务结果：

```bash
python util_scripts/robotwin_result_stats.py outputs/robotwin/wsa1_5_clean2random
```

CPU 测试使用合成 14 维 joint 观测和小尺寸随机专家，检查 abs/delta、schema 补位与逆重排、
state 归一化、历史与队列重置、文本编码，以及单卡/多卡命令；完整模型 GPU 仿真需在服务器验证。

## 与原评测接口的对应

- 环境配置通过原评测使用的 `envs.CONFIGS_PATH` 读取，不改变任务配置和 domain randomization。
- seed 从 `100000 * (1 + seed)` 开始递增，先验证专家轨迹，再用同一 seed 重建环境，
  设置指令、重置策略，以 `eval_success` 判断成功；无效专家 seed 不计入评测 episode 数。
- `get_obs` 提供三路 RGB 与 `joint_action.vector`，`take_action(..., action_type="qpos")`
  接收绝对关节目标和归一化夹爪目标，由原环境负责轨迹插值与物理步进。
- 保持原环境的 state 定义，不另行读取或替换 measured qpos。

本入口自行保存 WSA 拼图视频，显式关闭环境的 `eval_video_save_dir`，避免旧版
`take_action` 访问尚未安装的 ffmpeg 管道。策略或视频保存异常会清理环境并向上抛出；
第二次环境初始化出现 `UnStableError` 时跳过该 seed，不消耗评测 episode 名额。

如需运行直接调用官方历史方法的 CPU 契约测试，指定历史仓库源码路径：

```bash
ROBOTWIN_OFFICIAL_SOURCE=/path/to/the/same/RoboTwin \
PYTHONPATH=src:. pytest -q tests/test_wsa1_5_robotwin.py
```

测试从源码提取实际 `get_obs`、`get_rgb`、关节 getter、`take_action`、`eval_policy` 执行，
只替换渲染、物理场景和 TOPP planner；该检查不能替代 SAPIEN 上的完整 rollout。
