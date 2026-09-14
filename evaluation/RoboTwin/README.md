# RoboTwin Evaluation

RoboTwin is the only bundled open evaluation entry.

## Setup

```bash
git submodule update --init third_party/RoboTwin
cp evaluation/RoboTwin/requirements.txt third_party/RoboTwin/script/requirements.txt
cd third_party/RoboTwin
bash script/_install.sh
bash script/_download_assets.sh
cd ../..
```

Follow the official RoboTwin documentation if your machine needs additional rendering dependencies.

## Run

```bash
bash evaluation/RoboTwin/eval.sh /path/to/checkpoint outputs/robotwin/internvla_a1_5 demo_clean 0 abs 50
```

Arguments:

- `checkpoint`: local checkpoint directory or Hugging Face repo id.
- `output_path`: directory where replay videos are saved.
- `task_config`: RoboTwin task config, such as `demo_clean` or `demo_randomized`.
- `task_idx`: index into `TASK_NAMES` in `evaluation/RoboTwin/inference.py`.
- `action_type`: `delta` or `abs`.
- `horizon`: number of predicted actions to enqueue per policy call.

## Multi-GPU and resumable evaluation

Run one independent RoboTwin task per GPU:

```bash
GPUS=0,1,2,3 \
INFERENCE_BACKEND=standard \
bash evaluation/RoboTwin/eval_ngpu.sh \
  /inspire/hdd2/project/generative-large-model/zhangjianing-253108140206/pretrain/outputs/internvla_a1_5/2026_09_12_15_23_06-internvla_a1_5-robotwin-abs-a15_pretrain-finetune/checkpoints/120000/pretrained_model \
  outputs/eval-robotwin-randomized \
  demo_randomized \
  0 49 100
```

The last three arguments are the inclusive task range and the required number
of episodes per task. The scheduler considers a task complete when
`success_<n>.mp4` or `failure_<n>.mp4` exists and is non-empty for every
episode from 1 through the requested count. Completed tasks are skipped when
the same command is run again. An interrupted, partially completed task is
restarted, while completed tasks are preserved. Set `FORCE_RERUN=true` to
evaluate every requested task again.

Select a policy with the `POLICY_TYPE` environment variable:

```bash
POLICY_TYPE=pi05 bash evaluation/RoboTwin/eval.sh /path/to/pi05/checkpoint outputs/robotwin/pi05 demo_clean 0
```

Supported values are `pi0`, `pi0_fast`, `pi05`, and `internvla_a1_5`.

For `internvla_a1_5`, the inference entry defaults to `--action-loss-only`, which skips WAN weight loading. Use `--no-action-loss-only` only when the checkpoint and WAN assets are available.

## Results

Replay videos are written as `success_<id>.mp4` or `failure_<id>.mp4`.

To summarize a completed evaluation directory:

```bash
python util_scripts/robotwin_result_stats.py outputs/robotwin/internvla_a1_5
```
