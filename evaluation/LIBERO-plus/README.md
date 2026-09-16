# LIBERO-plus Evaluation

Zero-shot robustness evaluation on
[LIBERO-plus](https://github.com/sylvestf/LIBERO-plus): the same checkpoints we
eval on stock LIBERO, run against ~10k perturbed tasks across 4 suites and 7
perturbation categories (Camera, Robot, Language, Light, Background, Noise,
Layout).

This reuses the websocket policy server from
`evaluation/LIBERO/policy_server/` and the stock-LIBERO env adapter
(`evaluation/LIBERO/model2libero_interface.py::LiberoModelClient`)
**unchanged** — LIBERO-plus bakes every perturbation into its bddl/init files
and keeps the obs keys and action convention identical, so only the env-side
driver differs.

## Layout

| file | purpose |
|------|---------|
| `eval_libero_plus.py` | per-shard eval driver: resume task episodes, bucket results by perturbation category, write `logs/{suite}/{start}_to_{end}.json` only after the shard finishes |
| `resume_utils.py` | atomic per-task progress, run-configuration validation, and completed-shard checks |
| `aggregate_results.py` | merge all per-shard jsons into `overall_results.json` (per-category, per-suite, and a leaderboard summary row) |
| `run_eval_libero_plus.sh` | multi-GPU + sharded launcher (flock work queue of `(suite, shard)` units), aggregates at the end |
| `run_eval_libero_plus_single.sh` | single-GPU smoke test over a small task-id range |

## One-time setup (not done by the scripts)

1. **LIBERO-plus runtime deps** in the `libero_plus` conda env:
   ```bash
   conda activate libero_plus
   pip install -e <LIBERO_PLUS_REPO_ROOT>
   pip install torch torchvision mujoco==3.2.3 robosuite==1.4.0 bddl easydict \
     pyyaml opencv-python imageio imageio-ffmpeg websockets msgpack numpy==1.24.4 wand
   apt install -y libmagickwand-dev libfontconfig1-dev libexpat1
   ```
2. **Perturbation assets**: download `assets.zip` from the LIBERO-plus HF
   dataset and unzip into `<LIBERO_PLUS_REPO_ROOT>/libero/libero/assets/` so it
   contains `textures/ new_objects/ scenes/ stable_hope_objects/
   stable_scanned_objects/ turbosquid_objects/ articulated_objects/
   serving_region.xml wall_frames.stl wall.xml`.

The client side selects LIBERO-plus purely via env vars (the run scripts set
these): `LIBERO_CONFIG_PATH=$EVAL_LOG_DIR/libero_config` (auto-generated to
point at LIBERO-plus paths) and `PYTHONPATH` prepending `$LIBERO_HOME` so the
LIBERO-plus `libero` package shadows stock LIBERO.

## Usage

Two conda envs are used: server in `lerobot_lab` (default), client in
`libero_plus` (default). The run scripts launch both.

**Smoke test** (4 tasks, one GPU):

```bash
export STATS_KEY_MODE=suite
export ROBOT_TYPE_MODE=panda
export CKPT_PATH=<CKPT_PATH>
export LIBERO_HOME=<LIBERO_PLUS_REPO_ROOT>
GPU_ID=0 TASK_SUITE=libero_goal START_IDX=0 END_IDX=4 \
  bash evaluation/LIBERO-plus/run_eval_libero_plus_single.sh
```

**Full run** (all suites, fan out across free GPUs):

```bash
export CKPT_PATH=<CKPT_PATH>
export LIBERO_HOME=<LIBERO_PLUS_REPO_ROOT>
# SHARDS_PER_SUITE controls fan-out; total work units = 4 * SHARDS_PER_SUITE.
SHARDS_PER_SUITE=8 bash evaluation/LIBERO-plus/run_eval_libero_plus.sh
# or pin GPUs:
GPU_IDS=0,1,2,3 SHARDS_PER_SUITE=8 bash evaluation/LIBERO-plus/run_eval_libero_plus.sh
```

**Resume after an instance stops:** set a fixed `EVAL_LOG_DIR` and rerun the
same command with the same checkpoint and evaluation settings:

```bash
export EVAL_LOG_DIR=outputs/sim_eval/libero_plus/my_run
GPU_IDS=0,1,2,3 SHARDS_PER_SUITE=8 bash evaluation/LIBERO-plus/run_eval_libero_plus.sh
# On the next instance, use the same EVAL_LOG_DIR and command again.
```

Each finished episode is recorded atomically in
`progress/{suite}/{task_id}.json`. When videos are enabled, a result is trusted
only if its success/failure MP4 is nonempty; a missing or incomplete video is
rerun. Fully completed shards are skipped before starting the policy server.
Partially completed shards resume at the first unfinished episode of each task.
`logs/{suite}/{start}_to_{end}.json` is committed only after a full shard, so
aggregation never includes an interrupted shard. A changed checkpoint,
classification file, shard layout or evaluation setting is rejected for the
same output directory. Use a new directory for a new run.

Runs produced by older code have only shard-level category counts and cannot
prove which individual tasks or episodes finished. Those shards are evaluated
once with the new code to establish per-task progress; previous shard summaries
are preserved as `*.json.previous.*` but excluded from aggregation.

Common env-var overrides:

- `CKPT_PATH` *(required)*: checkpoint path for the policy server, you can download our fine-tuned ckpt on https://huggingface.co/InternRobotics/InternVLA-A1.5-Libero.
- `LIBERO_HOME` *(required)*: LIBERO-plus repo root.
- `SERVER_ENV` / `CLIENT_ENV` *(defaults `lerobot_lab` / `libero_plus`)*: conda envs.
- `CONDA_ROOT` *(default `$HOME/miniconda3`)*.
- `GPU_IDS` *(e.g. `0,1,2,3`)*: skip auto-detection.
- `GPU_MEM_FREE_THRESHOLD_MB` *(default `30000`)*: free-memory threshold for auto-detection.
- `BASE_PORT` *(default `5774`)*: first port; per-worker ports are `BASE_PORT + worker_idx`.
- `SHARDS_PER_SUITE` *(default `4`)*: task-id shards per suite.
- `NUM_TRIALS_PER_TASK` *(default `1`)*: LIBERO-plus bakes perturbations into task defs, so 1 is standard.
- `STATS_KEY_MODE` / `ROBOT_TYPE_MODE` *(default `suite`)*: `suite` binds stats / robot_type to the LIBERO suite name (for multi-subset stats.json checkpoints); a fixed string (e.g. `panda`) reuses one key across suites.
- `VLM_MODEL_PATH` *(InternVLA-A1.5 only)*: override the VLM path baked into the checkpoint.
- `WAN_MODEL_PATH` / `WAN_VAE_PATH`: only used when overriding `ACTION_LOSS_ONLY_FLAG=--no-action_loss_only`. The default `--action_loss_only` skips WAN weight loading, which matches the public InternVLA-A1.5 release.
- `INFERENCE_BACKEND` *(default `standard`)*: `optimized` requires `--action_loss_only`.
- `NO_VIDEO_FLAG=--no-save_videos` disables video recording (recommended for the ~10k full run).
- `EVAL_LOG_DIR`: destination log dir (default `outputs/sim_eval/libero_plus/<timestamp>_all`); set it explicitly to the same path on every resumed instance.
