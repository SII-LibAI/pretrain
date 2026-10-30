"""Dataset discovery and launcher dry runs; no model construction or training."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import draccus
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset, parse_repo_ids
from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA1_5Config, WSA1_5DatasetConfig


PROJECT = Path(__file__).resolve().parents[1]


def add_dataset(root):
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta/info.json").write_text(json.dumps({"total_frames": 40, "total_episodes": 1}), encoding="utf-8")
    (root / "data").mkdir(exist_ok=True)
    (root / "videos").mkdir(exist_ok=True)
    episode_dir = root / "meta/episodes/chunk-000"
    episode_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"episode_index": [0], "tasks": [["knit"]],
                             "dataset_from_index": [0], "dataset_to_index": [40]}), episode_dir / "file-000.parquet")


@pytest.fixture
def bash():
    git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
    bash = str(git_bash) if sys.platform == "win32" and git_bash.is_file() else shutil.which("bash")
    if not bash:
        pytest.skip("Bash is unavailable")
    return bash


@pytest.fixture
def launch(monkeypatch, tmp_path, bash):
    for name in ["RESUME_CONFIG", "RGB_LR", "MOTION_LR", "GOAL_LR", "HEADS_LR", "LAMBDA_RGB",
                 "LAMBDA_MOTION", "LAMBDA_3D", "FEATURE_MSE_WEIGHT", "FEATURE_COSINE_WEIGHT",
                 "SCHEDULER_WARMUP_STEPS", "SCHEDULER_DECAY_STEPS",
                 "PROC_PER_NODE", "NODE_COUNT", "WEIGHT_RULES_PATH", "ENABLE_3D_QUERIES", "FUTURE_FRAMES",
                 "USE_SUBTASK"]:
        monkeypatch.delenv(name, raising=False)

    default_root = tmp_path / "parent/fixture"
    add_dataset(default_root)

    def run(extra_args=(), **overrides):
        env = {**os.environ, "WSA_DRY_RUN": "1", "CONDA_PREFIX": "/tmp/wsa15-conda",
               "DATA_ROOT": default_root.parent.as_posix(), **overrides}
        result = subprocess.run([bash, "launch/wsa_1_5_pretrain.sh", *extra_args], cwd=PROJECT, env=env,
                                capture_output=True, text=True, check=True, encoding="utf-8")
        command = shlex.split(result.stdout)
        return dict(arg[2:].split("=", 1) for arg in command if arg.startswith("--") and "=" in arg)

    return run


def parse_policy(arguments):
    return draccus.parse(WSA1_5Config, args=["--device=cpu", *[
        f"--{key.removeprefix('policy.')}={value}" for key, value in arguments.items()
        if key.startswith("policy.") and key != "policy.type"
    ]])


@pytest.mark.parametrize("image_augment,object_paste", [("false", "false"), ("true", "false"),
                                                       ("false", "true"), ("true", "true")])
def test_launcher_passes_independent_rgb_augmentation_switches(launch, image_augment, object_paste):
    args = launch(IMAGE_AUGMENT=image_augment, OBJECT_PASTE=object_paste,
                  IMAGE_AUGMENTATION_CONFIG="custom recipe.yaml", PET_NNODES="1", PET_NODE_RANK="0",
                  PET_NPROC_PER_NODE="1", PET_MASTER_ADDR="127.0.0.1", PET_MASTER_PORT="29500")
    assert args["dataset.image_augment"] == image_augment
    assert args["dataset.object_paste"] == object_paste
    assert args["dataset.image_augmentation_config"] == "custom recipe.yaml"
    config = draccus.parse(WSA1_5DatasetConfig, args=["--repo_id=fixture", *[
        f"--{name}={args['dataset.' + name]}"
        for name in ["image_augment", "object_paste", "image_augmentation_config"]
    ]])
    assert config.image_augment == (image_augment == "true")
    assert config.object_paste == (object_paste == "true")


@pytest.mark.parametrize("use_subtask", [None, "false", "true"])
def test_launcher_subtask_prompt_switch_matches_precompute(launch, use_subtask):
    overrides = {} if use_subtask is None else {"USE_SUBTASK": use_subtask}
    args = launch(PET_NNODES="1", PET_NODE_RANK="0", PET_NPROC_PER_NODE="1",
                  PET_MASTER_ADDR="127.0.0.1", PET_MASTER_PORT="29500", **overrides)
    assert args["dataset.use_subtask"] == (use_subtask or "true")


def test_launcher_preserves_assigned_defaults_and_removes_supervision_switch(launch):
    args = launch()
    config = parse_policy(args)
    assert args["num_processes"] == "2"
    assert config.lambda_3d == 1
    assert [config.rgb_lr, config.motion_lr, config.goal_lr, config.heads_lr] == [5e-5] * 4
    assert config.lambda_rgb == config.lambda_motion == 1
    assert (config.feature_mse_weight, config.feature_cosine_weight) == (1, 1)
    assert not config.trainable_weights_fp32 and not config.load_text_encoder
    assert config.scheduler_warmup_steps == 5000
    assert config.scheduler_decay_steps == 300000
    assert config.future_frames == int(args["dataset.future_frames"]) == 16
    assert config.scheduler_decay_lr_ratio == 0.1
    assert config.use_3d_supervision
    assert "policy.enable_3d_supervision" not in args
    assert not hasattr(config, "enable_3d_supervision")


def test_launcher_loss_and_decay_overrides_reach_policy(launch):
    args = launch(LAMBDA_RGB="0.7", LAMBDA_MOTION="0.3", LAMBDA_3D="0",
                  FEATURE_MSE_WEIGHT="0.4", FEATURE_COSINE_WEIGHT="0.6",
                  SCHEDULER_WARMUP_STEPS="10", SCHEDULER_DECAY_STEPS="100",
                  extra_args=["--policy.scheduler_decay_lr_ratio=0.02"])
    config = parse_policy(args)
    assert (config.lambda_rgb, config.lambda_motion, config.lambda_3d) == (0.7, 0.3, 0)
    assert (config.feature_mse_weight, config.feature_cosine_weight) == (0.4, 0.6)
    assert config.enable_3d_queries and not config.use_3d_supervision
    for schedule in config.get_scheduler_preset().group_schedules.values():
        assert schedule["num_warmup_steps"] == 10
        assert schedule["num_decay_steps"] == 100
        assert schedule["decay_lr"] == pytest.approx(schedule["peak_lr"] * 0.02)


@pytest.mark.parametrize("frames", [4, 12, 16, 32, 48])
def test_launcher_passes_same_future_frames_to_dataset_and_policy(launch, frames):
    args = launch(FUTURE_FRAMES=str(frames))
    assert int(args["dataset.future_frames"]) == frames
    assert parse_policy(args).future_frames == frames


@pytest.mark.parametrize("repos", [["."], ["one", "a/b/c"], [".", "nested/child"]])
def test_launcher_discovery_reaches_actual_metadata_paths(launch, tmp_path, monkeypatch, repos):
    root = tmp_path / "data collection"
    for repo in repos:
        add_dataset(root / repo)
    args = launch(DATA_ROOT=root.as_posix(), DATASET_REPO_ID="['.']", DATASET_REPO_IDS='["ignored"]')
    cfg = draccus.parse(TrainPipelineConfig, args=["--policy.device=cpu", *[
        f"--{key}={value}" for key, value in args.items() if key.startswith(("policy.", "dataset."))
    ]])
    assert set(parse_repo_ids(cfg.dataset.repo_id)) == set(repos)
    assert Path(cfg.dataset.root) == root
    built = []

    def build(config, repo_id, transforms, rank):
        path = Path(config.dataset.root) / repo_id
        assert (path / "meta/info.json").is_file()
        built.append(path.resolve())
        return SimpleNamespace(repo_id=repo_id), {}, "fixture"

    monkeypatch.setattr("lerobot.datasets.factory._build_single_dataset", build)
    monkeypatch.setattr("lerobot.datasets.factory.MultiLeRobotDataset", lambda datasets, **kwargs: datasets)
    # Real metadata reads execute before the stub; no video decoding or model construction.
    make_dataset(cfg)
    assert set(built) == {(root / repo).resolve() for repo in repos}


@pytest.mark.parametrize("kind", ["missing", "empty", "incomplete", "whitespace"])
def test_discovery_errors_stop_before_launch(launch, tmp_path, kind):
    root = tmp_path / "scan"
    if kind == "empty":
        root.mkdir()
    elif kind == "incomplete":
        (root / "child/meta").mkdir(parents=True)
        (root / "child/data").mkdir()
    elif kind == "whitespace":
        add_dataset(root / "bad child")
    with pytest.raises(subprocess.CalledProcessError) as failure:
        launch(DATA_ROOT=root.as_posix())
    assert failure.value.returncode != 0
    assert not failure.value.stdout.strip()


def test_launcher_resume_uses_saved_hyperparameters(launch):
    args = launch(RESUME_CONFIG="/tmp/run/train_config.json", LAMBDA_RGB="0.7",
                  SCHEDULER_DECAY_STEPS="100")
    assert args["config_path"] == "/tmp/run/train_config.json"
    assert args["resume"] == "true"
    assert not any(key.startswith("policy.") for key in args)


def test_streaming_is_explicitly_unsupported():
    assert not WSA1_5DatasetConfig(repo_id="fixture").streaming
    with pytest.raises(ValueError, match="map-style"):
        WSA1_5DatasetConfig(repo_id="fixture", streaming=True)


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan"), float("inf")])
def test_decay_ratio_rejects_invalid_values(ratio):
    with pytest.raises(ValueError, match="scheduler_decay_lr_ratio"):
        WSA1_5Config(device="cpu", scheduler_decay_lr_ratio=ratio)
