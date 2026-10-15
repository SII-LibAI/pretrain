"""Dataset discovery and launcher dry runs; no model construction or training."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import draccus
import pytest

from lerobot.datasets.configuration_wsa1_5 import WSA1_5DatasetConfig
from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA1_5Config
from util_scripts.discover_wsa1_5_datasets import discover


PROJECT = Path(__file__).resolve().parents[1]


def add_dataset(root):
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text("{}", encoding="utf-8")
    (root / "data").mkdir()


def test_discovery_includes_root_and_all_nested_levels(tmp_path, monkeypatch):
    repos = [".", "one", "group/two", "a/b/c/three"]
    for repo in repos:
        add_dataset(tmp_path / repo)
    (tmp_path / "incomplete/meta").mkdir(parents=True)
    (tmp_path / "incomplete/meta/info.json").write_text("{}")
    (tmp_path / "not_a_dataset/data").mkdir(parents=True)
    assert discover(tmp_path) == sorted(repos)
    monkeypatch.chdir(tmp_path)
    assert discover() == sorted(repos)


def test_discovery_cli_defaults_to_current_directory(tmp_path):
    add_dataset(tmp_path)
    result = subprocess.run([sys.executable, str(PROJECT / "util_scripts/discover_wsa1_5_datasets.py")],
                            cwd=tmp_path, capture_output=True, text=True, check=True, encoding="utf-8")
    assert json.loads(result.stdout) == ["."]


def test_discovery_errors_are_not_an_empty_dataset_list(tmp_path, monkeypatch):
    with pytest.raises(NotADirectoryError):
        discover(tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="No LeRobot"):
        discover(tmp_path)

    def denied(root, *, onerror):
        onerror(PermissionError("unreadable dataset directory"))

    monkeypatch.setattr("util_scripts.discover_wsa1_5_datasets.os.walk", denied)
    with pytest.raises(PermissionError, match="unreadable"):
        discover(tmp_path)


@pytest.fixture
def launch(monkeypatch):
    git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
    bash = str(git_bash) if sys.platform == "win32" and git_bash.is_file() else shutil.which("bash")
    if not bash:
        pytest.skip("Bash is unavailable")
    for name in ["RESUME_CONFIG", "RGB_LR", "MOTION_LR", "GOAL_LR", "HEADS_LR", "LAMBDA_RGB",
                 "LAMBDA_MOTION", "LAMBDA_3D", "FEATURE_MSE_WEIGHT", "FEATURE_COSINE_WEIGHT",
                 "SCHEDULER_WARMUP_STEPS", "SCHEDULER_DECAY_STEPS", "SCHEDULER_DECAY_LR_RATIO",
                 "PROC_PER_NODE", "NODE_COUNT", "WSA_CACHE_DIR", "WEIGHT_RULES_PATH", "ENABLE_3D_QUERIES"]:
        monkeypatch.delenv(name, raising=False)

    def run(**overrides):
        env = {**os.environ, "WSA_DRY_RUN": "1", "CONDA_PREFIX": "/tmp/wsa15-conda",
               "DATASET_REPO_IDS": '["fixture"]', **overrides}
        result = subprocess.run([bash, "launch/wsa_1_5_pretrain.sh"], cwd=PROJECT, env=env,
                                capture_output=True, text=True, check=True, encoding="utf-8")
        command = shlex.split(result.stdout)
        return dict(arg[2:].split("=", 1) for arg in command if arg.startswith("--") and "=" in arg)

    return run


def parse_policy(arguments):
    return draccus.parse(WSA1_5Config, args=["--device=cpu", *[
        f"--{key.removeprefix('policy.')}={value}" for key, value in arguments.items()
        if key.startswith("policy.") and key != "policy.type"
    ]])


def test_launcher_preserves_assigned_defaults_and_removes_supervision_switch(launch):
    args = launch()
    config = parse_policy(args)
    assert args["num_processes"] == "2"
    assert config.lambda_3d == 0.01
    assert [config.rgb_lr, config.motion_lr, config.goal_lr, config.heads_lr] == [5e-5] * 4
    assert config.lambda_rgb == config.lambda_motion == 1
    assert config.feature_mse_weight == config.feature_cosine_weight == 1
    assert config.scheduler_warmup_steps == 2000
    assert config.scheduler_decay_steps == 600000
    assert config.scheduler_decay_lr_ratio == 0.1
    assert config.use_3d_supervision
    assert "policy.enable_3d_supervision" not in args
    assert not hasattr(config, "enable_3d_supervision")


def test_launcher_loss_and_decay_overrides_reach_policy(launch):
    args = launch(LAMBDA_RGB="0.7", LAMBDA_MOTION="0.3", LAMBDA_3D="0",
                  FEATURE_MSE_WEIGHT="0.4", FEATURE_COSINE_WEIGHT="0.6",
                  SCHEDULER_WARMUP_STEPS="10", SCHEDULER_DECAY_STEPS="100",
                  SCHEDULER_DECAY_LR_RATIO="0.02")
    config = parse_policy(args)
    assert (config.lambda_rgb, config.lambda_motion, config.lambda_3d) == (0.7, 0.3, 0)
    assert (config.feature_mse_weight, config.feature_cosine_weight) == (0.4, 0.6)
    assert config.enable_3d_queries and not config.use_3d_supervision
    for schedule in config.get_scheduler_preset().group_schedules.values():
        assert schedule["num_warmup_steps"] == 10
        assert schedule["num_decay_steps"] == 100
        assert schedule["decay_lr"] == pytest.approx(schedule["peak_lr"] * 0.02)


def test_launcher_autodiscovery_passes_current_and_nested_repo_ids(launch, tmp_path):
    add_dataset(tmp_path)
    add_dataset(tmp_path / "a/b/c")
    args = launch(DATA_ROOT=tmp_path.as_posix(), DATASET_REPO_IDS="")
    assert json.loads(args["dataset.repo_id"]) == [".", "a/b/c"]
    assert Path(args["dataset.root"]) == tmp_path


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
