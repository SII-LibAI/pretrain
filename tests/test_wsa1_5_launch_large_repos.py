"""Large launcher arguments and config round trips; no models or training."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA1_5Config  # Register the config, without constructing a model.

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bash():
    git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
    executable = str(git_bash) if sys.platform == "win32" and git_bash.is_file() else shutil.which("bash")
    if not executable:
        pytest.skip("Bash is unavailable")
    return executable


def test_large_repo_list_stays_out_of_exec_arguments_and_environment(tmp_path, bash, monkeypatch):
    wrapper = tmp_path / "large-list.sh"
    wrapper.write_text('''#!/usr/bin/env bash
set -euo pipefail
DATASET_REPO_ID=""
for ((i=0; i<12000; i++)); do
    printf -v repo '/tmp/large-repository-%06d ' "$i"
    DATASET_REPO_ID+="$repo"
done
export DATASET_REPO_ID
source launch/wsa_1_5_midtrain.sh
''', encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "accelerate"
    executable.write_text('''#!/usr/bin/env bash
set -euo pipefail
[[ -z "${DATASET_REPO_ID+x}" ]]
printf '%s\\0' "$@" > "$CAPTURE_FILE"
''', encoding="utf-8")
    executable.chmod(0o755)
    captured = tmp_path / "arguments.bin"
    env = {**os.environ, "WSA_DRY_RUN": "0", "CONDA_PREFIX": "/tmp/wsa15-conda",
           "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"], "TMPDIR": tmp_path.as_posix(),
           "CAPTURE_FILE": captured.as_posix(), "INIT_CHECKPOINT_PATH": "/tmp/stage1",
           "RETARGET_STATS_DIR": "/tmp/stats", "PET_NNODES": "1", "PET_NPROC_PER_NODE": "1",
           "PET_NODE_RANK": "0"}
    env.pop("RESUME_CONFIG", None)
    subprocess.run([bash, wrapper.as_posix()], cwd=PROJECT, env=env, capture_output=True,
                   text=True, encoding="utf-8", check=True)
    command = captured.read_text(encoding="utf-8").split("\0")[:-1]
    assert len(captured.read_bytes()) < 10000
    arguments = command[command.index("src/lerobot/scripts/lerobot_train.py") + 1:]
    assert not any(argument.startswith("--dataset.repo_id=") for argument in arguments)
    assert any(argument.startswith("--yaml_path=") for argument in arguments)

    @parser.wrap()
    def read_config(cfg: TrainPipelineConfig):
        return cfg

    if sys.platform == "win32":
        arguments = ["--yaml_path=" + subprocess.check_output(
            [bash, "-c", 'cygpath -w "$1"', "bash", argument.split("=", 1)[1]], text=True).strip()
            if argument.startswith("--yaml_path=") else argument for argument in arguments]
    monkeypatch.setattr(sys, "argv", ["lerobot_train.py", *arguments, "--policy.device=cpu"])
    config = read_config()
    expected = [f"/tmp/large-repository-{index:06d}" for index in range(12000)]
    assert config.dataset.repo_id.split() == expected
    # Checkpoint configs retain the expanded IDs, so resume needs no temporary YAML.
    config._save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "train_config.json").read_text(encoding="utf-8"))
    assert saved["dataset"]["repo_id"].split() == expected


@pytest.mark.parametrize("script", ["wsa_1_5_pretrain.sh", "wsa_1_5_midtrain.sh"])
def test_discovered_repos_and_resume_with_file_config(tmp_path, bash, script):
    root = tmp_path / "data collection"
    for repo in ("one", "nested/two"):
        for folder in ("meta", "data", "videos"):
            (root / repo / folder).mkdir(parents=True)
        (root / repo / "meta/info.json").write_text("{}", encoding="utf-8")
    env = {**os.environ, "WSA_DRY_RUN": "1", "CONDA_PREFIX": "/tmp/wsa15-conda",
           "DATA_ROOT": root.as_posix(), "TMPDIR": tmp_path.as_posix(),
           "INIT_CHECKPOINT_PATH": "/tmp/stage1", "RETARGET_STATS_DIR": "/tmp/stats",
           "PET_NNODES": "1", "PET_NPROC_PER_NODE": "1", "PET_NODE_RANK": "0",
           "PET_MASTER_ADDR": "127.0.0.1", "PET_MASTER_PORT": "29500"}
    for name in ("RESUME_CONFIG", "DATASET_REPO_ID"):
        env.pop(name, None)
    def run():
        result = subprocess.run([bash, f"launch/{script}"], cwd=PROJECT, env=env, capture_output=True,
                                text=True, encoding="utf-8", check=True)
        command = shlex.split(result.stdout.splitlines()[-1])
        return command[command.index("src/lerobot/scripts/lerobot_train.py") + 1:]
    arguments = run()
    path = next(argument.split("=", 1)[1] for argument in arguments if argument.startswith("--yaml_path="))
    if sys.platform == "win32":
        path = subprocess.check_output([bash, "-c", 'cygpath -w "$1"', "bash", path], text=True).strip()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    repos = payload["dataset"]["repo_id"].split()
    assert len(repos) == 2
    assert set(repos) == {"one", "nested/two"}
    assert not any(argument.startswith("--dataset.repo_id=") for argument in arguments)
    env["RESUME_CONFIG"] = "/tmp/run/train_config.json"
    assert run() == ["--config_path=/tmp/run/train_config.json", "--resume=true"]


