"""RoboTwin contracts with synthetic observations and tiny experts; no simulator/assets."""
import json
import ast
from copy import deepcopy
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from types import MethodType, SimpleNamespace
from typing import Literal

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from evaluation.RoboTwin.wsa1_5 import inference
from evaluation.RoboTwin.wsa1_5.policy import WSA1_5Agent, build_observation, load_codec, prepare_action
from lerobot.policies.wsa1_5.dataset_wsa1_5 import action_prompt, compose_rgb_views
from lerobot.policies.wsa1_5.text_embeddings import PROMPT_VERSION, text_embedding_path
from test_wsa1_5_action import action_config, action_policy


PROJECT = Path(__file__).resolve().parents[1]


def test_task_indices_follow_the_existing_evaluation_entry():
    source = (PROJECT / "evaluation/RoboTwin/inference.py").read_text(encoding="utf-8")
    assignment = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "TASK_NAMES" for target in node.targets))
    assert inference.TASK_NAMES == ast.literal_eval(assignment.value)


def codec_fixture(tmp_path, mode="delta"):
    cfg = action_config(state_dim=16, action_dim=16, action_mode=mode, history_frame_stride=2)
    mean = [1.] * 14
    mean[6] = mean[13] = .5
    stats = {key: {"mean": mean.copy(), "std": [2.] * 14} for key in ("observation.state", "action")}
    (tmp_path / "stats.json").write_text(json.dumps({"aloha": stats}), encoding="utf-8")
    return cfg, load_codec(tmp_path, cfg, "aloha"), stats


def observation(position=2, head=255):
    return {"observation": {
        "head_camera": {"rgb": np.full((12, 14, 3), [head, 0, 0], np.uint8)},
        "left_camera": {"rgb": np.full((8, 10, 3), [0, 255, 0], np.uint8)},
        "right_camera": {"rgb": np.full((8, 10, 3), [0, 0, 255], np.uint8)}},
        "joint_action": {"vector": np.array([position, 3, 4, 5, 6, 7, .2, 8, 9, 10, 11, 12, 13, .8], np.float32)}}


def save_text(directory, cfg, text="move"):
    directory.mkdir(exist_ok=True)
    prompt = action_prompt(text, "joint", cfg.action_mode)
    save_file({"context": torch.ones(2, cfg.text_dim, dtype=torch.bfloat16),
               "valid": torch.ones(2, dtype=torch.bool)}, text_embedding_path(directory, prompt),
              metadata={"prompt_version": PROMPT_VERSION, "text": prompt,
                        "max_length": str(cfg.max_text_length), "text_dim": str(cfg.text_dim)})


class RecordingPolicy:
    def __init__(self, cfg):
        self.config, self.calls, self.resets = cfg, [], 0

    def reset(self):
        self.resets += 1

    def predict_action_chunk(self, batch, *, codec, raw_state, seed):
        self.calls.append((batch, raw_state, seed))
        assert "targets" not in batch and "history_rgb" in batch
        return codec.decode(torch.zeros(1, self.config.action_chunk_size, self.config.action_dim), raw_state)


def test_observation_matches_training_layout_and_codec(tmp_path):
    cfg, codec, _ = codec_fixture(tmp_path)
    obs = observation()
    keys = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
    batch, raw, canvas = build_observation(obs, codec, keys, cfg)
    views = [torch.from_numpy(obs["observation"][name]["rgb"]).permute(2, 0, 1)[None].float() / 255
             for name in ("head_camera", "left_camera", "right_camera")]
    expected, valid = compose_rgb_views(views, [True] * 3, cfg.rgb_height, cfg.rgb_width)
    torch.testing.assert_close(batch["observed_rgb"], expected * 2 - 1)
    torch.testing.assert_close(batch["observed_valid"], valid)
    torch.testing.assert_close(canvas, expected[0])
    torch.testing.assert_close(batch["state"][0], codec.encode({key: value[0] for key, value in raw.items()})["state"])
    torch.testing.assert_close(raw["observation.state"][0], torch.from_numpy(obs["joint_action"]["vector"]))
    assert not batch["state"][0, [6, 14]].any()
    del obs["observation"]["right_camera"]
    missing, _, _ = build_observation(obs, codec, keys, cfg)
    assert not missing["observed_valid"][:, :, 64:, 32:].any()
    assert (missing["observed_rgb"][:, :, 64:, 32:] == -1).all()
    obs["observation"]["head_camera"]["rgb"] = np.zeros((5, 5, 3), np.float32)
    with pytest.raises(ValueError, match="uint8"):
        build_observation(obs, codec, keys, cfg)


@pytest.mark.parametrize("mode", ["delta", "abs"])
def test_replans_use_planning_state_history_and_reset(tmp_path, mode):
    cfg, codec, _ = codec_fixture(tmp_path, mode)
    save_text(tmp_path / "cache", cfg)
    policy = RecordingPolicy(cfg)
    agent = WSA1_5Agent(policy, codec, infer_horizon=2, text_cache=tmp_path / "cache")
    agent.reset(80)
    first, image = agent.step(observation(2, 255), "move")
    second, _ = agent.step(observation(3, 160), "move")
    third, _ = agent.step(observation(4, 80), "move")
    assert image.shape == (96, 64, 3) and image.dtype == np.uint8
    assert len(policy.calls) == 2 and [call[2] for call in policy.calls] == [80, 81]
    assert first[0] == pytest.approx(3 if mode == "delta" else 1)
    expected = np.asarray([1.] * 14, np.float32)
    if mode == "delta":
        expected += observation(2)["joint_action"]["vector"]
    expected[[6, 13]] = .5
    np.testing.assert_allclose(first, expected)
    np.testing.assert_equal(second, first)
    assert third[0] == pytest.approx(5 if mode == "delta" else 1)
    assert not policy.calls[0][0]["history_valid"].any()
    assert policy.calls[1][0]["history_valid"][:, -1].all()
    assert policy.calls[1][0]["history_rgb"][0, -1, 0, 0, 0] == 1
    assert policy.calls[1][0]["state"].shape == (1, 16)
    assert "Control Mode: joint" in policy.calls[0][0]["condition_text"][0]
    assert policy.calls[0][0]["condition_text"][0].endswith(f"Action Mode: {mode}")
    with pytest.raises(ValueError, match="reset"):
        agent.step(observation(), "changed task")
    agent.reset(90)
    agent.step(observation(), "move")
    assert policy.calls[-1][2] == 90 and not policy.calls[-1][0]["history_valid"].any()


def test_external_stats_override_and_invalid_layout(tmp_path):
    cfg, _, stats = codec_fixture(tmp_path)
    stats["observation.state"]["mean"] = [10.] * 14
    path = tmp_path / "override.json"
    path.write_text(json.dumps(stats), encoding="utf-8")
    codec = load_codec(tmp_path, cfg, "aloha", path)
    assert codec.stats["observation.state"]["mean"].tolist() == [10.] * 14
    del stats["action"]
    path.write_text(json.dumps(stats), encoding="utf-8")
    with pytest.raises(ValueError, match="14D"):
        load_codec(tmp_path, cfg, "aloha", path)


def test_action_conversion_inverts_schema_and_clamps_only_grippers(tmp_path):
    cfg, codec, _ = codec_fixture(tmp_path)
    native = np.arange(14, dtype=np.float32)
    native[[6, 13]] = [-1, 2]
    action = np.full(cfg.action_dim, 987., np.float32)
    for lo, hi, start, end in codec.schema.action_reorder:
        action[start:end] = native[lo:hi]
    prepared = prepare_action(action, codec)
    expected = native.copy()
    expected[[6, 13]] = [0, 1]
    np.testing.assert_equal(prepared, expected)
    assert prepared.shape == (14,) and action[6] == action[14] == 987
    assert prepare_action(np.zeros(cfg.action_dim), codec).shape == (14,)
    for bad in (np.full(cfg.action_dim, np.nan), np.zeros(14), np.zeros(13)):
        with pytest.raises(ValueError):
            prepare_action(bad, codec)


def test_new_instruction_requires_cache_or_external_encoder(tmp_path):
    cfg, codec, _ = codec_fixture(tmp_path)
    policy = RecordingPolicy(cfg)
    with pytest.raises(ValueError, match="Offline-text"):
        WSA1_5Agent(policy, codec)
    agent = WSA1_5Agent(policy, codec, text_cache=tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="new instructions"):
        agent.step(observation(), "not in training cache")
    with pytest.raises(ValueError, match="infer_horizon"):
        WSA1_5Agent(policy, codec, infer_horizon=cfg.action_chunk_size + 1)
    cfg.action_frame_stride = 2
    with pytest.raises(ValueError, match="dense control"):
        WSA1_5Agent(policy, codec)


def test_external_text_encoder_is_lazy_frozen_and_reused(tmp_path, monkeypatch):
    from lerobot.policies.wsa1_5.vendor import wan_video_text_encoder as vendor
    from evaluation.RoboTwin.wsa1_5 import policy as wsa1_5

    cfg, codec, _ = codec_fixture(tmp_path)
    calls = []

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(cfg.text_dim))
            calls.append("load")

        def forward(self, ids, valid):
            calls.append("encode")
            return self.weight.expand(*ids.shape, -1)

    class Tokenizer:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, texts, **kwargs):
            return torch.ones(len(texts), cfg.max_text_length, dtype=torch.long), torch.ones(len(texts), cfg.max_text_length)

    monkeypatch.setattr(vendor, "WanTextEncoder", Encoder)
    monkeypatch.setattr(vendor, "HuggingfaceTokenizer", Tokenizer)
    monkeypatch.setattr(wsa1_5, "read_checkpoint", lambda path: {"weight": torch.ones(cfg.text_dim)})
    agent = WSA1_5Agent(RecordingPolicy(cfg), codec, text_encoder_path="unused", tokenizer_path="unused")
    assert not calls
    agent.step(observation(), "move")
    agent.step(observation(), "move")
    assert calls == ["load", "encode"]
    assert not agent.text_encoder.training and not agent.text_encoder.weight.requires_grad
    agent.reset(2)
    agent.step(observation(), "another task")
    assert calls == ["load", "encode", "encode"]


def test_real_tiny_policy_generates_physical_14d_joint_action(tmp_path):
    cfg, codec, _ = codec_fixture(tmp_path)
    policy = action_policy(state_dim=16, action_dim=16, history_frame_stride=2)
    save_text(tmp_path / "cache", cfg)
    agent = WSA1_5Agent(policy, codec, text_cache=tmp_path / "cache")
    action, image = agent.step(observation(), "move")
    assert action.shape == (14,) and np.isfinite(action).all()
    assert image.shape == (96, 64, 3)
    assert agent.action_type == "qpos"
    assert 0 <= action[6] <= 1 and 0 <= action[13] <= 1


@pytest.mark.parametrize("sampler_steps", [None, 5, 20])
def test_checkpoint_loader_preserves_saved_geometry_modes_and_views(tmp_path, monkeypatch, sampler_steps):
    from evaluation.RoboTwin.wsa1_5 import policy as wsa1_5

    cfg, _, _ = codec_fixture(tmp_path)
    cfg.save_pretrained(tmp_path)
    view_keys = ["observation.images.cam_right_wrist", "observation.images.cam_high", ""]
    (tmp_path / "train_config.json").write_text(json.dumps({"dataset": {"view_keys": view_keys}}), encoding="utf-8")
    args = SimpleNamespace(ckpt_path=tmp_path, action_mode="auto", dtype="auto", device="cpu",
                           robot_type="aloha", stats_path=None, infer_horizon=2, action_sampler_steps=sampler_steps,
                           tokenizer_path=None, text_cache=tmp_path / "cache", text_encoder_path=None, text_device="cpu")
    calls = []

    def load(path, *, config):
        calls.append(config)
        return RecordingPolicy(config)

    monkeypatch.setattr(wsa1_5.WSA1_5_Policy, "from_pretrained", load)
    agent = WSA1_5Agent.from_checkpoint(args)
    assert agent.view_keys == view_keys and agent.horizon == 2
    assert calls[0].action_mode == cfg.action_mode and not calls[0].load_text_encoder
    assert agent.config.action_sampler_steps == (cfg.action_sampler_steps if sampler_steps is None else sampler_steps)
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))["action_sampler_steps"] == cfg.action_sampler_steps
    for name in ("future_frames", "rgb_height", "rgb_width", "num_history_frame", "sparse_frame_encoding"):
        assert getattr(agent.config, name) == getattr(cfg, name)
    args.action_mode = "abs"
    with pytest.raises(ValueError, match="action_mode"):
        WSA1_5Agent.from_checkpoint(args)
    args.action_mode, args.dtype = "auto", "bfloat16"
    with pytest.raises(ValueError, match="dtype"):
        WSA1_5Agent.from_checkpoint(args)
    assert len(calls) == 1
    args.action_mode, args.dtype = "auto", "auto"
    for invalid in (0, -1):
        args.action_sampler_steps = invalid
        with pytest.raises(ValueError, match="action_sampler_steps"):
            WSA1_5Agent.from_checkpoint(args)
    assert len(calls) == 1


def test_shared_evaluation_selects_valid_seeds_resets_and_reports(tmp_path, monkeypatch):
    class Unstable(Exception):
        pass

    class Env:
        step_lim, render_freq = 2, 0
        plan_success = True

        def __init__(self):
            self.actions, self.seeds = [], []

        def setup_demo(self, now_ep_num, seed, **kwargs):
            self.episode, self.seed = now_ep_num, seed
            self.take_action_cnt, self.eval_success = 0, False

        def play_once(self):
            if self.seed == 100000:
                raise Unstable("skip")
            return {"info": {}}

        def check_success(self):
            return True

        def close_env(self, **kwargs):
            pass

        def set_instruction(self, instruction):
            self.instruction = instruction

        def get_instruction(self):
            return self.instruction

        def get_obs(self):
            return observation()

        def take_action(self, action, action_type):
            self.actions.append((action, action_type))
            self.take_action_cnt += 1
            self.eval_success = self.episode == 0

    args = SimpleNamespace(task_idx=0, num_episodes=2, seed=0, instruction_type="unseen", video_dir=tmp_path, fps=30)
    env, videos, resets = Env(), [], []
    monkeypatch.setattr(inference, "save_replay_video", lambda path, images, fps: videos.append((path.name, len(images))))
    inference.run_episodes(args, env, {"render_freq": 0, "clear_cache_freq": 10, "task_config": "demo_randomized"},
                           Unstable, lambda *args: [{"unseen": ["move"]}],
                           lambda *args: (np.ones(14), np.zeros((8, 8, 3), np.uint8)),
                           resets.append, "qpos")
    assert resets == [100001, 100002]
    assert videos == [("success_1.mp4", 1), ("failure_2.mp4", 2)]
    assert env.suc == 1 and env.test_num == 2
    assert {kind for _, kind in env.actions} == {"qpos"}
    assert all(action.shape == (14,) for action, _ in env.actions)


def test_wsa_dispatch_requests_joint_observations_and_qpos(tmp_path, monkeypatch):
    import sys
    from evaluation.RoboTwin.wsa1_5 import policy as wsa1_5

    monkeypatch.setattr(sys, "argv", ["inference.py", "--ckpt-path", str(tmp_path)])
    args = inference.parse_args()
    assert args.robot_type == "aloha"
    assert args.action_sampler_steps is None
    task_args = {"data_type": {"endpose": False}}
    agent = SimpleNamespace(step=lambda *args: None, reset=lambda *args: None, action_type=WSA1_5Agent.action_type)
    calls = []
    monkeypatch.setattr(inference, "require_robotwin", lambda: ("unused", RuntimeError, None))
    monkeypatch.setattr(inference, "build_task_args", lambda *args: task_args)
    monkeypatch.setattr(inference, "make_env", lambda name: None)
    monkeypatch.setattr(wsa1_5.WSA1_5Agent, "from_checkpoint", lambda args: agent)
    monkeypatch.setattr(inference, "run_episodes", lambda *args: calls.append(args))
    inference.infer_once(args)
    assert calls[0][-1] == "qpos"
    assert calls[0][2]["data_type"] == {"endpose": False, "qpos": True, "rgb": True}


def bash_executable():
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    if not bash.is_file():
        found = shutil.which("bash")
        if not found:
            pytest.skip("Bash unavailable")
        bash = Path(found)
    return str(bash)


@pytest.mark.parametrize("sampler_steps", [None, "5"])
def test_launcher_keeps_checkpoint_settings_and_paths(tmp_path, monkeypatch, sampler_steps):
    bash = bash_executable()
    checkpoint = tmp_path / "checkpoint with spaces"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"dry run only")
    env = {**os.environ, "ROBOTWIN_DRY_RUN": "1", "CONDA_PREFIX": "/tmp/wsa15-conda",
           "TEXT_CACHE": checkpoint.as_posix(), "STATS_PATH": (checkpoint / "config.json").as_posix()}
    env.pop("ACTION_SAMPLER_STEPS", None)
    if sampler_steps is not None:
        env["ACTION_SAMPLER_STEPS"] = sampler_steps
    result = subprocess.run([bash, "evaluation/RoboTwin/wsa1_5/eval.sh", checkpoint.as_posix(),
                             (tmp_path / "results with spaces").as_posix(), "demo_randomized", "4"],
                            cwd=PROJECT, env=env, check=True, capture_output=True, text=True)
    command = shlex.split(result.stdout.strip())
    options = dict(zip(command[3::2], command[4::2], strict=True))
    assert options["--task-config"] == "demo_randomized" and "--policy-type" not in options
    assert "checkpoint with spaces" in options["--ckpt-path"]
    assert options["--action-mode"] == options["--dtype"] == "auto"
    assert options["--infer-horizon"] == "0" and options["--text-device"] == "cpu"
    assert options["--stats-path"].endswith("config.json")
    assert options["--robot-type"] == "aloha"
    assert options.get("--action-sampler-steps") == sampler_steps
    monkeypatch.setattr("sys.argv", command[2:])
    assert inference.parse_args().action_sampler_steps == (int(sampler_steps) if sampler_steps else None)


def test_multigpu_scheduler_forwards_wsa_defaults_and_collects_task_logs(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"dry run only")
    output = tmp_path / "results"
    env = {**os.environ, "ROBOTWIN_DRY_RUN": "1", "CONDA_PREFIX": "/tmp/wsa15-conda",
           "GPUS": "0,1", "JOBS_PER_GPU": "1", "TMPDIR": tmp_path.as_posix(), "BURNER_ENABLED": "false",
           "PRETRAINED_CKPT": checkpoint.as_posix(), "OUTPUT_ROOT": output.as_posix(),
           "TASK_CONFIG": "demo_randomized", "TASK_START": "0", "TASK_END": "1", "NUM_EPISODES": "1",
           "ACTION_MODE": "auto", "DTYPE": "auto", "INFER_HORIZON": "0", "ACTION_SAMPLER_STEPS": "7",
           "TEXT_ENCODER_PATH": (checkpoint / "config.json").as_posix(), "TOKENIZER_PATH": checkpoint.as_posix()}
    env.pop("FORCE_RERUN", None)
    result = subprocess.run([bash_executable(), "evaluation/RoboTwin/wsa1_5/eval_ngpu.sh"],
                            cwd=PROJECT, env=env, capture_output=True, text=True)
    # A dry run emits commands but creates no videos, so the scheduler must report incomplete tasks.
    assert result.returncode != 0 and "reason=incomplete_output" in result.stdout + result.stderr
    for name in ("adjust_bottle", "beat_block_hammer"):
        command = shlex.split((output / "logs/demo_randomized" / f"{name}.log").read_text(encoding="utf-8").strip())
        options = dict(zip(command[3::2], command[4::2], strict=True))
        assert "--policy-type" not in options
        assert options["--dtype"] == options["--action-mode"] == "auto"
        assert options["--infer-horizon"] == "0"
        assert options["--action-sampler-steps"] == "20"


@pytest.fixture
def official_legacy_function():
    """Run selected actual historical methods without importing SAPIEN or simulator assets."""
    root = Path(os.environ.get("ROBOTWIN_OFFICIAL_SOURCE", PROJECT / "third_party/RoboTwin"))
    if not (root / "script/eval_policy.py").is_file():
        pytest.skip("Set ROBOTWIN_OFFICIAL_SOURCE to an official historical RoboTwin checkout")

    def load(path, name):
        source = root / path
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
        namespace = {"np": np, "deepcopy": deepcopy, "Literal": Literal}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        return namespace[name]

    return load


@pytest.mark.parametrize("mode", ["abs", "delta"])
def test_official_get_obs_and_take_action_accept_codec_roundtrip(tmp_path, official_legacy_function, mode):
    load = official_legacy_function
    cfg, codec, _ = codec_fixture(tmp_path, mode)

    class Joint:
        def __init__(self, value):
            self.value = value

        def get_drive_target(self):
            return np.array([self.value])

    def bind(obj, path, names):
        for name in names:
            setattr(obj, name, MethodType(load(path, name), obj))

    robot = SimpleNamespace(left_arm_joints=[Joint(float(i)) for i in range(6)],
                            right_arm_joints=[Joint(float(i)) for i in range(7, 13)],
                            left_gripper=[], right_gripper=[], left_gripper_val=.2, right_gripper_val=.8)
    bind(robot, "envs/robot/robot.py", ["get_left_arm_jointState", "get_right_arm_jointState",
                                     "get_left_gripper_val", "get_right_gripper_val"])
    paths = []

    def topp(path, *args, **kwargs):
        paths.append(path.copy())
        return None, path, np.zeros_like(path), None, None

    def joints(position, velocity, side):
        for joint, value in zip(getattr(robot, f"{side}_arm_joints"), position, strict=True):
            joint.value = float(value)

    robot.left_mplib_planner = robot.right_mplib_planner = SimpleNamespace(TOPP=topp)
    robot.set_arm_joints = joints
    robot.set_gripper = lambda value, side: setattr(robot, f"{side}_gripper_val", float(value))
    cameras = SimpleNamespace(collect_head_camera=True, collect_wrist_camera=True,
                              get_config=lambda: {name: {} for name in ("head_camera", "left_camera", "right_camera")},
                              update_picture=lambda: None)
    cameras.static_camera_name = ["head_camera"]
    cameras.static_camera_list = [SimpleNamespace(get_picture=lambda name: np.full((12, 14, 4), [1, 0, 0, 1], np.float32))]
    cameras.left_camera = SimpleNamespace(get_picture=lambda name: np.full((8, 10, 4), [0, 1, 0, 1], np.float32))
    cameras.right_camera = SimpleNamespace(get_picture=lambda name: np.full((8, 10, 4), [0, 0, 1, 1], np.float32))
    bind(cameras, "envs/camera/camera.py", ["get_rgba", "get_rgb"])
    env = SimpleNamespace(robot=robot, cameras=cameras, data_type={"rgb": True, "qpos": True},
                          _update_render=lambda: None, check_success=lambda: False,
                          scene=SimpleNamespace(step=lambda: None), take_action_cnt=0, step_lim=10,
                          eval_success=False, eval_video_path=None, render_freq=0)
    bind(env, "envs/_base_task.py", ["get_obs", "take_action"])
    obs = env.get_obs()
    expected_state = np.array([0, 1, 2, 3, 4, 5, .2, 7, 8, 9, 10, 11, 12, .8])
    np.testing.assert_allclose(obs["joint_action"]["vector"], expected_state)
    keys = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
    batch, raw, _ = build_observation(obs, codec, keys, cfg)
    torch.testing.assert_close(batch["observed_rgb"][0, :, 0, 0], torch.tensor([1., -1., -1.]))
    target = np.array([.1, .2, .3, .4, .5, .6, .35, -.1, -.2, -.3, -.4, -.5, -.6, .65], np.float32)
    normalized = codec.encode({key: value[0] for key, value in raw.items()}, {"action": torch.from_numpy(target)[None]})["action"]
    physical = codec.decode(normalized[None], raw)[0, 0].numpy()
    action = prepare_action(physical, codec)
    env.take_action(action, action_type=WSA1_5Agent.action_type)
    assert env.take_action_cnt == 1 and len(paths) == 2
    np.testing.assert_allclose(paths[0][-1], target[:6], atol=1e-6)
    np.testing.assert_allclose(paths[1][-1], target[7:13], atol=1e-6)
    np.testing.assert_allclose(env.get_obs()["joint_action"]["vector"], target, atol=1e-6)


def test_seed_selection_and_success_match_official_historical_runner(tmp_path, monkeypatch, official_legacy_function):
    class Unstable(Exception):
        pass

    class Env:
        step_lim, render_freq, eval_video_path = 2, 0, None

        def __init__(self):
            self.actions, self.resets = [], []

        def setup_demo(self, now_ep_num, seed, **kwargs):
            self.episode, self.seed = now_ep_num, seed
            self.take_action_cnt, self.eval_success = 0, False
            self.plan_success = seed != 100001
            np.random.seed(seed)

        def play_once(self):
            if self.seed == 100000:
                raise Unstable("skip")
            return {"info": {}}

        def check_success(self):
            return True

        def close_env(self, **kwargs):
            pass

        def set_instruction(self, instruction):
            self.instruction = instruction

        def get_instruction(self):
            return self.instruction

        def get_obs(self):
            return observation()

        def take_action(self, action, action_type="qpos"):
            assert action.shape == (14,) and action_type == "qpos"
            self.actions.append((self.seed, self.instruction))
            self.take_action_cnt += 1
            self.eval_success = self.episode == 0

    descriptions = lambda *args: [{"unseen": ["move", "lift"]}]
    task_args = {"task_name": "adjust_bottle", "policy_name": "wsa1_5", "ckpt_setting": "test",
                 "task_config": "demo_clean", "clear_cache_freq": 5, "render_freq": 0, "eval_mode": True}
    official = official_legacy_function("script/eval_policy.py", "eval_policy")
    original_env = Env()
    official.__globals__.update({"UnStableError": Unstable, "generate_episode_descriptions": descriptions,
        "eval_function_decorator": lambda policy, name: (
            (lambda env, model, obs: env.take_action(np.ones(14))) if name == "eval" else
            (lambda model: original_env.resets.append(original_env.seed)))})
    official("adjust_bottle", original_env, task_args.copy(), None, 100000, test_num=2, instruction_type="unseen")
    our_env, videos = Env(), []
    monkeypatch.setattr(inference, "save_replay_video", lambda path, *args: videos.append(path.name))
    args = SimpleNamespace(task_idx=0, num_episodes=2, seed=0, instruction_type="unseen", video_dir=tmp_path, fps=30)
    inference.run_episodes(args, our_env, task_args.copy(), Unstable, descriptions,
                           lambda *args: (np.ones(14), np.zeros((8, 8, 3), np.uint8)), our_env.resets.append, "qpos")
    assert our_env.resets == original_env.resets == [100002, 100003]
    assert our_env.actions == original_env.actions
    assert our_env.suc == original_env.suc == 1 and our_env.test_num == original_env.test_num == 2
    assert videos == ["success_1.mp4", "failure_2.mp4"]


@pytest.mark.parametrize("task_config", ["demo_clean", "demo_randomized"])
def test_official_historical_task_config_keeps_randomization_and_own_video(monkeypatch, official_legacy_function, task_config):
    from omegaconf import OmegaConf

    root = Path(os.environ.get("ROBOTWIN_OFFICIAL_SOURCE", PROJECT / "third_party/RoboTwin"))
    monkeypatch.setattr(inference, "ROBOTWIN_ROOT", root)
    monkeypatch.setattr(inference, "get_embodiment_config", lambda path: {"arm_joints_name": [["joint"] * 6] * 2})
    configs = root / "task_config"
    official = OmegaConf.to_container(OmegaConf.load(configs / f"{task_config}.yml"), resolve=True)
    args = inference.build_task_args(task_config, "adjust_bottle", configs)
    assert args["domain_randomization"] == official["domain_randomization"]
    assert args["camera"] == official["camera"] and args["data_type"] == official["data_type"]
    assert args["eval_mode"] and args["eval_video_save_dir"] is None
    assert Path(args["left_robot_file"]).is_absolute() and args["dual_arm_embodied"]


@pytest.mark.parametrize("stage", ["instruction", "reset", "policy", "video"])
def test_rollout_errors_propagate_and_close_environment(tmp_path, monkeypatch, stage):
    class Env:
        step_lim, render_freq, plan_success = 1, 0, True

        def __init__(self):
            self.closed = []

        def setup_demo(self, **kwargs):
            self.take_action_cnt, self.eval_success = 0, False

        def play_once(self):
            return {"info": {}}

        def close_env(self, **kwargs):
            self.closed.append(kwargs)

        def check_success(self):
            return True

        def set_instruction(self, instruction):
            self.instruction = instruction

        def get_instruction(self):
            return self.instruction

        def get_obs(self):
            return {}

        def take_action(self, *args, **kwargs):
            self.take_action_cnt, self.eval_success = 1, True

    def fail_at(name):
        if stage == name:
            raise RuntimeError(f"failure in {name}")

    def descriptions(*args):
        fail_at("instruction")
        return [{"unseen": ["move"]}]

    def step(*args):
        fail_at("policy")
        return np.ones(14), np.zeros((8, 8, 3), np.uint8)

    monkeypatch.setattr(inference, "save_replay_video", lambda *args: fail_at("video"))
    env = Env()
    args = SimpleNamespace(task_idx=0, num_episodes=1, seed=0, instruction_type="unseen", video_dir=tmp_path, fps=30)
    with pytest.raises(RuntimeError, match=f"failure in {stage}"):
        inference.run_episodes(args, env, {"render_freq": 0, "clear_cache_freq": 5, "task_config": "demo_clean"},
                               ValueError, descriptions, step, lambda seed: fail_at("reset"), "qpos")
    assert len(env.closed) == 2 and env.test_num == 0
    assert not list(tmp_path.glob("*.mp4"))


@pytest.mark.parametrize("unstable", [True, False])
def test_second_setup_failure_does_not_consume_an_evaluation_episode(tmp_path, monkeypatch, unstable):
    class Unstable(Exception):
        pass

    class Env:
        step_lim, render_freq, plan_success = 1, 0, True

        def __init__(self):
            self.setups, self.closed = {}, 0

        def setup_demo(self, seed, **kwargs):
            self.seed = seed
            self.setups[seed] = self.setups.get(seed, 0) + 1
            self.take_action_cnt, self.eval_success = 0, False
            if seed == 100000 and self.setups[seed] == 2:
                raise (Unstable if unstable else RuntimeError)("failed setup")

        def play_once(self):
            return {"info": {}}

        def check_success(self):
            return True

        def close_env(self, **kwargs):
            self.closed += 1

        def set_instruction(self, instruction):
            pass

        def get_instruction(self):
            return "move"

        def get_obs(self):
            return {}

        def take_action(self, *args, **kwargs):
            self.take_action_cnt, self.eval_success = 1, True

    env, resets, videos = Env(), [], []
    monkeypatch.setattr(inference, "save_replay_video", lambda path, *args: videos.append(path.name))
    args = SimpleNamespace(task_idx=0, num_episodes=1, seed=0, instruction_type="unseen", video_dir=tmp_path, fps=30)

    def run():
        inference.run_episodes(args, env, {"render_freq": 0, "clear_cache_freq": 5, "task_config": "demo_clean"},
                               Unstable, lambda *args: [{"unseen": ["move"]}],
                               lambda *args: (np.ones(14), np.zeros((8, 8, 3), np.uint8)), resets.append, "qpos")

    if unstable:
        run()
        assert resets == [100001] and videos == ["success_1.mp4"] and env.test_num == 1
    else:
        with pytest.raises(RuntimeError, match="failed setup"):
            run()
        assert env.closed == 2 and not resets and not videos and env.test_num == 0
