"""独立未来帧脚本测试：真实 adapter、合成数据、假模型，不加载真实模型。"""
import json
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from safetensors.torch import save_file

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import lerobot_dataset
from lerobot.policies.wsa1_5.modeling_wsa1_5 import WSA1_5_Policy
from lerobot.policies.wsa1_5.dataset_wsa1_5 import condition_text
from lerobot.policies.wsa1_5.text_embeddings import PROMPT_VERSION, text_embedding_path
from test_wsa1_5_action import action_config, make_action_data
from test_wsa1_5_dataset import make_data
from util_scripts import infer_wsa1_5_future as script


@pytest.mark.parametrize("count", [4, 8])
@pytest.mark.parametrize("selection", ["explicit", "random", "parent"])
def test_dataset_inference_uses_only_current_rgb_and_checkpoint_frame_count(tmp_path, monkeypatch, count, selection):
    root = tmp_path / "dataset"
    dataset, seen = make_data(root, monkeypatch, future_frames=count, frame_stride=4)
    (root / "meta/info.json").write_text("{}", encoding="utf-8")
    cfg = action_config(future_frames=count, enable_action=False, lambda_rgb=1)
    data = dataset.config
    data.rgb_height, data.rgb_width = cfg.rgb_height, cfg.rgb_width
    data.motion_height, data.motion_width = cfg.motion_height, cfg.motion_width
    data.use_subtask = False
    checkpoint = tmp_path / "checkpoint"
    cfg.save_pretrained(checkpoint)
    TrainPipelineConfig(policy=cfg, dataset=data).save_pretrained(checkpoint)
    cache = root / data.text_embedding_dir
    cache.mkdir(parents=True)
    prompt = condition_text({"task_instruction": "knit"})
    save_file({"context": torch.ones(2, cfg.text_dim, dtype=torch.bfloat16),
               "valid": torch.ones(2, dtype=torch.bool)}, text_embedding_path(cache, prompt),
              metadata={"prompt_version": PROMPT_VERSION, "text": prompt,
                        "max_length": str(cfg.max_text_length), "text_dim": str(cfg.text_dim)})

    def load(repo_id, *, root, episodes, video_backend):
        assert episodes == [0]
        return dataset.base

    monkeypatch.setattr(lerobot_dataset, "LeRobotDataset", load)
    if selection == "parent":
        def discover(command, **kwargs):
            assert command[0] == "bash" and command[1].endswith("discover_wsa1_5_datasets.sh")
            assert command[2] == str(tmp_path.resolve())
            return "dataset\n"
        monkeypatch.setattr(script.subprocess, "check_output", discover)

    class Policy:
        def eval(self):
            return self

        def predict_future(self, observed, texts, *, seed, observed_valid, text_embedding, text_valid):
            assert texts == [prompt] and seed == 7
            assert observed.shape == (1, 3, cfg.rgb_height, cfg.rgb_width)
            assert text_valid.any() and text_embedding.shape == (1, cfg.max_text_length, cfg.text_dim)
            assert len(seen) == 1
            torch.testing.assert_close(observed[0, :, 0, 0], torch.full((3,), seen[0][1][0] / 4 * 2 - 1))
            assert not observed_valid[:, :, 64:].any()
            return {"future_rgb": torch.linspace(0, 1, count).reshape(1, count, 1, 1, 1).expand(
                1, count, 3, cfg.rgb_height, cfg.rgb_width)}

    monkeypatch.setattr(WSA1_5_Policy, "from_pretrained", lambda *args, **kwargs: Policy())
    output = tmp_path / "prediction"
    command = ["infer_wsa1_5_future.py", "--checkpoint", str(checkpoint),
               "--dataset-root", str(tmp_path if selection == "parent" else root),
               "--output", str(output), "--device", "cpu", "--seed", "7"]
    if selection != "random":
        command += ["--episode-index", "0", "--frame-index", "8"]
    monkeypatch.setattr(sys, "argv", command)
    script.main()
    assert len(seen) == 1 and seen[0][0].endswith("rgb")
    assert len(list(output.glob("future_*.png"))) == count
    metadata = json.loads((output / "prediction.json").read_text(encoding="utf-8"))
    assert metadata["future_frames"] == count and metadata["prompt"] == prompt
    assert metadata["episode_index"] == 0 and 0 <= metadata["frame_index"] < 40
    if selection != "random":
        assert metadata["frame_index"] == 8
    assert seen[0][1] == pytest.approx([2 + metadata["frame_index"] / 30])
    assert metadata["future_frame_offsets"] == list(range(4, 4 * count + 1, 4))
    with Image.open(output / "future_001.png") as first, Image.open(output / f"future_{count:03d}.png") as last:
        assert not np.array(first).any() and (np.array(last) == 255).all()
    with pytest.raises(FileExistsError):
        script.main()


def test_action_checkpoint_reuses_its_dataset_prompt_without_reading_action_targets(tmp_path, monkeypatch):
    from lerobot.datasets import factory  # Import class definitions before replacing the reader.

    dataset, seen = make_action_data(tmp_path, monkeypatch)
    config, data = dataset.action_config, dataset.config
    data.rgb_height, data.rgb_width = config.rgb_height, config.rgb_width
    data.motion_height, data.motion_width = config.motion_height, config.motion_width
    checkpoint = tmp_path / "checkpoint"
    config.save_pretrained(checkpoint)
    TrainPipelineConfig(policy=config, dataset=data).save_pretrained(checkpoint)
    monkeypatch.setattr(lerobot_dataset, "LeRobotDataset", lambda *args, **kwargs: dataset.base)
    args = SimpleNamespace(checkpoint=checkpoint, dataset_root=tmp_path, seed=7, episode_index=0, frame_index=8)
    observed, valid, prompt, source, cache = script.load_observation(args, config)
    assert prompt == dataset.segments[0].text
    assert "Control Mode: endpose_effector" in prompt and "Action Mode: delta" in prompt
    assert observed.shape == (1, 3, config.rgb_height, config.rgb_width)
    assert source["frame_index"] == 8 and len(seen) == 1
