"""Use the actual WSA adapter; no real model or encoder is loaded."""
import json
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import factory, lerobot_dataset
from test_wsa1_5_action import action_config, make_action_data
from test_wsa1_5_dataset import make_data
from util_scripts import export_wsa1_5_sample as script


@pytest.mark.parametrize("kind,frame", [("world", 8), ("missing_motion", 38), ("action", 8)])
def test_export_matches_dataset_tensors_prompt_and_masks(tmp_path, monkeypatch, kind, frame):
    root = tmp_path / "dataset"
    if kind == "action":
        dataset, _ = make_action_data(root, monkeypatch)
        cfg = dataset.action_config
    else:
        dataset, _ = make_data(root, monkeypatch, future_frames=8, frame_stride=4)
        cfg = action_config(future_frames=8, enable_action=False, lambda_rgb=1)
        (root / "meta/info.json").write_text("{}", encoding="utf-8")
    data = dataset.config
    data.rgb_height, data.rgb_width = cfg.rgb_height, cfg.rgb_width
    data.motion_height, data.motion_width = cfg.motion_height, cfg.motion_width
    if kind == "missing_motion":
        del dataset.base.meta.features[data.motion_key]
    expected = dataset[frame]
    config_dir = tmp_path / "config"
    TrainPipelineConfig(policy=cfg, dataset=data).save_pretrained(config_dir)
    monkeypatch.setattr(lerobot_dataset, "LeRobotDataset", lambda *args, **kwargs: dataset.base)
    output = tmp_path / "export"
    monkeypatch.setattr(sys, "argv", ["export_wsa1_5_sample.py", "--dataset-root", str(root),
                                     "--train-config", str(config_dir / "train_config.json"),
                                     "--output", str(output), "--episode-index", "0", "--frame-index", str(frame)])
    script.main()
    saved = torch.load(output / "sample.pt", weights_only=True)
    assert (output / "prompt.txt").read_text(encoding="utf-8") == expected["condition_text"]
    for name, value in (("current_rgb", expected["observed_rgb"]),
                        ("future_rgb", expected["targets"]["future_rgb"]),
                        ("future_motion", expected["targets"]["future_motion"])):
        torch.testing.assert_close(saved[name], value)
        with Image.open(output / f"{name}.png") as image:
            if name == "current_rgb":
                pixels = ((value + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
                np.testing.assert_array_equal(np.array(image), pixels)
            else:
                h, w = value.shape[-2:]
                assert image.size == (4 * (w + 2) + 2, ((len(value) + 3) // 4) * (h + 2) + 2)
    torch.testing.assert_close(saved["future_motion_valid"], expected["targets"]["motion_valid"])
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["frame_index"] == frame and metadata["future_frame_valid"] == expected["future_frame_valid"].tolist()
    if kind == "missing_motion":
        assert (saved["future_motion"] == -1).all() and not saved["future_motion_valid"].any()
    with pytest.raises(FileExistsError):
        script.main()
