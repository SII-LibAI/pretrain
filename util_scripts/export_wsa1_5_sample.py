"""直接导出 WSA1.5 dataset 的训练样本，不加载模型、VAE 或文本编码器。"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def save_sample(sample, output, source):
    import torch
    from torchvision.utils import make_grid
    from lerobot.policies.wsa1_5.utils.visualization import _save_image

    target = sample["targets"]
    tensors = {"current_rgb": sample["observed_rgb"], "future_rgb": target["future_rgb"],
               "future_motion": target["future_motion"]}
    for name, value in tensors.items():
        if not torch.isfinite(value).all() or (value < -1).any() or (value > 1).any():
            raise ValueError(f"{name} 必须是有限的 [-1,1] 张量")
    output.mkdir(parents=True, exist_ok=True)
    (output / "prompt.txt").write_text(sample["condition_text"], encoding="utf-8")
    _save_image(tensors["current_rgb"], output / "current_rgb.png")
    for name in ("future_rgb", "future_motion"):
        _save_image(make_grid(tensors[name], nrow=4, padding=2, pad_value=-1), output / f"{name}.png")
    masks = {"current_rgb_valid": sample["observed_valid"], "future_rgb_valid": target["rgb_valid"],
             "future_motion_valid": target["motion_valid"], "future_frame_valid": sample["future_frame_valid"]}
    torch.save({"prompt": sample["condition_text"], **{key: value.detach().cpu() for key, value in tensors.items()},
                **{key: value.detach().cpu() for key, value in masks.items()}}, output / "sample.pt")
    metadata = {**source, "sample_id": sample["sample_id"], "episode_index": int(sample["episode_index"]),
                "subtask_id": int(sample["subtask_id"]), "start_timestamp": float(sample["start_timestamp"]),
                "frame_timestamps": sample["frame_timestamps"].tolist(),
                "future_frame_valid": masks["future_frame_valid"].tolist(),
                "future_motion_valid": masks["future_motion_valid"].tolist(),
                "tensor_shapes": {key: list(value.shape) for key, value in tensors.items()},
                "tensor_range": "[-1,1]", "png_mapping": "round((tensor + 1) * 127.5)",
                "future_grid_columns": 4, "future_grid_order": "chronological, row-major"}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="LeRobot V3 数据集或父目录")
    parser.add_argument("--train-config", type=Path, required=True, help="训练保存的 train_config.json，只读取配置")
    parser.add_argument("--output", type=Path, required=True, help="空输出目录")
    parser.add_argument("--episode-index", type=int, help="省略时随机抽取一个 episode")
    parser.add_argument("--frame-index", type=int, help="episode 内帧号；省略时随机抽取")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.seed < 0 or any(x is not None and x < 0 for x in (args.episode_index, args.frame_index)):
        parser.error("seed、episode-index、frame-index 必须非负")
    if args.frame_index is not None and args.episode_index is None:
        parser.error("指定 frame-index 时必须同时指定 episode-index")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError(f"输出目录必须为空：{args.output}")
    if not args.train_config.is_file():
        raise FileNotFoundError(args.train_config)

    import numpy as np
    import torch
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import load_episodes
    from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA1_5DatasetConfig
    from lerobot.policies.wsa1_5.dataset_wsa1_5 import WSA1_5Dataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    training = TrainPipelineConfig.from_pretrained(args.train_config)
    data, policy = training.dataset, training.policy
    if not isinstance(data, WSA1_5DatasetConfig):
        raise ValueError("train_config 必须使用 wsa1_5 dataset")
    root = args.dataset_root.resolve()
    if not (root / "meta/info.json").is_file():
        discovery = Path(__file__).with_name("discover_wsa1_5_datasets.sh")
        repos = sorted(subprocess.check_output(["bash", str(discovery), str(root)], text=True).split())
        root = root / repos[int(rng.integers(len(repos)))]
    episodes = list(load_episodes(root).select_columns(["episode_index"]).with_format(None)["episode_index"])
    if not episodes:
        raise ValueError(f"数据集没有 episode：{root}")
    episode = args.episode_index if args.episode_index is not None else int(rng.choice(episodes))
    if episode not in episodes:
        raise IndexError(f"数据集中不存在 episode {episode}：{root}")
    data.root, data.repo_id, data.episodes = root, root.name, [episode]
    base = LeRobotDataset(root.name, root=root, episodes=[episode], video_backend=data.video_backend)
    if policy.enable_action and not (root / "meta/annotations/action.parquet").is_file():
        from lerobot.datasets.factory import _apply_dataset_stats_overrides

        _apply_dataset_stats_overrides(training, base)
    # Keep the configured training augmentation; teacher and offline embeddings are not needed to inspect pixels/text.
    dataset = WSA1_5Dataset(base, data, include_3d_targets=False, text_embedding_shape=None,
                           action_config=policy if policy.enable_action else None, training=True)
    if args.frame_index is None:
        position = int(rng.integers(len(dataset)))
    else:
        positions = np.flatnonzero(dataset.frame_indices[dataset.sample_indices] == args.frame_index)
        if len(positions) != 1:
            raise IndexError(f"episode {episode} 没有唯一的可用 frame_index={args.frame_index}")
        position = int(positions[0])
    frame = int(dataset.frame_indices[dataset.sample_indices[position]])
    save_sample(dataset[position], args.output, {
        "dataset_root": str(root), "frame_index": frame, "seed": args.seed,
        "train_config": str(args.train_config.resolve()), "view_keys": dataset.view_keys,
        "future_frames": data.future_frames, "frame_stride": data.frame_stride,
        "image_augment": data.image_augment, "object_paste": data.object_paste,
    })
    print(f"已导出 {root} episode={episode} frame={frame} → {args.output}")


if __name__ == "__main__":
    main()
