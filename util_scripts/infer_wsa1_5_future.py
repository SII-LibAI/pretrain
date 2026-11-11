"""从 LeRobot V3 抽取当前观测和文本，按 checkpoint 配置预测未来 RGB。"""
import argparse
import json
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_observation(args, config):
    import numpy as np

    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import load_episodes
    from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA1_5DatasetConfig
    from lerobot.policies.wsa1_5.dataset_wsa1_5 import WSA1_5Dataset, compose_rgb_views

    training_path = args.checkpoint / "train_config.json"
    if not training_path.is_file():
        raise FileNotFoundError(f"需要 checkpoint 的训练数据配置：{training_path}")
    training = TrainPipelineConfig.from_pretrained(training_path)
    data = training.dataset
    if not isinstance(data, WSA1_5DatasetConfig):
        raise ValueError("训练配置必须包含 wsa1_5 dataset adapter")
    for name in ("future_frames", "rgb_height", "rgb_width", "motion_height", "motion_width"):
        if getattr(data, name) != getattr(config, name):
            raise ValueError(f"训练数据配置与模型配置不一致：{name}")
    rng = np.random.default_rng(args.seed)
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
    if config.enable_action and not (root / "meta/annotations/action.parquet").is_file():
        from lerobot.datasets.factory import _apply_dataset_stats_overrides

        _apply_dataset_stats_overrides(training, base)
    dataset = WSA1_5Dataset(base, data, include_3d_targets=False, training=False,
                           action_config=config if config.enable_action else None)
    if dataset.retarget_source is not None:
        raise ValueError("当前脚本不支持需要选择虚拟 embodiment 的 retarget 数据，请使用普通 LeRobot V3 数据集")
    frame = args.frame_index if args.frame_index is not None else int(rng.integers(len(dataset)))
    positions = np.flatnonzero(dataset.frame_indices[dataset.sample_indices] == frame)
    if len(positions) != 1:
        raise IndexError(f"episode {episode} 中不存在唯一的 frame_index={frame}")
    position = int(positions[0])
    index = int(dataset.sample_indices[position])
    segment = dataset.segments[int(dataset.segment_for_sample[position])]
    # Read only current RGB; __getitem__ would also decode future supervision and motion.
    views, flags = zip(*(dataset._decode(key, [dataset.timestamps[index]], episode)
                         for key in dataset.view_keys), strict=True)
    canvas, valid = compose_rgb_views(views, flags, config.rgb_height, config.rgb_width)
    source = {"dataset_root": str(root), "episode_index": episode, "frame_index": frame,
              "start_timestamp": float(dataset.timestamps[index]), "frame_stride": data.frame_stride,
              "future_frame_offsets": [i * data.frame_stride for i in range(1, config.future_frames + 1)],
              "view_keys": dataset.view_keys}
    return canvas * 2 - 1, valid, segment.text, source, root / data.text_embedding_dir


def save_prediction(output, observed, prediction, config, prompt, seed, source):
    from PIL import Image
    import torch

    future = prediction["future_rgb"]
    expected = (1, config.future_frames, 3, config.rgb_height, config.rgb_width)
    if future is None or tuple(future.shape) != expected:
        raise ValueError(f"Expected future RGB shape {expected}")
    if not torch.isfinite(future).all() or (future < 0).any() or (future > 1).any():
        raise ValueError("Predicted RGB must be finite and in [0,1]")
    frames = torch.cat([(observed.cpu().float() + 1) / 2, future[0].cpu().float()])
    pixels = (frames * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    for index, image in enumerate(pixels):
        name = "current.png" if index == 0 else f"future_{index:03d}.png"
        Image.fromarray(image).save(output / name)
    (output / "prediction.json").write_text(json.dumps({
        "prompt": prompt, "seed": seed, "future_frames": config.future_frames,
        "sampler_steps": config.sampler_steps, "rgb_height": config.rgb_height,
        "rgb_width": config.rgb_width, **source,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True, help="LeRobot V3 数据集或数据集父目录")
    parser.add_argument("--episode-index", type=int, help="原始 episode ID；省略则随机抽取")
    parser.add_argument("--frame-index", type=int, help="episode 内帧索引；省略则随机抽取")
    parser.add_argument("--output", type=Path, required=True, help="空输出目录")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    text = parser.add_mutually_exclusive_group()
    text.add_argument("--text-cache", type=Path, help="已有的文本 embedding 缓存目录")
    text.add_argument("--text-encoder-path", type=Path, help="离线文本 checkpoint 的外部 UMT5 权重")
    parser.add_argument("--tokenizer-path", type=Path)
    args = parser.parse_args()
    if args.seed < 0 or any(value is not None and value < 0 for value in (args.episode_index, args.frame_index)):
        parser.error("seed、episode-index、frame-index 必须非负")
    if args.frame_index is not None and args.episode_index is None:
        parser.error("指定 frame-index 时也需要指定 episode-index")
    if bool(args.text_encoder_path) != bool(args.tokenizer_path):
        parser.error("text-encoder-path 和 tokenizer-path 必须同时提供")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError(f"输出目录必须为空：{args.output}")

    import torch

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA1_5Config
    from lerobot.policies.wsa1_5.initialization import load_report, read_checkpoint
    from lerobot.policies.wsa1_5.modeling_wsa1_5 import WSA1_5_Policy, construction_dtype
    from lerobot.policies.wsa1_5.text_embeddings import TextEmbeddingCache, encode_text

    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    if not isinstance(config, WSA1_5Config) or not config.enable_rgb:
        raise ValueError("需要启用 RGB 分支的 WSA1.5 checkpoint")
    config.device = args.device
    observed, valid, prompt, source, default_cache = load_observation(args, config)
    condition = {}
    if args.text_cache or (not config.load_text_encoder and not args.text_encoder_path):
        context, text_valid = TextEmbeddingCache(args.text_cache or default_cache, config.max_text_length,
                                                 config.text_dim).get(prompt)
        condition = {"text_embedding": context[None].to(args.device),
                     "text_valid": text_valid[None].to(args.device)}
    elif args.text_encoder_path:
        from lerobot.policies.wsa1_5.vendor.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder

        with construction_dtype(getattr(torch, config.dtype)):
            encoder = WanTextEncoder().eval().requires_grad_(False)
        load_report(encoder, read_checkpoint(args.text_encoder_path), name="inference UMT5", strict=True)
        tokenizer = HuggingfaceTokenizer(str(args.tokenizer_path), seq_len=config.max_text_length,
                                         clean="whitespace", local_files_only=True)
        context, text_valid = encode_text(encoder, tokenizer, [prompt], "cpu",
                                          config.max_text_length, config.text_dim)
        condition = {"text_embedding": context.to(args.device), "text_valid": text_valid.to(args.device)}
        del encoder, tokenizer
    policy = WSA1_5_Policy.from_pretrained(args.checkpoint, config=config).eval()
    print(f"样本：{source['dataset_root']} episode={source['episode_index']} frame={source['frame_index']}", flush=True)
    print(f"预测未来 {config.future_frames} 帧，尺寸 {config.rgb_height}x{config.rgb_width}，"
          f"采样步数 {config.sampler_steps}", flush=True)
    prediction = policy.predict_future(observed.to(args.device), [prompt], seed=args.seed,
                                        observed_valid=valid.to(args.device), **condition)
    args.output.mkdir(parents=True, exist_ok=True)
    save_prediction(args.output, observed, prediction, config, prompt, args.seed, source)
    print(f"已保存 current.png、{config.future_frames} 张 future_*.png 和 prediction.json：{args.output}")


if __name__ == "__main__":
    main()
