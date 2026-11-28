"""
cd /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series
python util_scripts/infer_wsa1_5_future.py \
    --checkpoint /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series/outputs/wsa1_5_1w5/2026_11_22_08_52_47/checkpoints/010000/pretrained_model \
    --dataset-root /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series/data/ego/Egodex-LerobotV3/extra/basic_pick_place \
    --output /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series/outputs/future/filter_1w/egodex/00 \
    --device cuda \
    --seed 0 \
    --omega-checkpoint /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/vggt-omega/vggt_omega_1b_256_text.pt
"""
"""从 LeRobot V3 抽取当前观测和文本，按 checkpoint 配置预测未来 RGB。"""
import argparse
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
              "view_keys": dataset.view_keys,
              "goal_reference_frame_index": (int(dataset.frame_indices[segment.last]) if segment.metadata_valid else
                                               len(dataset.timestamps) - 1 if data.allow_endframe_3d else None)}
    from lerobot.policies.wsa1_5.teacher import preprocess_omega_views

    reference = (dataset, index, preprocess_omega_views([view[0] for view in views])[None])
    return canvas * 2 - 1, valid, segment.text, source, root / data.text_embedding_dir, reference


def load_action_condition(reference, config):
    """Match Action training conditions using only current state and past RGB."""
    from lerobot.policies.wsa1_5.dataset_wsa1_5 import compose_rgb_views
    from lerobot.policies.wsa1_5.utils.temporal import history_indices

    dataset, index, _ = reference
    codec = dataset.action_codec
    if not codec.has_state:
        raise ValueError("Action checkpoint 的视频推理需要当前 state")
    table = dataset.base.hf_dataset.select_columns(codec.state_keys).with_format("arrow")[index:index + 1]
    inputs = codec.encode({key: table[key][0].as_py() for key in codec.state_keys})
    if config.num_history_frame:
        episode = int(dataset.episode_indices[index])
        first = dataset.meta.episodes[episode]["dataset_from_index"]
        indices, time_valid = history_indices(index - first, config.num_history_frame, config.history_frame_stride)
        timestamps = dataset.timestamps[first + indices[:-1]]
        views, flags = zip(*(dataset._decode(key, timestamps, episode) for key in dataset.view_keys), strict=True)
        rgb, valid = compose_rgb_views(views, flags, config.rgb_height, config.rgb_width)
        inputs.update(history_rgb=rgb * 2 - 1, history_valid=valid & time_valid[:-1, None, None, None])
    return {key: value[None] for key, value in inputs.items()}


def load_future_gt(reference, episode):
    import torch.nn.functional as F
    from lerobot.policies.wsa1_5.dataset_wsa1_5 import LeRobotDataset, compose_rgb_views

    dataset, index, _ = reference
    queries, padding = LeRobotDataset._get_query_indices(dataset, index, episode)
    key = dataset.config.motion_key
    indices = queries[key]
    timestamps = dataset.timestamps[indices]
    views, flags = zip(*(dataset._decode(view, timestamps, episode) for view in dataset.view_keys), strict=True)
    rgb, valid = compose_rgb_views(views, flags, dataset.config.rgb_height, dataset.config.rgb_width)
    time_valid = ~padding[f"{key}_is_pad"]
    valid &= time_valid[:, None, None, None]
    motion, motion_valid = dataset._decode(key, timestamps, episode)
    motion = F.interpolate(motion, size=(dataset.config.motion_height, dataset.config.motion_width),
                           mode="bilinear", align_corners=False)
    motion_valid &= time_valid
    return rgb.masked_fill(~valid, 0), valid, time_valid, motion.masked_fill(~motion_valid[:, None, None, None], 0), motion_valid


def load_goal_gt(reference, source, omega_path, device, dtype):
    import torch
    from lerobot.policies.wsa1_5.teacher import OmegaTeacher, preprocess_omega_views

    frame = source["goal_reference_frame_index"]
    if frame is None:
        return None  # No target annotation: never substitute an arbitrary future frame.
    dataset, _, _ = reference
    views, flags = zip(*(dataset._decode(key, [dataset.timestamps[frame]], source["episode_index"])
                         for key in dataset.view_keys), strict=True)
    valid = torch.tensor([[bool(flag[0]) for flag in flags]])
    if not valid.any():
        return None
    teacher = OmegaTeacher(str(omega_path), dtype=dtype)
    features, mask = teacher.features([[view[0] for view in views]], valid, torch.device(device))
    images = preprocess_omega_views([view[0] for view in views])[None]
    # Reuse the exact same decoder for target and predicted G features.
    decoder = teacher.model.dense_head.float().eval().requires_grad_(False)
    return decoder, {"goal_3d_layer_features": features, "goal_3d_valid": mask}, images


def load_depth_decoder(path, device):
    import torch
    from vggt_omega.models.heads import DenseHead
    from lerobot.policies.wsa1_5.initialization import load_report, read_checkpoint
    from lerobot.policies.wsa1_5.modeling_wsa1_5 import construction_dtype

    state = read_checkpoint(path)
    weights = {key.removeprefix("dense_head."): value for key, value in state.items()
               if key.startswith("dense_head.")}
    del state
    if not weights:
        raise ValueError("Omega checkpoint 不包含 dense_head 权重")
    with construction_dtype(torch.float32):
        decoder = DenseHead(dim_in=2048, patch_size=16)
    load_report(decoder, weights, name="VGGT-Omega DenseHead", strict=True)
    return decoder.to(device=device, dtype=torch.float32).eval().requires_grad_(False)


def decode_goal_depth(decoder, prediction, images, device):
    import torch

    layers = tuple(decoder.intermediate_layer_idx)
    if layers != (4, 11, 17, 23):
        raise ValueError(f"Omega decoder 层索引不匹配：{layers}")
    features = prediction["goal_3d_layer_features"]
    tokens = [None] * (max(layers) + 1)  # DenseHead only consumes its four cached layers.
    for layer in layers:
        value = features[layer]
        if value.shape != (1, 3, 273, 2048) or not torch.isfinite(value).all():
            raise ValueError(f"非法预测 G 特征：layer={layer}")
        tokens[layer] = value.to(device=device, dtype=torch.float32)
    valid = prediction["goal_3d_valid"]
    if valid.dtype != torch.bool or valid.shape != (1, 3, 273):
        raise ValueError("3D token validity 必须是 bool [1,3,273]")
    if not torch.equal(valid, valid[..., :1].expand_as(valid)) or not valid.any():
        raise ValueError("3D token validity 必须按完整视角标记，且至少一个视角有效")
    with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        depth, confidence = decoder(tokens, images=images.to(device=device, dtype=torch.float32), patch_token_start=17)
    if depth.shape != (1, 3, 256, 256, 1) or confidence.shape != (1, 3, 256, 256):
        raise ValueError("Omega decoder 输出尺寸错误")
    if not torch.isfinite(depth).all() or not torch.isfinite(confidence).all() or (depth <= 0).any():
        raise FloatingPointError("Omega decoder 输出非有限或非正深度")
    mask = valid[0, :, 0].cpu()
    return (depth[0, ..., 0].float().cpu().masked_fill(~mask[:, None, None], 0),
            confidence[0].float().cpu().masked_fill(~mask[:, None, None], 0), mask)


def image_pair(gt, predicted, left_label, right_label, size):
    from PIL import Image, ImageDraw

    width, height = size
    pair = Image.new("RGB", (width * 2, height + 24))
    if gt is not None:
        pair.paste(gt, (0, 24))
    if predicted is not None:
        pair.paste(predicted, (width, 24))
    draw = ImageDraw.Draw(pair)
    draw.text((4, 4), left_label, fill="white")
    draw.text((width + 4, 4), right_label, fill="white")
    return pair


def save_depth(output, predicted, target, goal_frame):
    import numpy as np
    from PIL import Image
    from matplotlib import colormaps

    datasets = [predicted] + ([target] if target is not None else [])
    values = np.concatenate([depth[valid].numpy().reshape(-1) for depth, _, valid in datasets])
    low, high = map(float, np.percentile(values, [2, 98]))
    overview = Image.new("RGB", (512, 3 * 280))
    for view in range(3):
        images = []
        for result in (target, predicted):
            if result is None or not result[2][view]:
                images.append(None)
                continue
            depth = result[0][view].numpy()
            scaled = np.clip((depth - low) / (high - low), 0, 1) if high > low else np.zeros_like(depth)
            images.append(Image.fromarray((colormaps["turbo"](scaled)[..., :3] * 255).round().astype(np.uint8)))
        left = f"GT (Omega) view {view}, frame {goal_frame}" if images[0] is not None else f"GT view {view}: unavailable"
        right = f"Predicted G view {view}" if images[1] is not None else f"Predicted view {view}: invalid"
        overview.paste(image_pair(*images, left, right, (256, 256)), (0, view * 280))
    overview.save(output / "goal_3d.png")
    return {"color_min": low, "color_max": high, "gt_available": target is not None,
            "gt_source": "Omega pseudo-ground-truth on the annotated goal image"}


def save_prediction(output, observed, prediction, config, source, gt):
    from PIL import Image
    import torch

    def pixels(value):
        if not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError("Visualization pixels must be finite and in [0,1]")
        return (value.detach().cpu().float() * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()

    future = prediction["future_rgb"]
    expected = (1, config.future_frames, 3, config.rgb_height, config.rgb_width)
    if future is None or tuple(future.shape) != expected:
        raise ValueError(f"Expected future RGB shape {expected}")
    rgb_pred = pixels(future[0])
    Image.fromarray(pixels((observed.cpu().float() + 1) / 2)[0]).save(output / "current.png")
    gt_rgb, rgb_valid, time_valid, gt_motion, motion_valid = gt
    rgb_gt, motion_gt = pixels(gt_rgb), pixels(gt_motion)
    motion = prediction.get("future_motion")
    if config.enable_motion and motion is None:
        raise ValueError("Motion 分支已启用，但模型没有返回 future_motion")
    motion_pred = None
    if motion is not None:
        expected_motion = (1, config.future_frames, 3, config.motion_height, config.motion_width)
        if tuple(motion.shape) != expected_motion:
            raise ValueError(f"Expected future motion shape {expected_motion}")
        motion_pred = pixels(motion[0])
    width = max(config.rgb_width, config.motion_width) * 2
    for index, offset in enumerate(source["future_frame_offsets"]):
        status = "padding - invalid" if not time_valid[index] else (
            "valid" if rgb_valid[index].all() else "partial views")
        rgb_pair = image_pair(Image.fromarray(rgb_gt[index]), Image.fromarray(rgb_pred[index]),
                              f"RGB GT +{offset}: {status}", f"RGB prediction +{offset}",
                              (config.rgb_width, config.rgb_height))
        motion_pair = image_pair(Image.fromarray(motion_gt[index]) if motion_valid[index] else None,
                                 Image.fromarray(motion_pred[index]) if motion_pred is not None else None,
                                 f"Motion GT +{offset}" if motion_valid[index] else f"Motion GT +{offset}: invalid",
                                 f"Motion prediction +{offset}" if motion_pred is not None else "Motion prediction: disabled",
                                 (config.motion_width, config.motion_height))
        combined = Image.new("RGB", (width, rgb_pair.height + motion_pair.height))
        combined.paste(rgb_pair, ((width - rgb_pair.width) // 2, 0))
        combined.paste(motion_pair, ((width - motion_pair.width) // 2, rgb_pair.height))
        combined.save(output / f"future_{index + 1:03d}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True, help="LeRobot V3 数据集或数据集父目录")
    parser.add_argument("--episode-index", type=int, help="原始 episode ID；省略则随机抽取")
    parser.add_argument("--frame-index", type=int, help="episode 内帧索引；省略则随机抽取")
    parser.add_argument("--output", type=Path, required=True, help="空输出目录")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--omega-checkpoint", type=Path, help="256 Text-Alignment Omega 权重；默认使用模型配置中的路径")
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
    if not config.enable_3d_queries:
        raise ValueError("深度解码需要 checkpoint 启用 G/3D queries")
    omega_path = args.omega_checkpoint or config.omega_checkpoint_path
    if not omega_path or not Path(omega_path).is_file():
        parser.error("请用 --omega-checkpoint 提供 VGGT-Omega-1B-256-Text-Alignment 权重")
    config.device = args.device
    observed, valid, prompt, source, default_cache, reference = load_observation(args, config)
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
    if config.enable_action:
        # This video-only command uses RGB's step count for the joint rollout.
        config.action_sampler_steps = config.sampler_steps
    policy = WSA1_5_Policy.from_pretrained(args.checkpoint, config=config).eval()
    print(f"样本：{source['dataset_root']} episode={source['episode_index']} frame={source['frame_index']}", flush=True)
    print(f"预测未来 {config.future_frames} 帧，尺寸 {config.rgb_height}x{config.rgb_width}，"
          f"采样步数 {config.sampler_steps}", flush=True)
    source.update(checkpoint=str(args.checkpoint.resolve()), enable_action=config.enable_action)
    if config.enable_action:
        batch = {"observed_rgb": observed.to(args.device), "observed_valid": valid.to(args.device),
                 "condition_text": [prompt], **condition,
                 **{key: value.to(args.device) for key, value in load_action_condition(reference, config).items()}}
        prediction = policy.predict_action_chunk(batch, seed=args.seed, return_normalized=True,
                                                  return_world=True)
    else:
        prediction = policy.predict_future(observed.to(args.device), [prompt], seed=args.seed,
                                            observed_valid=valid.to(args.device), **condition)
    args.output.mkdir(parents=True, exist_ok=True)
    # Load GT only after generation; it cannot enter the model's conditioning path.
    gt = load_future_gt(reference, source["episode_index"])
    target = load_goal_gt(reference, source, omega_path, args.device, getattr(torch, config.dtype))
    decoder = target[0] if target is not None else load_depth_decoder(omega_path, args.device)
    gt_depth = decode_goal_depth(*target, args.device) if target is not None else None
    pred_depth = decode_goal_depth(decoder, prediction, reference[2], args.device)
    source["depth"] = save_depth(args.output, pred_depth, gt_depth, source["goal_reference_frame_index"])
    source["future_frame_valid"] = gt[2].tolist()
    save_prediction(args.output, observed, prediction, config, source, gt)
    print(f"已保存当前图、逐时刻 RGB/motion 对比及目标 3D 深度对比：{args.output}")
    return {"prompt": prompt, "seed": args.seed, "future_frames": config.future_frames,
            "sampler_steps": config.sampler_steps, **source}


if __name__ == "__main__":
    main()
