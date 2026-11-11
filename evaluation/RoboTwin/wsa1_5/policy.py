"""WSA1.5 adapter for RoboTwin's 14D ALOHA joint control interface."""
from collections import deque
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA1_5Config
from lerobot.policies.wsa1_5.dataset_wsa1_5 import ActionCodec, action_prompt, compose_rgb_views
from lerobot.policies.wsa1_5.initialization import load_report, read_checkpoint
from lerobot.policies.wsa1_5.modeling_wsa1_5 import WSA1_5_Policy, construction_dtype
from lerobot.policies.wsa1_5.text_embeddings import TextEmbeddingCache, encode_text, text_embedding_path
from lerobot.policies.wsa1_5.utils.temporal import ObservationHistory


CAMERAS = {"observation.images.cam_high": "head_camera",
           "observation.images.cam_left_wrist": "left_camera",
           "observation.images.cam_right_wrist": "right_camera"}


def load_codec(checkpoint, config, robot_type, stats_path=None):
    path = Path(stats_path) if stats_path else Path(checkpoint) / "stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    if robot_type in stats:
        stats = stats[robot_type]
    features = {key: {"shape": [len(value["mean"])]} for key, value in stats.items() if "mean" in value}
    codec = ActionCodec(SimpleNamespace(robot_type=robot_type, features=features, stats=stats), config)
    if (codec.control_mode != "joint" or codec.state_keys != ["observation.state"]
            or codec.action_keys != ["action"] or not codec.has_action
            or features["observation.state"]["shape"] != [14] or features["action"]["shape"] != [14]):
        raise ValueError("WSA RoboTwin evaluation requires the aloha joint schema and 14D observation.state/action stats")
    return codec


def build_observation(observation, codec, view_keys, config):
    """Match training's RGB canvas, missing-camera mask and state normalization."""
    views, flags = [], []
    for key in view_keys:
        image = observation["observation"].get(CAMERAS[key], {}).get("rgb") if key else None
        if image is None:
            views.append(torch.zeros(1, 3, 16, 16))
            flags.append(False)
            continue
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("RoboTwin camera RGB must be uint8 [H,W,3]")
        views.append(torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float() / 255)
        flags.append(True)
    if not any(flags):
        raise ValueError("RoboTwin observation has no valid RGB camera")
    canvas, valid = compose_rgb_views(views, torch.tensor(flags), config.rgb_height, config.rgb_width)
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    if state.shape != (14,):
        raise ValueError("RoboTwin joint state must be left joints(6)/gripper/right joints(6)/gripper")
    raw_state = {"observation.state": torch.from_numpy(state.copy())}
    encoded = codec.encode(raw_state)
    batch = {"observed_rgb": canvas * 2 - 1, "observed_valid": valid, "state": encoded["state"][None]}
    return batch, {key: value[None] for key, value in raw_state.items()}, canvas[0]


def prepare_action(action, codec):
    """Undo schema padding/reorder after ActionCodec returns absolute physical targets."""
    action = np.asarray(action, dtype=np.float32).copy()
    if action.shape != (codec.config.action_dim,) or not np.isfinite(action).all():
        raise ValueError("WSA action must be a finite vector matching checkpoint action_dim")
    if codec.schema.action_reorder is not None:
        native = np.zeros(14, dtype=np.float32)
        covered = np.zeros(14, dtype=bool)
        for lo, hi, start, end in codec.schema.action_reorder:
            if covered[lo:hi].any():
                raise ValueError("Cannot invert overlapping schema action source slices")
            native[lo:hi] = action[start:end]
            covered[lo:hi] = True
        if not covered.all():
            raise ValueError("Schema action reorder must cover all 14 RoboTwin joint coordinates")
        action = native
    if action.shape != (14,):
        raise ValueError("RoboTwin qpos action must have 14 coordinates after schema inverse reorder")
    action[[6, 13]] = np.clip(action[[6, 13]], 0, 1)
    return action


class WSA1_5Agent:
    action_type = "qpos"

    def __init__(self, policy, codec, *, view_keys=None, infer_horizon=None, text_cache=None,
                 text_encoder_path=None, tokenizer_path=None, text_device="cpu"):
        self.policy, self.codec, self.config = policy, codec, policy.config
        cfg = self.config
        if cfg.action_frame_stride != 1:
            raise ValueError("RoboTwin dense control requires checkpoint action_frame_stride=1")
        self.horizon = infer_horizon or cfg.action_chunk_size
        if not 1 <= self.horizon <= cfg.action_chunk_size:
            raise ValueError("infer_horizon must be within [1, checkpoint action_chunk_size]")
        reverse_images = {target: source for source, target in codec.schema.image_mapping.items()}
        self.view_keys = view_keys if view_keys is not None else [
            reverse_images.get(f"observation.images.image{i}", "") for i in range(3)]
        if len(self.view_keys) != 3 or any(key and key not in CAMERAS for key in self.view_keys):
            raise ValueError("Expected three RoboTwin schema camera keys")
        self.device = torch.device(cfg.device)
        self.text_cache = TextEmbeddingCache(text_cache, cfg.max_text_length, cfg.text_dim) if text_cache else None
        self.text_encoder_path, self.tokenizer_path = text_encoder_path, tokenizer_path
        self.text_device = torch.device(text_device)
        self.text_encoder = self.tokenizer = None
        self._text = self._text_pair = None
        if not cfg.load_text_encoder and self.text_cache is None and not (text_encoder_path and tokenizer_path):
            raise ValueError("Offline-text checkpoint needs --text-cache or both --text-encoder-path and --tokenizer-path for new instructions")
        self.reset(0)

    @classmethod
    def from_checkpoint(cls, args):
        cfg = PreTrainedConfig.from_pretrained(args.ckpt_path)
        if not isinstance(cfg, WSA1_5Config) or not cfg.enable_action:
            raise ValueError("WSA RoboTwin evaluation requires a WSA1.5 Action checkpoint")
        if args.action_mode not in (None, "auto", cfg.action_mode):
            raise ValueError("action_mode must match the saved checkpoint; use auto")
        if args.dtype not in (None, "auto", cfg.dtype):
            raise ValueError("dtype must match the saved checkpoint; use auto")
        if args.action_sampler_steps is not None:
            if type(args.action_sampler_steps) is not int or args.action_sampler_steps <= 0:
                raise ValueError("action_sampler_steps must be a positive integer")
            cfg.action_sampler_steps = args.action_sampler_steps
        cfg.device = args.device
        if args.tokenizer_path:
            cfg.tokenizer_path = str(args.tokenizer_path)
        codec = load_codec(args.ckpt_path, cfg, args.robot_type, args.stats_path)
        view_keys = None
        training_path = args.ckpt_path / "train_config.json"
        if training_path.is_file():
            data_cfg = json.loads(training_path.read_text(encoding="utf-8"))["dataset"]
            names = data_cfg.get("view_names", [f"observation.images.image{i}" for i in range(3)])
            reverse = {target: source for source, target in codec.schema.image_mapping.items()}
            view_keys = data_cfg.get("view_keys") or [reverse.get(name, "") for name in names]
            if data_cfg.get("use_subtask"):
                logging.warning("RoboTwin evaluation uses the original task instruction; no ground-truth subtask/endpoint is supplied")
        policy = WSA1_5_Policy.from_pretrained(args.ckpt_path, config=cfg)
        logging.info("[WSA eval] action_mode=%s dtype=%s history=%sx%s encoding=%s chunk=%s action_sampler_steps=%s",
                     cfg.action_mode, cfg.dtype, cfg.num_history_frame, cfg.history_frame_stride,
                     cfg.sparse_frame_encoding, cfg.action_chunk_size, cfg.action_sampler_steps)
        return cls(policy, codec, view_keys=view_keys, infer_horizon=args.infer_horizon,
                   text_cache=args.text_cache, text_encoder_path=args.text_encoder_path,
                   tokenizer_path=args.tokenizer_path, text_device=args.text_device)

    def reset(self, seed):
        self.policy.reset()
        self.history = ObservationHistory(self.config.num_history_frame, self.config.history_frame_stride)
        self.actions = deque()
        self.seed, self.replans = seed, 0
        self.instruction = None

    def text_condition(self, text):
        if self.config.load_text_encoder:
            return {}
        if text != self._text:
            if self.text_cache is not None and text_embedding_path(self.text_cache.directory, text).is_file():
                pair = self.text_cache.get(text)
                self._text_pair = (pair[0][None].to(self.device), pair[1][None].to(self.device))
            else:
                if not (self.text_encoder_path and self.tokenizer_path):
                    raise FileNotFoundError("Evaluation instruction is absent from --text-cache; provide UMT5/tokenizer paths to encode new instructions")
                if self.text_encoder is None:
                    from lerobot.policies.wsa1_5.vendor.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder

                    logging.info("[WSA eval] loading frozen UMT5 on %s for evaluation instructions", self.text_device)
                    with construction_dtype(getattr(torch, self.config.dtype)):
                        self.text_encoder = WanTextEncoder().eval().requires_grad_(False)
                    load_report(self.text_encoder, read_checkpoint(self.text_encoder_path), name="evaluation UMT5", strict=True)
                    self.text_encoder.to(self.text_device)
                    self.tokenizer = HuggingfaceTokenizer(str(self.tokenizer_path), seq_len=self.config.max_text_length,
                                                         clean="whitespace", local_files_only=True)
                pair = encode_text(self.text_encoder, self.tokenizer, [text], self.text_device,
                                   self.config.max_text_length, self.config.text_dim)
                self._text_pair = tuple(value.to(self.device) for value in pair)
            self._text = text
        return dict(zip(("text_embedding", "text_valid"), self._text_pair, strict=True))

    @torch.no_grad()
    def step(self, observation, instruction):
        if self.instruction is not None and instruction != self.instruction:
            raise ValueError("Instruction changed within an episode; reset the agent before switching tasks")
        self.instruction = instruction
        batch, raw_state, canvas = build_observation(observation, self.codec, self.view_keys, self.config)
        batch = {key: value.to(self.device) for key, value in batch.items()}
        self.history.append(batch["observed_rgb"], batch["observed_valid"])
        if not self.actions:
            text = action_prompt(instruction, self.codec.control_mode, self.config.action_mode,
                                 embodiment=self.codec.embodiment_name if self.config.action_space == "unified72" else None)
            batch["condition_text"] = [text]
            batch.update(self.text_condition(text))
            if self.config.num_history_frame:
                frames, masks = self.history.sample()
                batch.update(history_rgb=frames[:, :-1], history_valid=masks[:, :-1])
            actions = self.policy.predict_action_chunk(
                batch, codec=self.codec, raw_state={key: value.to(self.device) for key, value in raw_state.items()},
                seed=self.seed + self.replans)
            if actions.shape != (1, self.config.action_chunk_size, self.config.action_dim):
                raise ValueError("WSA prediction must be [1,action_chunk_size,checkpoint action_dim]")
            self.actions.extend(prepare_action(value, self.codec) for value in actions[0, :self.horizon].float().cpu().numpy())
            self.replans += 1
        image = (canvas.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
        return self.actions.popleft(), image
