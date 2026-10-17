"""World-only policy integrated with InternVLA's existing trainer and checkpoints."""
from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_wsa1_5 import WSA1_5Config
from .experts import FrozenWanVAE, RGBVideoExpert, MotionDiTExpert, WorldFlow, wan_config
from .future_3d import Future3DExpert, geometric_feature_loss
from .initialization import interpolate_wan_state, load_report, read_checkpoint
from .teacher import OmegaTeacher
from .temporal import TemporalLayout
from .text_embeddings import encode_text, validate_text_embedding
from .timing import StageTimer
from .validation import require_bool, require_finite
from .vendor.helpers.state_dict_converters import wan_video_dit_state_dict_converter


@dataclass
class WorldComponents:
    flow: WorldFlow
    future_3d: Future3DExpert | None = None
    vae: FrozenWanVAE | None = None
    text_encoder: nn.Module | None = None
    tokenizer: object | None = None


@contextmanager
def construction_dtype(dtype):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(previous)


class WSA1_5_Policy(PreTrainedPolicy):
    config_class = WSA1_5Config
    name = "WSA_1_5"

    def __init__(self, config, *, components=None, initialize_pretrained=True):
        super().__init__(config)
        self.layout = TemporalLayout(future_frames=config.future_frames)
        self.model_dtype = getattr(torch, config.dtype)
        if components is None:
            components = self._build_components(initialize_pretrained)
        self.flow = components.flow
        if self.flow.layout != self.layout:
            raise ValueError("WorldFlow temporal layout must match policy.future_frames")
        self.future_3d = components.future_3d if config.enable_3d_queries else None
        if config.enable_3d_queries and self.future_3d is None:
            raise ValueError("enable_3d_queries requires a Future3DExpert")
        self.vae = components.vae
        self.text_encoder = components.text_encoder
        self.tokenizer = components.tokenizer
        self.teacher = OmegaTeacher(config.omega_checkpoint_path, config.teacher_layers,
                                    dtype=self.model_dtype) if config.use_3d_supervision else None
        self._motion_placeholder_latent = None
        self._motion_placeholder_key = None
        self._freeze_modules()

    def _apply(self, fn, recurse=True):
        self._motion_placeholder_latent = None
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self._motion_placeholder_latent = None
        return super()._load_from_state_dict(*args, **kwargs)

    def prepare_for_training(self, device, *, log_parameters=True):
        """Load the enabled teacher on each rank and summarize main modules once."""
        if self.teacher is not None:
            self.teacher._load(device)
        if not log_parameters:
            return
        modules = {"RGB/Observation": self.flow.rgb, "Motion": self.flow.motion,
                   "Future3D": self.future_3d, "Wan VAE": self.vae, "UMT5": self.text_encoder,
                   "VGGT-Omega": self.teacher.model if self.teacher is not None else None}
        for name, module in modules.items():
            if module is None:
                logging.info("[WSA parameters] %s disabled", name)
                continue
            logging.info("[WSA parameters] %s total=%.3fB trainable=%.3fB", name,
                         sum(p.numel() for p in module.parameters()) / 1e9,
                         sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e9)

    def _build_components(self, initialize):
        cfg = self.config
        if initialize:
            paths = ["wan_model_path", "vae_path"]
            if cfg.load_text_encoder:
                paths += ["text_encoder_path", "tokenizer_path"]
            for name in paths:
                path = getattr(cfg, name)
                if not path or not Path(path).exists():
                    raise FileNotFoundError(f"Provide local server asset {name}: {path!r}")
        from .vendor.wan_video_vae import WanVideoVAE38

        common = dict(layers=cfg.num_layers, heads=cfg.num_heads, head_dim=cfg.head_dim,
                      channels=cfg.latent_channels, text_dim=cfg.text_dim, freq_dim=cfg.freq_dim)
        weight_dtype = torch.float32 if cfg.trainable_weights_fp32 else self.model_dtype
        with construction_dtype(weight_dtype):
            rgb = RGBVideoExpert(**wan_config(cfg.rgb_hidden_dim, cfg.rgb_ffn_dim, **common))
            motion = MotionDiTExpert(**wan_config(cfg.motion_hidden_dim, cfg.motion_ffn_dim, **common)) if cfg.enable_motion else None
            student = Future3DExpert(cfg.goal_hidden_dim, cfg.goal_ffn_dim, cfg.num_layers, cfg.num_heads,
                                     cfg.head_dim, cfg.freq_dim, teacher_layers=cfg.teacher_layers,
                                     student_layers=cfg.student_layers) if cfg.enable_3d_queries else None
        text_encoder, tokenizer = None, None
        if cfg.load_text_encoder:
            from .vendor.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer

            with construction_dtype(self.model_dtype):
                text_encoder = WanTextEncoder().eval().requires_grad_(False)
            tokenizer = HuggingfaceTokenizer(cfg.tokenizer_path, seq_len=cfg.max_text_length,
                                            clean="whitespace", local_files_only=True)
        # Keep official mean/std and inverse-std calculation in float32;
        # converting the learned weights must not quantize these constants early.
        vae = WanVideoVAE38().to(dtype=self.model_dtype)
        if initialize:
            state = wan_video_dit_state_dict_converter(read_checkpoint(cfg.wan_model_path))
            load_report(rgb, state, name="RGB Wan DiT", strict=True)
            if motion is not None and cfg.motion_initialization == "wan_width_interpolation":
                converted, _ = interpolate_wan_state(state, motion.state_dict())
                load_report(motion, converted, name="Motion Wan DiT", strict=True)
            if student is not None and cfg.goal_initialization == "wan_width_interpolation":
                # Slot/noise buffers are deterministic; only Wan-compatible
                # backbone keys are transplanted, leaving new feature heads random.
                backbone = {key: value for key, value in student.state_dict().items()
                            if key.startswith(("blocks.", "time_embedding.", "time_projection."))}
                converted, _ = interpolate_wan_state(state, backbone, apply_alpha_scaling=True)
                expected_missing = set(student.state_dict()) - set(backbone)
                report = load_report(student, converted, name="Future3D Wan backbone", strict=False)
                if set(report.missing_keys) != expected_missing or report.unexpected_keys:
                    raise RuntimeError("3D backbone initialization is incomplete beyond the explicit new heads")
            vae_state = read_checkpoint(cfg.vae_path)
            if not all(key.startswith("model.") for key in vae_state):
                vae_state = {f"model.{key}": value for key, value in vae_state.items()}
            load_report(vae, vae_state, name="Wan2.2 VAE", strict=True)
            if text_encoder is not None:
                load_report(text_encoder, read_checkpoint(cfg.text_encoder_path), name="Wan UMT5", strict=True)
        flow = WorldFlow(rgb, motion, backend=cfg.attention_backend, gradient_checkpointing=cfg.gradient_checkpointing,
                         shift=cfg.flow_shift, future_frames=cfg.future_frames)
        return WorldComponents(flow, student, FrozenWanVAE(vae), text_encoder, tokenizer)

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        # The saved policy contains all enabled module weights. Never initialize them
        # from source checkpoints before loading the complete saved policy.
        kwargs["initialize_pretrained"] = False
        if kwargs.get("strict") is False:
            raise ValueError("WSA checkpoint loading requires strict=True; random missing weights are forbidden")
        kwargs["strict"] = True
        return super().from_pretrained(pretrained_name_or_path, **kwargs)

    def _freeze_modules(self):
        cfg = self.config
        if cfg.freeze_rgb:
            self.flow.rgb.requires_grad_(False)
        if self.flow.motion is not None and cfg.freeze_motion:
            self.flow.motion.requires_grad_(False)
        if self.future_3d is not None:
            # R/M cannot read G, so without 3D loss these parameters have no
            # gradient path. Freeze them to avoid unused trainable DDP parameters.
            if cfg.freeze_3d or not cfg.use_3d_supervision:
                self.future_3d.requires_grad_(False)
            if cfg.freeze_heads:
                self.future_3d.feature_heads.requires_grad_(False)
        if self.vae is not None:
            self.vae.eval().requires_grad_(False)
        if self.text_encoder is not None:
            self.text_encoder.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.vae is not None:
            self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    def get_optim_params(self):
        groups = []
        cfg = self.config
        for name, module, lr in [("rgb", self.flow.rgb, cfg.rgb_lr), ("motion", self.flow.motion, cfg.motion_lr),
                                 ("3d", self.future_3d, cfg.goal_lr)]:
            if module is None:
                continue
            params = [p for key, p in module.named_parameters() if p.requires_grad and not key.startswith("feature_heads.")]
            if params:
                groups.append({"name": name, "params": params, "lr": lr})
        if self.future_3d is not None:
            params = [p for p in self.future_3d.feature_heads.parameters() if p.requires_grad]
            if params:
                groups.append({"name": "heads", "params": params, "lr": cfg.heads_lr})
        if not groups:
            raise ValueError("All WSA trainable modules are frozen")
        return groups

    def reset(self):
        pass

    def select_action(self, *args, **kwargs):
        raise NotImplementedError("WSA1.5 predicts world representations; call predict_future")

    def predict_action_chunk(self, *args, **kwargs):
        raise NotImplementedError("WSA1.5 has no action expert")

    @torch.no_grad()
    def _encode_text(self, texts):
        if self.text_encoder is None or self.tokenizer is None:
            raise RuntimeError("load_text_encoder=false requires offline text_embedding and text_valid; no online fallback")
        texts = [texts] if isinstance(texts, str) else list(texts)
        device = next(self.flow.parameters()).device
        context, valid = encode_text(self.text_encoder, self.tokenizer, texts, device,
                                     self.config.max_text_length, self.config.text_dim)
        context = context.to(self.model_dtype)
        require_finite(context, "UMT5 embedding after compute-dtype conversion")
        return context, valid

    def _encode_video(self, video):
        require_finite(video, "VAE input")
        if self.vae is None:
            raise RuntimeError("Pixel-space execution requires the actual frozen Wan2.2 VAE")
        latent = self.vae.encode(video.to(self.model_dtype)).to(self.model_dtype)
        self._validate_encoded_latent(latent, video)
        return latent

    def _validate_encoded_latent(self, latent, video):
        expected = (video.shape[0], self.config.latent_channels, (video.shape[2] - 1) // 4 + 1,
                    video.shape[3] // 16, video.shape[4] // 16)
        if tuple(latent.shape) != expected:
            raise ValueError(f"VAE output shape {tuple(latent.shape)} does not match {expected}")
        require_finite(latent, "Frozen VAE latent")

    def encode_observation(self, observed, text, observed_valid=None, *, latent=None,
                           text_embedding=None, text_valid=None):
        if observed.ndim != 4 or observed.shape[1:] != (3, self.config.rgb_height, self.config.rgb_width):
            raise ValueError(f"Observed RGB must be [B,3,{self.config.rgb_height},{self.config.rgb_width}] "
                             "in Wan-normalized [-1,1] coordinates")
        if not torch.isfinite(observed).all():
            raise ValueError("Non-finite observed RGB")
        pixel_valid = (torch.ones_like(observed[:, :1], dtype=torch.bool) if observed_valid is None else observed_valid)
        if pixel_valid.dtype != torch.bool:
            raise TypeError("Observed pixel validity must be boolean; implicit coercion is forbidden")
        if pixel_valid.shape != observed[:, :1].shape:
            raise ValueError("Observed pixel mask shape mismatch")
        if not pixel_valid.flatten(1).any(1).all():
            raise ValueError("Every sample must contain at least one valid observed RGB view")
        if latent is None:
            latent = self._encode_video(observed[:, :, None])
        else:
            self._validate_encoded_latent(latent, observed[:, :, None])
        if text_embedding is None and text_valid is None:
            context, context_valid = self._encode_text(text)
        else:
            if text_embedding is None or text_valid is None:
                raise ValueError("Provide both text_embedding and text_valid")
            validate_text_embedding(text_embedding, text_valid, self.config.max_text_length, self.config.text_dim)
            context = text_embedding.to(device=latent.device, dtype=self.model_dtype)
            context_valid = text_valid.to(latent.device)
            require_finite(context, "Text embedding after compute-dtype conversion")
            context = context.masked_fill(~context_valid[..., None], 0)
        if context.shape[0] != latent.shape[0]:
            raise ValueError("Text batch must match observed RGB batch")
        spatial = F.interpolate(pixel_valid.float(), size=latent.shape[-2:], mode="area") >= 1 - 1e-6
        return latent, context, context_valid, spatial[:, :, None], pixel_valid

    def _placeholder(self, batch_size, device):
        key = (id(self.vae), torch.device(device), self.model_dtype, self.config.motion_height, self.config.motion_width)
        if self._motion_placeholder_latent is None or self._motion_placeholder_key != key:
            # Keep a normal detached tensor, even when first called under inference_mode.
            with torch.inference_mode(False), torch.no_grad():
                zero = torch.zeros(1, 3, 1, self.config.motion_height, self.config.motion_width, device=device)
                self._motion_placeholder_latent = self._encode_video(zero).detach()
            self._motion_placeholder_key = key
        return self._motion_placeholder_latent.expand(batch_size, -1, -1, -1, -1)

    def _shapes(self, batch_size):
        cfg = self.config
        rgb = (batch_size, cfg.latent_channels, self.layout.latent_frames,
               cfg.rgb_height // 16, cfg.rgb_width // 16) if cfg.enable_rgb else None
        motion = (batch_size, cfg.latent_channels, self.layout.latent_frames,
                  cfg.motion_height // 16, cfg.motion_width // 16) if cfg.enable_motion else None
        return rgb, motion

    def sample_generated_latents(self, observed_latent, context, context_valid, observed_valid, *, seed, steps,
                                 pixel_valid=None, return_features=False):
        rgb_shape, motion_shape = self._shapes(observed_latent.shape[0])
        placeholder = self._placeholder(observed_latent.shape[0], observed_latent.device) if motion_shape else None
        # Returning features is an API choice, not an expert ablation switch.
        future_3d = self.future_3d
        if future_3d is not None and pixel_valid is None:
            raise ValueError("Joint 3D sampling requires observed pixel validity")
        view_valid = self.observed_view_validity(pixel_valid) if future_3d is not None else None
        return self.flow.sample_latents(observed_latent, context, context_valid, rgb_shape=rgb_shape,
                                       motion_shape=motion_shape, motion_placeholder=placeholder,
                                       seed=seed, steps=steps, observed_valid=observed_valid,
                                       future_3d=future_3d, goal_view_valid=view_valid, return_features=return_features)

    @staticmethod
    def observed_view_validity(pixel_valid):
        h, w = pixel_valid.shape[-2:]
        return torch.stack([pixel_valid[:, :, :h * 2 // 3].flatten(1).any(1),
                            pixel_valid[:, :, h * 2 // 3:, :w // 2].flatten(1).any(1),
                            pixel_valid[:, :, h * 2 // 3:, w // 2:].flatten(1).any(1)], 1)

    def prepare_training_targets(self, batch):
        target = batch["targets"]
        observed = batch["observed_rgb"]
        clean, masks = {}, {}
        if self.config.enable_rgb:
            video = self.layout.assemble_rgb(observed, target["future_rgb"])
            clean["rgb"] = self._encode_video(video)
            self.layout.check_latents(clean["rgb"], self.config.rgb_height, self.config.rgb_width)
            future_valid = self.layout.latent_validity(target["rgb_valid"], clean["rgb"].shape[-2:])
            masks["rgb"] = torch.cat([torch.ones_like(future_valid[:, :, :1]), future_valid], 2)
        if self.config.enable_motion:
            video = self.layout.prepend_motion_placeholder(target["future_motion"])
            clean["motion"] = self._encode_video(video)
            self.layout.check_latents(clean["motion"], self.config.motion_height, self.config.motion_width)
            future_valid = self.layout.latent_validity(target["motion_valid"], clean["motion"].shape[-2:])
            masks["motion"] = torch.cat([torch.zeros_like(future_valid[:, :, :1]), future_valid], 2)
        teacher, teacher_valid = {}, None
        if self.config.use_3d_supervision:
            teacher, teacher_valid = self.teacher.features(target["goal_rgb_views"], target["goal_view_valid"], observed.device)
        return clean, masks, teacher, teacher_valid

    def compute_context(self):
        device = next(self.flow.parameters()).device
        return torch.autocast(device_type=device.type, dtype=self.model_dtype,
                              enabled=self.model_dtype != torch.float32 and device.type in {"cpu", "cuda"})

    def training_forward(self, batch):
        with self.compute_context():
            return self._training_forward(batch)

    def _require_supervision(self, masks, teacher_valid):
        cfg = self.config
        weights = {"rgb": cfg.lambda_rgb, "motion": cfg.lambda_motion, "3d": cfg.lambda_3d}
        active = any(weights[name] > 0 and flags[:, :, 1:].any().item() for name, flags in masks.items())
        active = active or (cfg.use_3d_supervision
                            and cfg.feature_mse_weight + cfg.feature_cosine_weight > 0
                            and teacher_valid is not None and teacher_valid.any().item())
        if not active:
            # shortcut: 无监督 batch 直接终止；需连续训练无标签区间时再加入监督感知采样。
            raise ValueError("Batch has no valid positive-weight supervision; refusing a zero-loss optimizer update")

    def _training_forward(self, batch):
        if "targets" not in batch:
            raise ValueError("training_forward requires an explicit targets dictionary")
        if "observed_valid" not in batch:
            raise ValueError("Training requires explicit observed_valid; missing views cannot default to valid")
        timer = StageTimer(batch["observed_rgb"].device)
        require_bool(batch["observed_valid"], "Observed validity")
        target = batch["targets"]
        for name, enabled in [("rgb", self.config.enable_rgb), ("motion", self.config.enable_motion)]:
            if enabled:
                require_bool(target[f"{name}_valid"], f"{name} target validity")
                require_finite(target[f"future_{name}"], f"Future {name} pixels")
        if self.config.use_3d_supervision:
            require_bool(target["goal_view_valid"], "Goal view validity")
        clean, masks, teacher_targets, teacher_valid = self.prepare_training_targets(batch)
        observed_latent = clean["rgb"][:, :, :1] if "rgb" in clean else None
        observed, context, context_valid, observed_valid, pixel_valid = self.encode_observation(
            batch["observed_rgb"], batch["condition_text"], batch["observed_valid"], latent=observed_latent,
            text_embedding=batch.get("text_embedding"), text_valid=batch.get("text_valid"))
        if teacher_valid is not None:
            require_bool(teacher_valid, "Teacher feature validity")
            # An absent observed camera has no corresponding active G queries.
            teacher_valid = teacher_valid & self.observed_view_validity(pixel_valid).repeat_interleave(273, 1).reshape(-1, 3, 273)
        weights = {"rgb": self.config.lambda_rgb, "motion": self.config.lambda_motion, "3d": self.config.lambda_3d}
        self._require_supervision(masks, teacher_valid)
        states, velocities, timesteps = {}, {}, {}
        for name, value in clean.items():
            condition = observed if name == "rgb" else self._placeholder(observed.shape[0], observed.device)
            states[name], velocities[name], timesteps[name] = self.flow.training_state(value, condition)
        prediction = self.flow.velocity(observed, context, context_valid, states.get("rgb"), states.get("motion"),
                                        timesteps.get("rgb"), timesteps.get("motion"), observed_valid=observed_valid,
                                        rgb_valid=masks.get("rgb"), motion_valid=masks.get("motion"),
                                        future_3d=self.future_3d,
                                        goal_view_valid=self.observed_view_validity(pixel_valid), timer=timer)
        losses, metrics = {}, {}
        for name in clean:
            losses[name] = self.flow.flow_loss(prediction[name], velocities[name], timesteps[name], masks[name])
            metrics[f"loss_{name}_flow"] = losses[name].detach().item()
        if self.config.use_3d_supervision:
            losses["3d"], feature_metrics = geometric_feature_loss(prediction["3d"], teacher_targets, teacher_valid,
                                                                  mse_weight=self.config.feature_mse_weight,
                                                                  cosine_weight=self.config.feature_cosine_weight)
            metrics.update(feature_metrics)
            metrics["loss_3d_feature"] = losses["3d"].detach().item()
        loss = sum(weights[name] * value for name, value in losses.items())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite WSA loss: {metrics}")
        metrics["loss"] = loss.detach().item()
        metrics.update(timer.values())
        return loss, metrics

    def forward(self, batch):
        return self.training_forward(batch)

    @torch.no_grad()
    def predict_future(self, observed_rgb, text, seed=0, *, observed_valid=None, text_embedding=None, text_valid=None):
        was_training = self.training
        self.eval()
        try:
            with self.compute_context():
                return self._predict_future(observed_rgb, text, seed, observed_valid, text_embedding, text_valid)
        finally:
            self.train(was_training)

    def _predict_future(self, observed_rgb, text, seed, observed_valid, text_embedding, text_valid):
        observed, context, context_valid, latent_valid, pixel_valid = self.encode_observation(
            observed_rgb, text, observed_valid, text_embedding=text_embedding, text_valid=text_valid)
        generated_rgb, generated_motion, features = self.sample_generated_latents(
            observed, context, context_valid, latent_valid, seed=seed, steps=self.config.sampler_steps,
            pixel_valid=pixel_valid, return_features=True)
        goal_valid = (self.observed_view_validity(pixel_valid) & (self.future_3d is not None))
        goal_valid = goal_valid.repeat_interleave(273, dim=1).reshape(-1, 3, 273)
        features = {key: value.masked_fill(~goal_valid[..., None], 0) for key, value in features.items()}
        result = {"goal_3d_features": features[self.config.teacher_layers[-1]] if features else None,
                  "goal_3d_layer_features": features, "goal_3d_valid": goal_valid,
                  "future_rgb_valid": (pixel_valid & (generated_rgb is not None))[:, None].expand(
                      -1, self.layout.future_frames, -1, -1, -1)}
        for name, latent in [("rgb", generated_rgb), ("motion", generated_motion)]:
            if latent is not None:
                decoded = self.layout.future_from_decoded(self.vae.decode(latent))
                if not torch.isfinite(decoded).all():
                    raise FloatingPointError(f"Non-finite decoded future {name}")
                result[f"future_{name}"] = (decoded.clamp(-1, 1) + 1) / 2
                if name == "rgb":
                    result["future_rgb"] = result["future_rgb"].masked_fill(~result["future_rgb_valid"], 0)
            else:
                result[f"future_{name}"] = None
        return result
