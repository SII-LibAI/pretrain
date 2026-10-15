"""World-only policy integrated with InternVLA's existing trainer and checkpoints."""
from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from lerobot.datasets.wsa1_5 import imagenet_denormalize
from lerobot.policies.pretrained import PreTrainedPolicy

from .cache import ArtifactCache, fingerprint_path
from .configuration_wsa1_5 import WSA1_5Config
from .experts import FrozenWanVAE, RGBVideoExpert, MotionDiTExpert, WorldFlow, wan_config
from .future_3d import Future3DExpert, geometric_feature_loss
from .initialization import interpolate_wan_state, load_report, read_checkpoint
from .teacher import OMEGA_PREPROCESS_VERSION, OmegaTeacher
from .temporal import TemporalLayout
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
        self.layout = TemporalLayout()
        self.model_dtype = getattr(torch, config.dtype)
        if components is None:
            if initialize_pretrained:
                config.frozen_asset_fingerprints.clear()
            components = self._build_components(initialize_pretrained)
        self.flow = components.flow
        self.future_3d = components.future_3d
        self.vae = components.vae
        self.text_encoder = components.text_encoder
        self.tokenizer = components.tokenizer
        self.teacher = OmegaTeacher(config.omega_checkpoint_path, config.teacher_layers,
                                    dtype=self.model_dtype) if config.enable_3d else None
        self.cache = ArtifactCache(config.cache_dir) if config.cache_dir else None
        self._fingerprints = {}
        self._freeze_modules()
        for name, module in self.named_children():
            logging.info("[WSA policy parameters] %s total=%d trainable=%d", name,
                         sum(p.numel() for p in module.parameters()), sum(p.numel() for p in module.parameters() if p.requires_grad))

    def _build_components(self, initialize):
        cfg = self.config
        if initialize:
            for name in ["wan_model_path", "vae_path", "text_encoder_path", "tokenizer_path"]:
                path = getattr(cfg, name)
                if not path or not Path(path).exists():
                    raise FileNotFoundError(f"Provide local server asset {name}: {path!r}")
        from .vendor.wan_video_vae import WanVideoVAE38
        from .vendor.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer

        common = dict(layers=cfg.num_layers, heads=cfg.num_heads, head_dim=cfg.head_dim,
                      channels=cfg.latent_channels, text_dim=cfg.text_dim, freq_dim=cfg.freq_dim)
        weight_dtype = torch.float32 if cfg.trainable_weights_fp32 else self.model_dtype
        with construction_dtype(weight_dtype):
            rgb = RGBVideoExpert(**wan_config(cfg.rgb_hidden_dim, cfg.rgb_ffn_dim, **common))
            motion = MotionDiTExpert(**wan_config(cfg.motion_hidden_dim, cfg.motion_ffn_dim, **common)) if cfg.enable_motion else None
            student = Future3DExpert(cfg.goal_hidden_dim, cfg.goal_ffn_dim, cfg.num_layers, cfg.num_heads,
                                     cfg.head_dim, cfg.freq_dim, teacher_layers=cfg.teacher_layers,
                                     student_layers=cfg.student_layers) if cfg.enable_3d else None
        with construction_dtype(self.model_dtype):
            text_encoder = WanTextEncoder()
        # Keep official mean/std and inverse-std calculation in float32;
        # converting the learned weights must not quantize these constants early.
        vae = WanVideoVAE38().to(dtype=self.model_dtype)
        if initialize:
            state = wan_video_dit_state_dict_converter(read_checkpoint(cfg.wan_model_path))
            load_report(rgb, state, name="RGB Wan DiT", strict=True)
            if motion is not None and cfg.motion_initialization == "wan_width_interpolation":
                converted, rules = interpolate_wan_state(state, motion.state_dict())
                logging.info("[WSA motion interpolation] rules=%s", rules)
                load_report(motion, converted, name="Motion Wan DiT", strict=True)
            if student is not None and cfg.goal_initialization == "wan_width_interpolation":
                # Geometric queries and projection heads are intentionally random;
                # only the known Wan-compatible backbone keys are transplanted.
                backbone = {key: value for key, value in student.state_dict().items()
                            if key.startswith(("blocks.", "time_embedding.", "time_projection."))}
                converted, rules = interpolate_wan_state(state, backbone)
                logging.info("[WSA 3D interpolation] rules=%s", rules)
                expected_missing = set(student.state_dict()) - set(backbone)
                report = load_report(student, converted, name="Future3D Wan backbone", strict=False)
                if set(report.missing_keys) != expected_missing or report.unexpected_keys:
                    raise RuntimeError("3D backbone initialization is incomplete beyond the explicit new query/heads")
            vae_state = read_checkpoint(cfg.vae_path)
            if not all(key.startswith("model.") for key in vae_state):
                vae_state = {f"model.{key}": value for key, value in vae_state.items()}
            load_report(vae, vae_state, name="Wan2.2 VAE", strict=True)
            load_report(text_encoder, read_checkpoint(cfg.text_encoder_path), name="Wan UMT5", strict=True)
        tokenizer = HuggingfaceTokenizer(cfg.tokenizer_path, seq_len=cfg.max_text_length,
                                        clean="whitespace", local_files_only=True)
        flow = WorldFlow(rgb, motion, backend=cfg.attention_backend, gradient_checkpointing=cfg.gradient_checkpointing,
                         shift=cfg.flow_shift)
        return WorldComponents(flow, student, FrozenWanVAE(vae), text_encoder.eval().requires_grad_(False), tokenizer)

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        # The saved policy contains Wan/VAE/text weights. Never initialize them
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
            if cfg.freeze_3d:
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

    def _asset_fingerprint(self, field):
        if field not in self._fingerprints:
            if field in {"vae_path", "text_encoder_path"} and field in self.config.frozen_asset_fingerprints:
                self._fingerprints[field] = self.config.frozen_asset_fingerprints[field]
            else:
                self._fingerprints[field] = fingerprint_path(getattr(self.config, field))
                if field in {"vae_path", "text_encoder_path"}:
                    self.config.frozen_asset_fingerprints[field] = self._fingerprints[field]
        return self._fingerprints[field]

    def _cached(self, kind, identity, compute, asset_fields):
        if self.cache is None:
            return compute()
        identity = {**identity, "dtype": self.config.dtype,
                    "weights": {field: self._asset_fingerprint(field) for field in asset_fields}}
        device = next(self.flow.parameters()).device
        return {key: value.to(device) for key, value in self.cache.get_or_compute(kind, identity, compute).items()}

    @torch.no_grad()
    def _encode_text(self, texts):
        if self.text_encoder is None or self.tokenizer is None:
            raise RuntimeError("Raw-text inference/training requires the actual frozen Wan UMT5 and tokenizer")
        texts = [texts] if isinstance(texts, str) else list(texts)
        device = next(self.flow.parameters()).device
        contexts, masks = [], []
        for text in texts:
            def encode(text=text):
                ids, mask = self.tokenizer([text], return_mask=True, add_special_tokens=True)
                if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
                    raise ValueError("Tokenizer attention mask must contain only finite binary values")
                ids, mask = ids.to(device), mask.to(device).bool()
                context = self.text_encoder(ids, mask)
                context = context.masked_fill(~mask[..., None], 0)
                if not torch.isfinite(context).all():
                    raise FloatingPointError("Non-finite frozen UMT5 embeddings")
                return {"context": context[0], "valid": mask[0]}
            value = self._cached("text", {"text": text, "max_length": self.config.max_text_length}, encode,
                                 ["text_encoder_path", "tokenizer_path"])
            if (value["context"].shape != (self.config.max_text_length, self.config.text_dim)
                    or value["valid"].shape != (self.config.max_text_length,)
                    or value["valid"].dtype != torch.bool):
                raise ValueError("Frozen text artifact shape/mask differs from the configured UMT5 contract")
            contexts.append(value["context"])
            masks.append(value["valid"])
        context = torch.stack(contexts).to(self.model_dtype)
        if not torch.isfinite(context).all():
            raise FloatingPointError("Non-finite UMT5 artifact after compute-dtype conversion")
        return context, torch.stack(masks)

    def _encode_video(self, kind, video, metadata):
        from .validation import require_finite

        require_finite(video, f"{kind} VAE input")
        if self.vae is None:
            raise RuntimeError("Pixel-space execution requires the actual frozen Wan2.2 VAE")
        if self.cache is None:
            latent = self.vae.encode(video.to(self.model_dtype)).to(self.model_dtype)
            self._validate_encoded_latent(latent, video)
            return latent
        # Cache by sample, not by shuffled batch composition. Offline artifacts
        # remain reusable with different worker counts, batch sizes and mixing.
        latents = []
        for index in range(video.shape[0]):
            item = video[index:index + 1]
            info = self._sample_metadata(metadata, index, video.shape[0])
            values = self._cached(kind, {"video": item, "metadata": info},
                                  lambda item=item: {"latent": self.vae.encode(item.to(self.model_dtype))}, ["vae_path"])
            latents.append(values["latent"])
        latent = torch.cat(latents).to(self.model_dtype)
        self._validate_encoded_latent(latent, video)
        return latent

    def _validate_encoded_latent(self, latent, video):
        from .validation import require_finite

        expected = (video.shape[0], self.config.latent_channels, (video.shape[2] - 1) // 4 + 1,
                    video.shape[3] // 16, video.shape[4] // 16)
        if tuple(latent.shape) != expected:
            raise ValueError(f"VAE output shape {tuple(latent.shape)} does not match {expected}")
        require_finite(latent, "Frozen VAE latent")

    @staticmethod
    def _sample_metadata(metadata, index, batch_size):
        result = {}
        for key, value in metadata.items():
            if torch.is_tensor(value) and value.ndim and value.shape[0] == batch_size:
                result[key] = value[index]
            elif isinstance(value, (list, tuple)) and len(value) == batch_size:
                result[key] = value[index]
            else:
                result[key] = value
        return result

    def encode_observation(self, observed, text, observed_valid=None):
        if observed.ndim != 4 or observed.shape[1:] != (3, self.config.rgb_height, self.config.rgb_width):
            raise ValueError("Observed RGB must be [B,3,384,320] in adapter ImageNet-normalized coordinates")
        if not torch.isfinite(observed).all():
            raise ValueError("Non-finite observed RGB")
        pixel_valid = (torch.ones_like(observed[:, :1], dtype=torch.bool) if observed_valid is None else observed_valid)
        if pixel_valid.dtype != torch.bool:
            raise TypeError("Observed pixel validity must be boolean; implicit coercion is forbidden")
        if pixel_valid.shape != observed[:, :1].shape:
            raise ValueError("Observed pixel mask shape mismatch")
        if not pixel_valid.flatten(1).any(1).all():
            raise ValueError("Every sample must contain at least one valid observed RGB view")
        latent = self._encode_video("observation", observed[:, :, None], {})
        context, context_valid = self._encode_text(text)
        if context.shape[0] != latent.shape[0]:
            raise ValueError("Text batch must match observed RGB batch")
        spatial = F.interpolate(pixel_valid.float(), size=latent.shape[-2:], mode="area") >= 1 - 1e-6
        return latent, context, context_valid, spatial[:, :, None], pixel_valid

    def _placeholder(self, batch_size, device):
        zero = torch.zeros(batch_size, 3, 1, self.config.motion_height, self.config.motion_width, device=device)
        return self._encode_video("motion_placeholder", zero, {})

    def _shapes(self, batch_size):
        cfg = self.config
        rgb = (batch_size, cfg.latent_channels, 9, cfg.rgb_height // 16, cfg.rgb_width // 16) if cfg.enable_rgb else None
        motion = (batch_size, cfg.latent_channels, 9, cfg.motion_height // 16, cfg.motion_width // 16) if cfg.enable_motion else None
        return rgb, motion

    def sample_generated_latents(self, observed_latent, context, context_valid, observed_valid, *, seed, steps):
        rgb_shape, motion_shape = self._shapes(observed_latent.shape[0])
        placeholder = self._placeholder(observed_latent.shape[0], observed_latent.device) if motion_shape else None
        return self.flow.sample_latents(observed_latent, context, context_valid, rgb_shape=rgb_shape,
                                       motion_shape=motion_shape, motion_placeholder=placeholder,
                                       seed=seed, steps=steps, observed_valid=observed_valid)

    def predict_3d_from_generated(self, observed_latent, context, context_valid, generated_rgb, generated_motion,
                                  observed_valid, pixel_valid):
        if self.future_3d is None:
            return {}
        b = observed_latent.shape[0]
        zero_t = observed_latent.new_zeros(b)
        rgb_valid = observed_valid.expand(-1, -1, 9, -1, -1) if generated_rgb is not None else None
        states, experts, _, _ = self.flow.prepare_states(observed_latent, context, context_valid,
                                                        generated_rgb, generated_motion, zero_t, zero_t,
                                                        observed_valid=observed_valid, rgb_valid=rgb_valid)
        view_valid = self.observed_view_validity(pixel_valid)
        states["G"] = self.future_3d.prepare(b, view_valid)
        experts["G"] = self.future_3d
        _, captured = self.flow.mot(states, experts, capture_layers=self.future_3d.student_layers)
        return self.future_3d.project(captured)

    @staticmethod
    def observed_view_validity(pixel_valid):
        h, w = pixel_valid.shape[-2:]
        return torch.stack([pixel_valid[:, :, :h * 2 // 3].flatten(1).any(1),
                            pixel_valid[:, :, h * 2 // 3:, :w // 2].flatten(1).any(1),
                            pixel_valid[:, :, h * 2 // 3:, w // 2:].flatten(1).any(1)], 1)

    def prepare_training_targets(self, batch, observed_latent):
        target = batch["targets"]
        observed = batch["observed_rgb"]
        metadata = {key: batch.get(key) for key in ["sample_id", "sidecar_digest", "prompt_version", "frame_timestamps"]}
        clean, masks = {}, {}
        if self.config.enable_rgb:
            video = self.layout.assemble_rgb(observed, target["future_rgb"])
            clean["rgb"] = self._encode_video("rgb_targets", video, metadata)
            self.layout.check_latents(clean["rgb"], self.config.rgb_height, self.config.rgb_width)
            future_valid = self.layout.latent_validity(target["rgb_valid"], clean["rgb"].shape[-2:])
            masks["rgb"] = torch.cat([torch.ones_like(future_valid[:, :, :1]), future_valid], 2)
        if self.config.enable_motion:
            video = self.layout.prepend_motion_placeholder(target["future_motion"])
            clean["motion"] = self._encode_video("motion_targets", video, metadata)
            self.layout.check_latents(clean["motion"], self.config.motion_height, self.config.motion_width)
            future_valid = self.layout.latent_validity(target["motion_valid"], clean["motion"].shape[-2:])
            masks["motion"] = torch.cat([torch.ones_like(future_valid[:, :, :1]), future_valid], 2)
        teacher, teacher_valid = {}, None
        if self.config.enable_3d:
            endpoint_metadata = {key: batch.get(key) for key in [
                "sidecar_digest", "prompt_version", "episode_index", "subtask_id", "goal_timestamp"]}
            feature_rows, valid_rows = [], []
            for index, views in enumerate(target["goal_rgb_views"]):
                view_valid = target["goal_view_valid"][index:index + 1]
                def extract(views=views, view_valid=view_valid):
                    features, valid = self.teacher.features([views], view_valid, observed.device)
                    return {**{f"layer_{key}": value for key, value in features.items()}, "valid": valid}
                values = self._cached("omega_targets", {"views": views, "valid": view_valid,
                                                       "metadata": self._sample_metadata(endpoint_metadata, index, observed.shape[0]),
                                                       "preprocess": OMEGA_PREPROCESS_VERSION, "layers": self.config.teacher_layers},
                                      extract, ["omega_checkpoint_path"])
                feature_rows.append(values)
                valid_rows.append(values["valid"])
            teacher = {layer: torch.cat([row[f"layer_{layer}"] for row in feature_rows]) for layer in self.config.teacher_layers}
            teacher_valid = torch.cat(valid_rows)
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
        active = active or (cfg.enable_3d and weights["3d"] > 0
                            and cfg.feature_mse_weight + cfg.feature_cosine_weight > 0
                            and teacher_valid is not None and teacher_valid.any().item())
        if not active:
            raise ValueError("Batch has no valid positive-weight supervision; refusing a zero-loss optimizer update")

    def _training_forward(self, batch):
        if "targets" not in batch:
            raise ValueError("training_forward requires an explicit targets dictionary")
        if "observed_valid" not in batch:
            raise ValueError("Training requires explicit observed_valid; missing views cannot default to valid")
        from .validation import require_bool, require_finite

        require_bool(batch["observed_valid"], "Observed validity")
        target = batch["targets"]
        for name, enabled in [("rgb", self.config.enable_rgb), ("motion", self.config.enable_motion)]:
            if enabled:
                require_bool(target[f"{name}_valid"], f"{name} target validity")
                require_finite(target[f"future_{name}"], f"Future {name} pixels")
        if self.config.enable_3d:
            require_bool(target["goal_view_valid"], "Goal view validity")
        observed, context, context_valid, observed_valid, pixel_valid = self.encode_observation(
            batch["observed_rgb"], batch["condition_text"], batch.get("observed_valid"))
        clean, masks, teacher_targets, teacher_valid = self.prepare_training_targets(batch, observed)
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
                                        rgb_valid=masks.get("rgb"), motion_valid=masks.get("motion"))
        losses, metrics = {}, {}
        for name in clean:
            losses[name] = self.flow.flow_loss(prediction[name], velocities[name], timesteps[name], masks[name])
            metrics[f"loss_{name}_flow"] = losses[name].detach().item()
        if self.config.enable_3d:
            seed = int(torch.randint(0, 2 ** 31 - 2, (), device=observed.device).item())
            # Noise generation is completely independent of clean/noisy targets.
            with torch.no_grad():
                generated = self.sample_generated_latents(observed, context, context_valid, observed_valid,
                                                          seed=seed, steps=self.config.goal_sampler_steps)
            features = self.predict_3d_from_generated(observed, context, context_valid, *generated, observed_valid, pixel_valid)
            losses["3d"], feature_metrics = geometric_feature_loss(features, teacher_targets, teacher_valid,
                                                                  mse_weight=self.config.feature_mse_weight,
                                                                  cosine_weight=self.config.feature_cosine_weight)
            metrics.update(feature_metrics)
            metrics["loss_3d_feature"] = losses["3d"].detach().item()
        loss = sum(weights[name] * value for name, value in losses.items())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite WSA loss: {metrics}")
        metrics["loss"] = loss.detach().item()
        return loss, metrics

    def forward(self, batch):
        return self.training_forward(batch)

    @torch.no_grad()
    def predict_future(self, observed_rgb, text, seed=0, *, observed_valid=None):
        was_training = self.training
        self.eval()
        try:
            with self.compute_context():
                return self._predict_future(observed_rgb, text, seed, observed_valid)
        finally:
            self.train(was_training)

    def _predict_future(self, observed_rgb, text, seed, observed_valid):
        observed, context, context_valid, latent_valid, pixel_valid = self.encode_observation(observed_rgb, text, observed_valid)
        generated_rgb, generated_motion = self.sample_generated_latents(observed, context, context_valid, latent_valid,
                                                                        seed=seed, steps=self.config.sampler_steps)
        features = self.predict_3d_from_generated(observed, context, context_valid, generated_rgb, generated_motion,
                                                  latent_valid, pixel_valid)
        goal_valid = self.observed_view_validity(pixel_valid).repeat_interleave(273, dim=1).reshape(-1, 3, 273)
        features = {key: value.masked_fill(~goal_valid[..., None], 0) for key, value in features.items()}
        result = {"goal_3d_features": features[self.config.teacher_layers[-1]] if features else None,
                  "goal_3d_layer_features": features, "goal_3d_valid": goal_valid,
                  "future_rgb_valid": pixel_valid[:, None].expand(-1, 32, -1, -1, -1)}
        for name, latent in [("rgb", generated_rgb), ("motion", generated_motion)]:
            if latent is not None:
                decoded = self.layout.future_from_decoded(self.vae.decode(latent))
                if not torch.isfinite(decoded).all():
                    raise FloatingPointError(f"Non-finite decoded future {name}")
                result[f"future_{name}"] = imagenet_denormalize(decoded)
                if name == "rgb":
                    result["future_rgb"] = result["future_rgb"].masked_fill(~result["future_rgb_valid"], 0)
            else:
                result[f"future_{name}"] = None
        return result
