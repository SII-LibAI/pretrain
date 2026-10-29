"""RGB, Motion and Action experts, with their joint MoT and flow sampling."""
from dataclasses import replace

import torch
from torch import nn

from .mot import CausalMoT, ExpertState
from .text_embeddings import validate_text_embedding
from .utils.temporal import TemporalLayout
from .utils.validation import require_bool, require_finite
from .vendor.scheduler import WanContinuousFlowMatchScheduler
from .vendor.wan_video_dit import DiTBlock, WanVideoDiT, precompute_freqs_cis, sinusoidal_embedding_1d


class RGBVideoExpert(WanVideoDiT):
    def forward(self, *args, **kwargs):
        raise RuntimeError("Use WorldFlow.velocity so O/R/M/G causal rules are enforced")


class MotionDiTExpert(WanVideoDiT):
    """An independent set of Wan-style Transformer weights; no observed motion."""

    def forward(self, *args, **kwargs):
        raise RuntimeError("Use WorldFlow.velocity with RGB/text observations only")


class ActionExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config.action_hidden_dim
        self.action_encoder = nn.Linear(config.action_dim, width)
        self.state_encoder = nn.Linear(config.state_dim, width)
        self.text_embedding = nn.Sequential(nn.Linear(config.text_dim, width), nn.GELU(approximate="tanh"),
                                            nn.Linear(width, width))
        self.time_embedding = nn.Sequential(nn.Linear(config.freq_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 6))
        self.blocks = nn.ModuleList([DiTBlock(width, config.head_dim, config.num_heads, config.action_ffn_dim)
                                    for _ in range(config.action_num_layers or config.num_layers)])
        self.head = nn.Linear(width, config.action_dim)
        positions = torch.arange(config.action_chunk_size, dtype=torch.float32)
        self.register_buffer("position", sinusoidal_embedding_1d(width, positions)[None], persistent=False)
        self.register_buffer("freqs", precompute_freqs_cis(config.head_dim, config.action_chunk_size)[:, None],
                             persistent=False)
        self.scheduler = WanContinuousFlowMatchScheduler(shift=config.action_flow_shift)

    def prepare(self, noisy, timestep, state, context, context_valid, valid=None):
        cfg = self.config
        if noisy.shape != (state.shape[0], cfg.action_chunk_size, cfg.action_dim):
            raise ValueError("Action noise must be [B,action_chunk_size,action_dim]")
        if state.shape != (noisy.shape[0], cfg.state_dim) or timestep.shape != (noisy.shape[0],):
            raise ValueError("Action state/timestep shape mismatch")
        require_finite(noisy, "Action input")
        require_finite(state, "Action state")
        self.scheduler._validate_timestep(timestep)
        validate_text_embedding(context, context_valid, cfg.max_text_length, cfg.text_dim)
        dtype = self.action_encoder.weight.dtype
        tokens = self.action_encoder(noisy.to(dtype)) + self.position.to(noisy.device, dtype)
        t = self.time_embedding(sinusoidal_embedding_1d(cfg.freq_dim, timestep).to(dtype))
        modulation = self.time_projection(t).reshape(noisy.shape[0], 6, -1)
        # Proprioception is an Action-only context token; O/R/M/G never receive state.
        text = self.text_embedding(context.to(dtype))
        text = torch.cat([text, self.state_encoder(state.to(dtype))[:, None]], 1)
        text_valid = torch.cat([context_valid, torch.ones(noisy.shape[0], 1, dtype=torch.bool, device=noisy.device)], 1)
        if valid is None:
            valid = torch.ones(noisy.shape[:2], dtype=torch.bool, device=noisy.device)
        require_bool(valid, "Action token validity")
        if valid.shape != noisy.shape[:2]:
            raise ValueError("Action token validity shape mismatch")
        return ExpertState(tokens, modulation, valid, self.freqs.to(noisy.device), text, text_valid)

    def training_state(self, target, *, noise=None, timestep=None):
        require_finite(target, "Action labels")
        if noise is None:
            noise = torch.randn(target.shape, device=target.device, dtype=torch.float32)
        if noise.shape != target.shape:
            raise ValueError("Action noise/target shapes differ")
        require_finite(noise, "Action noise")
        if timestep is None:
            timestep = self.scheduler.sample_training_t(target.shape[0], target.device, torch.float32)
        return self.scheduler.add_noise(target, noise, timestep), noise.float() - target.float(), timestep

    def loss(self, predicted, target, timestep, valid):
        require_bool(valid, "Action loss validity")
        if predicted.shape != target.shape or valid.shape != target.shape:
            raise ValueError("Action prediction/target/element mask shapes differ")
        require_finite(predicted, "Action velocity")
        require_finite(target, "Action target velocity")
        weights = self.scheduler.training_weight(timestep).reshape(-1, 1, 1)
        error = (predicted.float() - target.float()).square().masked_fill(~valid, 0)
        return (error * weights).sum() / valid.sum().clamp_min(1)


def wan_config(hidden_dim=3072, ffn_dim=14336, *, layers=30, heads=24, head_dim=128,
               channels=48, text_dim=4096, freq_dim=256):
    return dict(hidden_dim=hidden_dim, in_dim=channels, out_dim=channels, ffn_dim=ffn_dim,
                text_dim=text_dim, freq_dim=freq_dim, eps=1e-6, patch_size=(1, 2, 2),
                num_heads=heads, attn_head_dim=head_dim, num_layers=layers,
                has_image_input=False, seperated_timestep=True, fuse_vae_embedding_in_latents=True)


def state_from_pre(pre, valid):
    return ExpertState(pre["tokens"], pre["t_mod"], valid, pre["freqs"], pre["context"],
                       # WSA expands context masks along query dimension.
                       pre["context_mask"][:, 0])


def token_validity(latent_valid, grid, *, patch_size=(1, 2, 2)):
    # Fully masked temporal/spatial patches are never visible keys.
    require_bool(latent_valid, "Latent validity")
    pooled = torch.nn.functional.avg_pool3d(latent_valid.float(), patch_size, stride=patch_size)
    if tuple(pooled.shape[2:]) != tuple(grid):
        raise ValueError("Latent validity/patch grid mismatch")
    return (pooled >= 1 - 1e-6).flatten(2)[:, 0]


class WorldFlow(nn.Module):
    def __init__(self, rgb, motion=None, *, backend="sdpa", gradient_checkpointing=False, shift=5.0,
                 future_frames=32):
        super().__init__()
        self.layout = TemporalLayout(future_frames=future_frames)
        self.rgb, self.motion = rgb, motion
        self.mot = CausalMoT(backend=backend, gradient_checkpointing=gradient_checkpointing)
        self.scheduler = WanContinuousFlowMatchScheduler(shift=shift)

    def prepare_states(self, observed_latent, text, text_valid, rgb_latent=None, motion_latent=None,
                       rgb_t=None, motion_t=None, observed_valid=None, rgb_valid=None, motion_valid=None,
                       history_latent=None, history_valid=None):
        require_bool(text_valid, "Text validity")
        if text_valid.shape != text.shape[:2]:
            raise ValueError("Text validity must match the context exactly")
        b = observed_latent.shape[0]
        zero_t = observed_latent.new_zeros(b)
        if observed_valid is None:
            observed_valid = torch.ones(b, 1, *observed_latent.shape[2:], device=observed_latent.device, dtype=torch.bool)
        observation_latents, observation_valid = observed_latent, observed_valid
        history_count = 0
        if history_latent is not None:
            if history_valid is None or history_latent.ndim != 5:
                raise ValueError("History requires [B,C,T,H,W] latents and explicit validity")
            history_count = history_latent.shape[2] - 1
            observation_latents, observation_valid = history_latent, history_valid
        observed_pre = self.rgb.pre_dit(observation_latents, zero_t, text, text_valid,
                                         fuse_vae_embedding_in_latents=True)
        obs_n = observed_pre["meta"]["tokens_per_frame"]
        observation = state_from_pre(observed_pre, token_validity(observation_valid, observed_pre["meta"]["grid_size"]))
        states, experts, pre_states = {"O": observation}, {"O": self.rgb}, {}
        if rgb_latent is not None:
            if rgb_latent.ndim != 5 or rgb_latent.shape[2] != self.layout.latent_frames or rgb_t is None:
                raise ValueError(f"RGB flow requires {self.layout.latent_frames} latent timepoints and a timestep")
            rgb_latent = torch.cat([observed_latent, rgb_latent[:, :, 1:]], 2)
            pre = self.rgb.pre_dit(rgb_latent, rgb_t, text, text_valid, fuse_vae_embedding_in_latents=True,
                                   temporal_rope_positions=range(history_count, history_count + rgb_latent.shape[2]))
            valid = (torch.ones(b, 1, *rgb_latent.shape[2:], dtype=torch.bool, device=rgb_latent.device)
                     if rgb_valid is None else rgb_valid)
            rgb_state = state_from_pre(pre, token_validity(valid, pre["meta"]["grid_size"]))
            states["R"] = replace(rgb_state, tokens=rgb_state.tokens[:, obs_n:],
                                  modulation=rgb_state.modulation[:, obs_n:], valid=rgb_state.valid[:, obs_n:],
                                  freqs=rgb_state.freqs[obs_n:])
            experts["R"], pre_states["R"] = self.rgb, pre
        if motion_latent is not None:
            if (self.motion is None or motion_t is None or motion_latent.ndim != 5
                    or motion_latent.shape[2] != self.layout.latent_frames):
                raise ValueError(f"Motion flow requires an expert, timestep and {self.layout.latent_frames} latent timepoints")
            pre = self.motion.pre_dit(motion_latent, motion_t, text, text_valid, fuse_vae_embedding_in_latents=True,
                                      temporal_rope_positions=range(history_count, history_count + motion_latent.shape[2]))
            valid = (torch.ones(b, 1, *motion_latent.shape[2:], dtype=torch.bool, device=motion_latent.device)
                     if motion_valid is None else motion_valid)
            motion_state = state_from_pre(pre, token_validity(valid, pre["meta"]["grid_size"]))
            # t0 exists only for the VAE's 4n+1 layout. Remove its tokens before
            # MoT so it never supplies Q/K/V, text queries, or keys read by G.
            prefix_n = pre["meta"]["tokens_per_frame"]
            states["M"] = replace(motion_state, tokens=motion_state.tokens[:, prefix_n:],
                                  modulation=motion_state.modulation[:, prefix_n:],
                                  valid=motion_state.valid[:, prefix_n:], freqs=motion_state.freqs[prefix_n:])
            experts["M"], pre_states["M"] = self.motion, pre
        return states, experts, pre_states, obs_n

    def velocity(self, observed_latent, text, text_valid, rgb_latent=None, motion_latent=None,
                 rgb_t=None, motion_t=None, *, future_3d=None, goal_view_valid=None, timer=None,
                 action_expert=None, action_latent=None, action_t=None, state=None, action_valid=None,
                 project_3d=True, **validity):
        states, experts, pre, obs_n = self.prepare_states(observed_latent, text, text_valid,
                                                       rgb_latent, motion_latent, rgb_t, motion_t, **validity)
        capture_layers = ()
        if future_3d is not None:
            if goal_view_valid is None:
                raise ValueError("Joint 3D prediction requires observed view validity")
            states["G"] = future_3d.prepare(observed_latent.shape[0], goal_view_valid,
                                             num_train_timesteps=self.scheduler.num_train_timesteps)
            experts["G"] = future_3d
            capture_layers = future_3d.student_layers
        if action_expert is not None:
            if action_latent is None or action_t is None or state is None:
                raise ValueError("Parallel action prediction requires action state, timestep and robot state")
            states["A"] = action_expert.prepare(action_latent, action_t, state, text, text_valid, action_valid)
            experts["A"] = action_expert
        if timer is not None:
            timer.mark("input_s")
        tokens, captured = self.mot(states, experts, capture_layers=capture_layers if project_3d else ())
        if timer is not None:
            timer.mark("mot_s")
        result = {}
        if "R" in pre:
            full = torch.cat([tokens["O"][:, -obs_n:], tokens["R"]], 1)
            result["rgb"] = self.rgb.post_dit(full, pre["R"])
        if "M" in pre:
            # Restore the codec layout only after all attention has finished.
            # The placeholder has no predicted velocity or supervision.
            prefix_n = pre["M"]["meta"]["tokens_per_frame"]
            full = torch.cat([torch.zeros_like(pre["M"]["tokens"][:, :prefix_n]), tokens["M"]], 1)
            motion = self.motion.post_dit(full, pre["M"])
            result["motion"] = torch.cat([torch.zeros_like(motion[:, :, :1]), motion[:, :, 1:]], 2)
        if future_3d is not None and project_3d:
            result["3d"] = future_3d.project(captured)
        if action_expert is not None:
            result["action"] = action_expert.head(tokens["A"]).masked_fill(~states["A"].valid[..., None], 0)
        if timer is not None:
            timer.mark("pred_s")
        return result

    def training_state(self, clean, condition, timestep=None, noise=None):
        if (clean.ndim != 5 or clean.shape[2] != self.layout.latent_frames
                or condition.shape != clean[:, :, :1].shape):
            raise ValueError(f"Flow target requires {self.layout.latent_frames} latent timepoints and a matching t0 condition")
        if timestep is None:
            timestep = self.scheduler.sample_training_t(clean.shape[0], clean.device, clean.dtype)
        if noise is None:
            noise = torch.randn(clean.shape, device=clean.device, dtype=torch.float32)
        require_finite(clean, "Clean flow target")
        require_finite(condition, "Flow condition")
        require_finite(noise, "Flow noise")
        if noise.shape != clean.shape:
            raise ValueError("Flow noise must match the clean target exactly")
        if timestep.shape != (clean.shape[0],):
            raise ValueError("Training timestep must have exactly one entry per sample")
        noisy = self.scheduler.add_noise(clean, noise, timestep)
        noisy = torch.cat([condition, noisy[:, :, 1:]], 2)
        return noisy, noise.float() - clean.float(), timestep

    def flow_loss(self, prediction, target, timestep, valid):
        require_bool(valid, "Flow loss validity")
        require_finite(prediction, "Flow prediction")
        require_finite(target, "Flow target")
        if (target.ndim != 5 or target.shape[2] != self.layout.latent_frames
                or prediction.shape != target.shape or valid.shape != (target.shape[0], 1, *target.shape[2:])):
            raise ValueError("Flow prediction/target/valid shapes disagree")
        valid = valid.clone()
        valid[:, :, 0] = False  # Only future latent groups contribute.
        error = (prediction.float() - target.float()).square().masked_fill(~valid, 0)
        counts = valid.flatten(1).sum(1) * target.shape[1]
        per_sample = error.flatten(1).sum(1) / counts.clamp_min(1)
        weights = self.scheduler.training_weight(timestep).reshape(-1)
        if weights.shape != (target.shape[0],):
            raise ValueError("Flow loss timestep must match the batch exactly")
        active = counts > 0
        if active.any() and not (weights[active] > 0).any():
            raise ValueError("Active flow supervision has zero timestep weight")
        return (per_sample * weights * active).sum() / active.sum().clamp_min(1)

    def sample_latents(self, observed_latent, text, text_valid, *, rgb_shape=None, motion_shape=None,
                       motion_placeholder=None, seed=0, steps=20, observed_valid=None,
                       future_3d=None, goal_view_valid=None, return_features=False,
                       action_expert=None, state=None, history_latent=None, history_valid=None):
        """Joint denoising; optional 3D features come from the last mixed forward.

        Inputs are observations/text plus shapes/constants; there is no motion
        observation. G is rebuilt from fixed slot noise at each denoising step.
        """
        for name, shape in [("RGB", rgb_shape), ("Motion", motion_shape)]:
            if shape is not None and (len(shape) != 5 or shape[2] != self.layout.latent_frames):
                raise ValueError(f"{name} sampler requires {self.layout.latent_frames} latent timepoints")
        rgb_gen = torch.Generator(device=observed_latent.device).manual_seed(int(seed))
        motion_gen = torch.Generator(device=observed_latent.device).manual_seed(int(seed) + 1)
        # Keep the Euler state in FP32; transformer operations still use AMP.
        rgb = torch.randn(rgb_shape, generator=rgb_gen, device=observed_latent.device, dtype=torch.float32) if rgb_shape else None
        motion = torch.randn(motion_shape, generator=motion_gen, device=observed_latent.device, dtype=torch.float32) if motion_shape else None
        if motion is not None and motion_placeholder is None:
            raise ValueError("Motion sampler requires a separately encoded fixed-zero placeholder")
        if motion is not None and motion_placeholder.shape != motion[:, :, :1].shape:
            raise ValueError("Motion placeholder must match the sampler t0 shape exactly")
        if motion_placeholder is not None:
            require_finite(motion_placeholder, "Motion placeholder")
        rgb_flags = (observed_valid.expand(-1, -1, self.layout.latent_frames, -1, -1)
                     if rgb is not None and observed_valid is not None else None)
        if rgb_flags is not None:
            rgb = rgb.masked_fill(~rgb_flags, 0)
        times, deltas = self.scheduler.build_inference_schedule(steps, observed_latent.device, observed_latent.dtype)
        features = {}
        action = None
        if action_expert is not None:
            if state is None:
                raise ValueError("Action sampling requires the current normalized state")
            cfg = action_expert.config
            generator = torch.Generator(device=observed_latent.device).manual_seed(int(seed) + 2)
            action = torch.randn(observed_latent.shape[0], cfg.action_chunk_size, cfg.action_dim,
                                 generator=generator, device=observed_latent.device, dtype=torch.float32)
            action_times, action_deltas = action_expert.scheduler.build_inference_schedule(steps, observed_latent.device, torch.float32)
        for index, (time, delta) in enumerate(zip(times, deltas, strict=True)):
            t = time.expand(observed_latent.shape[0])
            if rgb is not None:
                rgb = torch.cat([observed_latent, rgb[:, :, 1:]], 2)
            if motion is not None:
                motion = torch.cat([motion_placeholder, motion[:, :, 1:]], 2)
            velocity = self.velocity(observed_latent, text, text_valid, rgb, motion,
                                     t if rgb is not None else None, t if motion is not None else None,
                                     observed_valid=observed_valid, rgb_valid=rgb_flags,
                                     future_3d=future_3d, goal_view_valid=goal_view_valid,
                                     history_latent=history_latent, history_valid=history_valid,
                                     action_expert=action_expert, action_latent=action,
                                     action_t=action_times[index].expand(observed_latent.shape[0]) if action is not None else None, state=state,
                                     project_3d=return_features or action_expert is None)
            features = velocity.get("3d", {})
            if rgb is not None:
                rgb = self.scheduler.step(velocity["rgb"], delta, rgb)
                if rgb_flags is not None:
                    rgb = rgb.masked_fill(~rgb_flags, 0)
            if motion is not None:
                motion = self.scheduler.step(velocity["motion"], delta, motion)
            if action is not None:
                action = action_expert.scheduler.step(velocity["action"], action_deltas[index], action)
        if rgb is not None:
            rgb = torch.cat([observed_latent, rgb[:, :, 1:]], 2)
        if motion is not None:
            motion = torch.cat([motion_placeholder, motion[:, :, 1:]], 2)
        if action is not None:
            require_finite(action, "Sampled actions")
            return {"rgb": rgb, "motion": motion, "action": action, "3d": features}
        return (rgb, motion, features) if return_features else (rgb, motion)


class FrozenWanVAE(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self, video):
        if video.shape[2] % 4 != 1:
            raise ValueError("Refusing Wan VAE's silent temporal tail truncation; supply 4n+1 frames")
        return self.vae.encode(video, device=video.device)

    @torch.no_grad()
    def decode(self, latent):
        # Return raw values so the policy can reject non-finite output before clamping.
        return torch.cat([self.vae.model.decode(z[None], self.vae.scale) for z in latent], 0)
