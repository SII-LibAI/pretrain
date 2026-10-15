"""Wan RGB/Motion experts. Importing this file never constructs pretrained modules."""
from dataclasses import replace

import torch
from torch import nn

from .mot import CausalMoT, ExpertState
from .validation import require_bool, require_finite
from .vendor.scheduler import WanContinuousFlowMatchScheduler
from .vendor.wan_video_dit import WanVideoDiT, precompute_freqs_cis


class RGBVideoExpert(WanVideoDiT):
    def forward(self, *args, **kwargs):
        raise RuntimeError("Use WorldFlow.velocity so O/R/M/G causal rules are enforced")


class MotionDiTExpert(WanVideoDiT):
    """An independent set of Wan-style Transformer weights; no observed motion."""

    def forward(self, *args, **kwargs):
        raise RuntimeError("Use WorldFlow.velocity with RGB/text observations only")


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
    def __init__(self, rgb, motion=None, *, backend="sdpa", gradient_checkpointing=False, shift=5.0):
        super().__init__()
        self.rgb, self.motion = rgb, motion
        self.mot = CausalMoT(backend=backend, gradient_checkpointing=gradient_checkpointing)
        self.scheduler = WanContinuousFlowMatchScheduler(shift=shift)

    def prepare_states(self, observed_latent, text, text_valid, rgb_latent=None, motion_latent=None,
                       rgb_t=None, motion_t=None, observed_valid=None, rgb_valid=None, motion_valid=None):
        require_bool(text_valid, "Text validity")
        if text_valid.shape != text.shape[:2]:
            raise ValueError("Text validity must match the context exactly")
        b = observed_latent.shape[0]
        zero_t = observed_latent.new_zeros(b)
        observed_pre = self.rgb.pre_dit(observed_latent, zero_t, text, text_valid, fuse_vae_embedding_in_latents=True)
        obs_n = observed_pre["tokens"].shape[1]
        if observed_valid is None:
            observed_valid = torch.ones(b, 1, *observed_latent.shape[2:], device=observed_latent.device, dtype=torch.bool)
        observed_tokens_valid = token_validity(observed_valid, observed_pre["meta"]["grid_size"])
        # Text tokens belong to O and cannot read any future branch.
        text_tokens = observed_pre["context"]
        text_freqs = precompute_freqs_cis(self.rgb.attn_head_dim, end=text.shape[1]).to(observed_latent.device)[:, None]
        text_mod = observed_pre["t_mod"][:, :1].expand(-1, text.shape[1], -1, -1)
        observation = ExpertState(torch.cat([observed_pre["tokens"], text_tokens], 1),
                                  torch.cat([observed_pre["t_mod"], text_mod], 1),
                                  torch.cat([observed_tokens_valid, text_valid], 1),
                                  torch.cat([observed_pre["freqs"], text_freqs], 0),
                                  observed_pre["context"], text_valid)
        states, experts, pre_states = {"O": observation}, {"O": self.rgb}, {}
        if rgb_latent is not None:
            if rgb_latent.shape[2] != 9 or rgb_t is None:
                raise ValueError("RGB flow requires the explicit 9-frame latent sequence and timestep")
            # Never trust the clean/noisy target's first latent as observed input.
            rgb_latent = torch.cat([observed_latent, rgb_latent[:, :, 1:]], 2)
            pre = self.rgb.pre_dit(rgb_latent, rgb_t, text, text_valid, fuse_vae_embedding_in_latents=True)
            valid = (torch.ones(b, 1, *rgb_latent.shape[2:], dtype=torch.bool, device=rgb_latent.device)
                     if rgb_valid is None else rgb_valid)
            rgb_state = state_from_pre(pre, token_validity(valid, pre["meta"]["grid_size"]))
            states["R"] = replace(rgb_state, tokens=rgb_state.tokens[:, obs_n:],
                                  modulation=rgb_state.modulation[:, obs_n:], valid=rgb_state.valid[:, obs_n:],
                                  freqs=rgb_state.freqs[obs_n:])
            experts["R"], pre_states["R"] = self.rgb, pre
        if motion_latent is not None:
            if self.motion is None or motion_t is None or motion_latent.shape[2] != 9:
                raise ValueError("Motion flow requires an expert, timestep and explicit 9-frame latent sequence")
            pre = self.motion.pre_dit(motion_latent, motion_t, text, text_valid, fuse_vae_embedding_in_latents=True)
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
                 rgb_t=None, motion_t=None, *, future_3d=None, goal_view_valid=None, **validity):
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
        tokens, captured = self.mot(states, experts, capture_layers=capture_layers)
        result = {}
        if "R" in pre:
            full = torch.cat([tokens["O"][:, :obs_n], tokens["R"]], 1)
            result["rgb"] = self.rgb.post_dit(full, pre["R"])
        if "M" in pre:
            # Restore the codec layout only after all attention has finished.
            # The placeholder has no predicted velocity or supervision.
            prefix_n = pre["M"]["meta"]["tokens_per_frame"]
            full = torch.cat([torch.zeros_like(pre["M"]["tokens"][:, :prefix_n]), tokens["M"]], 1)
            motion = self.motion.post_dit(full, pre["M"])
            result["motion"] = torch.cat([torch.zeros_like(motion[:, :, :1]), motion[:, :, 1:]], 2)
        if future_3d is not None:
            result["3d"] = future_3d.project(captured)
        return result

    def training_state(self, clean, condition, timestep=None, noise=None):
        if clean.ndim != 5 or clean.shape[2] != 9 or condition.shape != clean[:, :, :1].shape:
            raise ValueError("Flow target and separately encoded t0 condition must obey Wan temporal layout")
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
        if prediction.shape != target.shape or valid.shape != (target.shape[0], 1, *target.shape[2:]):
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
                       future_3d=None, goal_view_valid=None, return_features=False):
        """Joint denoising; optional 3D features come from the last mixed forward.

        Inputs are observations/text plus shapes/constants; there is no motion
        observation. G is rebuilt from fixed slot noise at each denoising step.
        """
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
        rgb_flags = observed_valid.expand(-1, -1, 9, -1, -1) if rgb is not None and observed_valid is not None else None
        if rgb_flags is not None:
            rgb = rgb.masked_fill(~rgb_flags, 0)
        times, deltas = self.scheduler.build_inference_schedule(steps, observed_latent.device, observed_latent.dtype)
        features = {}
        for time, delta in zip(times, deltas, strict=True):
            t = time.expand(observed_latent.shape[0])
            if rgb is not None:
                rgb = torch.cat([observed_latent, rgb[:, :, 1:]], 2)
            if motion is not None:
                motion = torch.cat([motion_placeholder, motion[:, :, 1:]], 2)
            velocity = self.velocity(observed_latent, text, text_valid, rgb, motion,
                                     t if rgb is not None else None, t if motion is not None else None,
                                     observed_valid=observed_valid, rgb_valid=rgb_flags,
                                     future_3d=future_3d, goal_view_valid=goal_view_valid)
            features = velocity.get("3d", {})
            if rgb is not None:
                rgb = self.scheduler.step(velocity["rgb"], delta, rgb)
                if rgb_flags is not None:
                    rgb = rgb.masked_fill(~rgb_flags, 0)
            if motion is not None:
                motion = self.scheduler.step(velocity["motion"], delta, motion)
        if rgb is not None:
            rgb = torch.cat([observed_latent, rgb[:, :, 1:]], 2)
        if motion is not None:
            motion = torch.cat([motion_placeholder, motion[:, :, 1:]], 2)
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
        # Official convenience decode clamps [-1,1]. Inputs here use ImageNet
        # normalization by request, so preserve the model's numeric output.
        return torch.cat([self.vae.model.decode(z[None], self.vae.scale) for z in latent], 0)
