"""Explicit Wan2.2 4n+1 temporal contract; no implicit tail truncation."""
from dataclasses import dataclass

import torch

from .validation import require_bool


@dataclass(frozen=True)
class TemporalLayout:
    future_frames: int = 32
    temporal_factor: int = 4
    spatial_factor: int = 16

    @property
    def latent_frames(self):
        if self.future_frames <= 0 or self.future_frames % self.temporal_factor:
            raise ValueError("Future horizon must be a positive multiple of the temporal factor")
        return 1 + self.future_frames // self.temporal_factor

    def prepend_motion_placeholder(self, future):
        if future.ndim != 5 or future.shape[1:3] != (self.future_frames, 3):
            raise ValueError("Expected future motion [B,32,3,H,W]")
        # Fixed zero in the model's normalized input domain, not observed motion.
        zero = torch.zeros_like(future[:, :1])
        return torch.cat([zero, future], 1).transpose(1, 2).contiguous()

    def assemble_rgb(self, observed, future):
        if future.ndim != 5 or future.shape[1:3] != (self.future_frames, 3):
            raise ValueError("Expected future RGB [B,32,3,H,W]")
        if observed.shape != (future.shape[0], *future.shape[2:]):
            raise ValueError("Observed and future RGB spatial/batch shapes must match")
        return torch.cat([observed[:, None], future], 1).transpose(1, 2).contiguous()

    def check_latents(self, latent, height, width):
        expected = (self.latent_frames, height // self.spatial_factor, width // self.spatial_factor)
        if height % self.spatial_factor or width % self.spatial_factor or latent.shape[2:] != expected:
            raise ValueError(f"Wan VAE latent alignment mismatch: {tuple(latent.shape[2:])} != {expected}")

    def future_from_decoded(self, decoded):
        if decoded.ndim != 5 or decoded.shape[2] != self.future_frames + 1:
            raise ValueError(f"Expected decoded 33-frame sequence, got {tuple(decoded.shape)}")
        # The validated mapping removes only the explicit t0 condition/placeholder.
        return decoded[:, :, 1:].transpose(1, 2).contiguous()

    def latent_validity(self, frame_valid, spatial_size):
        """Require complete groups and their causal VAE history to be valid."""
        require_bool(frame_valid, "Future frame validity")
        if frame_valid.ndim == 2:
            frame_valid = frame_valid[:, :, None, None, None]
        if frame_valid.ndim != 5 or frame_valid.shape[1:3] != (self.future_frames, 1):
            raise ValueError("Expected [B,32] or [B,32,1,H,W] validity")
        b, _, _, h, w = frame_valid.shape
        grouped = frame_valid.reshape(b, self.latent_frames - 1, self.temporal_factor, 1, h, w).all(2)
        # A later valid group still depends on earlier encoder cache entries.
        # Once history is missing, it cannot become a trustworthy clean target.
        grouped = grouped.to(torch.int32).cumprod(1).bool()
        resized = torch.nn.functional.interpolate(grouped.float().flatten(0, 1), size=spatial_size, mode="area")
        return (resized >= 1 - 1e-6).reshape(b, self.latent_frames - 1, 1, *spatial_size).transpose(1, 2)
