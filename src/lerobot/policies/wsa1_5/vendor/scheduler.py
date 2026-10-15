import math

import torch


class WanContinuousFlowMatchScheduler:
    """Continuous-time Flow-Matching scheduler with shift-based sampling."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10):
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("Scheduler epsilon must be finite and positive")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()
        if self._weight_norm_const <= 0:
            raise ValueError("Scheduler has no positive training-weight normalization")

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
        y_min = float(y_grid.min().item())
        y_shifted_grid = y_grid - y_min
        norm_const = float(y_shifted_grid.mean().item())
        return y_min, norm_const

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = self._phi(u, self.shift)
        timestep = sigma * float(self.num_train_timesteps)
        # BF16 rounds many distinct continuous times (e.g. 997 -> 996).
        # dtype describes model compute, never the integration precision.
        return timestep

    def _validate_timestep(self, timestep):
        if (not torch.is_tensor(timestep) or not torch.isfinite(timestep).all()
                or (timestep < 0).any() or (timestep > self.num_train_timesteps).any()):
            raise ValueError("Flow timestep must be finite and within [0,num_train_timesteps]")
        if timestep.dtype != torch.float32:
            raise TypeError("Flow timestep must be FP32; low-precision time quantization is forbidden")

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        self._validate_timestep(timestep)
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        y_shifted = y - self._y_min
        weight = y_shifted / (self._weight_norm_const + self.eps)
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        self._validate_timestep(timestep)
        sigma = (timestep / float(self.num_train_timesteps)).to(original_samples.device)
        if sigma.ndim == 0:
            return (1 - sigma) * original_samples.float() + sigma * noise.float()
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples.float() + sigma * noise.float()

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps, deltas

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(model_output).all():
            raise FloatingPointError("Non-finite flow sampler velocity")
        if (delta.dtype != torch.float32 or not torch.isfinite(delta).all()
                or (delta > 0).any() or (delta < -1).any()):
            raise ValueError("Flow Euler delta must be finite FP32 in [-1,0]")
        delta = delta.to(sample.device)
        if delta.ndim == 0:
            return sample.float() + model_output.float() * delta
        delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return sample.float() + model_output.float() * delta
