"""Configuration registered with InternVLA's actual draccus policy mechanism."""
from dataclasses import dataclass
import math

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import MultiGroupCosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("WSA_1_5")
@dataclass
class WSA1_5Config(PreTrainedConfig):
    wan_model_path: str = ""
    vae_path: str = ""
    text_encoder_path: str = ""
    tokenizer_path: str = ""
    omega_checkpoint_path: str = ""
    dtype: str = "bfloat16"
    trainable_weights_fp32: bool = False
    load_text_encoder: bool = False
    rgb_hidden_dim: int = 3072
    rgb_ffn_dim: int = 14336
    motion_hidden_dim: int = 1024
    motion_ffn_dim: int = 4096
    goal_hidden_dim: int = 768
    goal_ffn_dim: int = 3072
    num_layers: int = 30
    num_heads: int = 24
    head_dim: int = 128
    latent_channels: int = 48
    text_dim: int = 4096
    freq_dim: int = 256
    max_text_length: int = 512
    rgb_height: int = 384
    rgb_width: int = 320
    motion_height: int = 192
    motion_width: int = 160
    future_frames: int = 32
    teacher_layers: tuple[int, ...] = (4, 11, 17, 23)
    student_layers: tuple[int, ...] | None = None
    enable_rgb: bool = True
    enable_motion: bool = True
    enable_3d_queries: bool = True  # Expert/query existence, including zero-loss ablations.
    enable_3d: bool | None = None  # Legacy query flag, normalized into enable_3d_queries.
    lambda_rgb: float = 1.0
    lambda_motion: float = 1.0
    lambda_3d: float = 1.0  # Zero skips teacher construction, endpoint loading and 3D loss.
    feature_mse_weight: float = 1.0
    feature_cosine_weight: float = 1.0
    freeze_rgb: bool = False
    freeze_motion: bool = False
    freeze_3d: bool = False
    freeze_heads: bool = False
    motion_initialization: str = "wan_width_interpolation"
    goal_initialization: str = "wan_width_interpolation"
    attention_backend: str = "sdpa"
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1
    flow_shift: float = 5.0
    sampler_steps: int = 20
    # Legacy checkpoint field; joint 3D training performs no extra sampling.
    goal_sampler_steps: int | None = None
    goal_conditioning: str = "joint"
    rgb_lr: float = 1e-5
    motion_lr: float = 5e-5
    goal_lr: float = 5e-5
    heads_lr: float = 1e-4
    optimizer_weight_decay: float = 0.01
    scheduler_warmup_steps: int = 2000
    scheduler_decay_steps: int = 600000
    scheduler_decay_lr_ratio: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        for name in ["trainable_weights_fp32", "load_text_encoder"]:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        if self.enable_3d is not None:
            if type(self.enable_3d) is not bool:
                raise ValueError("enable_3d must be boolean when supplied")
            self.enable_3d_queries = self.enable_3d
            self.enable_3d = None
        if type(self.enable_3d_queries) is not bool:
            raise ValueError("3D query switch must be boolean")
        scalars = ["lambda_rgb", "lambda_motion", "lambda_3d", "feature_mse_weight", "feature_cosine_weight",
                   "rgb_lr", "motion_lr", "goal_lr", "heads_lr", "optimizer_weight_decay", "flow_shift",
                   "scheduler_decay_lr_ratio"]
        for name in scalars:
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        dimensions = ["rgb_hidden_dim", "rgb_ffn_dim", "motion_hidden_dim", "motion_ffn_dim",
                      "goal_hidden_dim", "goal_ffn_dim", "latent_channels", "text_dim", "freq_dim", "max_text_length",
                      "num_layers", "num_heads", "head_dim", "gradient_accumulation_steps", "sampler_steps",
                      "rgb_height", "rgb_width", "motion_height", "motion_width"]
        if any(type(getattr(self, name)) is not int or getattr(self, name) <= 0 for name in dimensions):
            raise ValueError("Model dimensions and step counts must be positive integers")
        if self.freq_dim % 2:
            raise ValueError("Wan sinusoidal frequency dimension must be even")
        if self.flow_shift <= 0 or self.optimizer_weight_decay < 0:
            raise ValueError("Flow shift must be positive and weight decay non-negative")
        if (type(self.scheduler_warmup_steps) is not int or type(self.scheduler_decay_steps) is not int
                or self.scheduler_warmup_steps < 0 or self.scheduler_decay_steps <= self.scheduler_warmup_steps):
            raise ValueError("Scheduler requires integer 0 <= warmup < decay steps")
        if not 0 <= self.scheduler_decay_lr_ratio <= 1:
            raise ValueError("scheduler_decay_lr_ratio must be within [0, 1]")
        if self.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("Unsupported WSA dtype")
        if self.future_frames != 32 or self.n_obs_steps != 1:
            raise ValueError("WSA1.5 first version requires one observation and 32 future frames")
        if not any((self.enable_rgb, self.enable_motion, self.enable_3d_queries)):
            raise ValueError("Enable at least one supervised modality")
        # Accept serialized legacy configs while making their new semantics explicit.
        if self.goal_conditioning == "generated":
            self.goal_conditioning = "joint"
        if self.goal_conditioning != "joint":
            raise ValueError("Only joint O/R/M/G mixed-attention 3D conditioning is supported")
        if self.motion_initialization not in {"wan_width_interpolation", "random"} or self.goal_initialization not in {"wan_width_interpolation", "random"}:
            raise ValueError("Unknown expert initialization rule")
        if self.goal_hidden_dim % 2:
            raise ValueError("goal_hidden_dim must be even for 3D slot positional encoding")
        if self.attention_backend not in {"sdpa", "math", "reference"}:
            raise ValueError("Attention backend must preserve explicit masks")
        rope_width = (self.head_dim - 2 * (self.head_dim // 3)) // 2 + 2 * ((self.head_dim // 3) // 2)
        if self.head_dim % 2 or rope_width != self.head_dim // 2:
            raise ValueError("head_dim must support the Wan three-axis RoPE partition")
        if self.teacher_layers != (4, 11, 17, 23):
            raise ValueError("Audited Omega checkpoint caches layers (4,11,17,23)")
        if any(size <= 0 or size % 32 for size in [self.rgb_height, self.rgb_width, self.motion_height, self.motion_width]):
            raise ValueError("Image dimensions must be positive multiples of VAE factor 16 * patch size 2")
        if min(self.lambda_rgb, self.lambda_motion, self.lambda_3d, self.feature_mse_weight, self.feature_cosine_weight) < 0:
            raise ValueError("Loss weights cannot be negative")
        if min(self.rgb_lr, self.motion_lr, self.goal_lr, self.heads_lr) <= 0:
            raise ValueError("Per-module learning rates must be positive")
        effective = [self.lambda_rgb if self.enable_rgb else 0, self.lambda_motion if self.enable_motion else 0,
                     self.lambda_3d * (self.feature_mse_weight + self.feature_cosine_weight) if self.use_3d_supervision else 0]
        if not any(weight > 0 for weight in effective):
            raise ValueError("At least one enabled modality must have positive effective loss weight")

    @property
    def use_3d_supervision(self):
        return (self.enable_3d_queries and self.lambda_3d > 0
                and self.feature_mse_weight + self.feature_cosine_weight > 0)

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return None

    @property
    def reward_delta_indices(self):
        return None

    def validate_features(self):
        # World-model pretraining requires no action or robot-state feature.
        if self.n_obs_steps != 1:
            raise ValueError("Only one observed RGB timepoint is supported")

    def get_optimizer_preset(self):
        return AdamWConfig(lr=self.motion_lr, weight_decay=self.optimizer_weight_decay, grad_clip_norm=1.0)

    def get_scheduler_preset(self):
        rates = {"rgb": self.rgb_lr, "motion": self.motion_lr, "3d": self.goal_lr, "heads": self.heads_lr}
        return MultiGroupCosineDecayWithWarmupSchedulerConfig(auto_scale_to_training_steps=False, group_schedules={name: {
            "peak_lr": lr, "decay_lr": lr * self.scheduler_decay_lr_ratio, "num_warmup_steps": self.scheduler_warmup_steps,
            "num_decay_steps": self.scheduler_decay_steps,
        } for name, lr in rates.items()})
