import logging
from dataclasses import dataclass, field, replace
from typing import Sequence

from lerobot.configs.default import DatasetConfig, VQADatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import MultiGroupCosineDecayWithWarmupSchedulerConfig
from lerobot.policies.wsa_pretrain.transform_wsa_pretrain import (
    ExtractVideoFramesTransformFn,
    FASTWSAPretrainActionTokenizerTransformFn,
    WSAPretrainChatProcessorTransformFn,
    WSAPretrainVQAProcessorTransformFn,
)
from lerobot.transforms.core import *
from lerobot.utils.constants import HF_HOME, OBS_IMAGES


@DatasetConfig.register_subclass("wsa_pretrain")
@dataclass
class WSAPretrainDatasetConfig(DatasetConfig):
    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32
    tokenize_state: bool = False
    max_prompt_length: int = 650
    mode: str = "train"
    chunk_size: int = 50
    use_fast_action_tokens: bool = True
    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                DeltaActionTransformFn(),
                ResizeImagesWithPadFn(
                    height=WSAPretrainDatasetConfig.height,
                    width=WSAPretrainDatasetConfig.width,
                ),
                RemapImageKeyTransformFn(),
                ExtractVideoFramesTransformFn(),
                NormalizeTransformFn(),
                ComposeFieldsTransform(),
                FASTWSAPretrainActionTokenizerTransformFn(),
                LoadActionTextFromJsonlTransformFn(),
                WSAPretrainChatProcessorTransformFn(),
                PadStateAndActionTransformFn(
                    max_state_dim=WSAPretrainDatasetConfig.max_state_dim,
                    max_action_dim=WSAPretrainDatasetConfig.max_action_dim,
                ),
                ReorderStateActionTransform(),
                UnifyWSAPretrainInputsTransformFn(
                    num_video_frames=WSAPretrainDatasetConfig.num_video_frames,
                    video_height=WSAPretrainDatasetConfig.video_height,
                    video_width=WSAPretrainDatasetConfig.video_width,
                ),
            ],
            outputs=[],
        )
    )

    def __post_init__(self):
        super().__post_init__()
        inputs = list(self.data_transforms.inputs)

        uses_pi05_image_aug = (
            self.image_transforms.enable
            and self.image_transforms.preset in {"pi05", "pi0.5", "pi05_style"}
        )
        inputs = [t for t in inputs if not isinstance(t, Pi05ImageAugmentFn)]
        if uses_pi05_image_aug:
            insert_idx = next(
                (idx + 1 for idx, transform in enumerate(inputs) if isinstance(transform, ResizeImagesWithPadFn)),
                0,
            )
            inputs.insert(insert_idx, Pi05ImageAugmentFn())

        has_delta = any(isinstance(t, DeltaActionTransformFn) for t in inputs)
        if self.action_mode == "delta" and not has_delta:
            logging.info("action_mode='delta' -> Adding DeltaActionTransformFn")
            inputs = [DeltaActionTransformFn(), *inputs]
        elif self.action_mode == "abs" and has_delta:
            logging.info("action_mode='abs' -> Removing DeltaActionTransformFn")
            inputs = [t for t in inputs if not isinstance(t, DeltaActionTransformFn)]

        processor = WSAPretrainChatProcessorTransformFn(
            tokenize_state=self.tokenize_state,
            max_state_dim=self.max_state_dim,
            max_length=self.max_prompt_length,
            use_fast_action_tokens=self.use_fast_action_tokens,
            mode=self.mode,
        )
        inputs = [t for t in inputs if not isinstance(t, WSAPretrainChatProcessorTransformFn)]
        insert_idx = next(
            (i for i, t in enumerate(inputs) if isinstance(t, PadStateAndActionTransformFn)),
            len(inputs),
        )
        inputs.insert(insert_idx, processor)

        has_fast = any(isinstance(t, FASTWSAPretrainActionTokenizerTransformFn) for t in inputs)
        if self.use_fast_action_tokens and not has_fast:
            inputs.insert(insert_idx, FASTWSAPretrainActionTokenizerTransformFn())
        elif not self.use_fast_action_tokens and has_fast:
            inputs = [t for t in inputs if not isinstance(t, FASTWSAPretrainActionTokenizerTransformFn)]

        for t in inputs:
            if isinstance(t, FASTWSAPretrainActionTokenizerTransformFn):
                t.chunk_size = self.chunk_size
                break

        self.data_transforms = replace(self.data_transforms, inputs=inputs)


@DataTransformFn.register_subclass("unify_wsa_pretrain_inputs")
@dataclass
class UnifyWSAPretrainInputsTransformFn(DataTransformFn):
    """Unify robot samples and always include video_frames for WAN.

    Always outputs observation.video_frames so that robot and VQA samples
    have identical keys and can be collated in the same batch.
    """

    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224

    def __call__(self, data: DataDict) -> DataDict:
        from lerobot.utils.constants import OBS_STATE, ACTION, OBS_STR
        from lerobot.policies.wsa_pretrain.transform_wsa_pretrain import LABEL_MODE_NONE
        import torch

        input_ids = data[f"{OBS_STR}.input_ids"]
        fast_token_mask = data.get(
            f"{OBS_STR}.fast_token_mask",
            torch.zeros_like(input_ids, dtype=torch.bool),
        )
        label_mode = data.get("label_mode", torch.tensor(LABEL_MODE_NONE, dtype=torch.long))

        video_key = "observation.video_frames"
        if video_key in data:
            video_frames = data[video_key]
        else:
            video_frames = torch.zeros(
                self.num_video_frames + 1, 3, self.video_height, self.video_width
            )

        return {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": input_ids,
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
            f"{OBS_STR}.fast_token_mask": fast_token_mask,
            "vqa_type": data["vqa_type"],
            "VQA.labels": data["VQA.labels"],
            "label_mode": label_mode,
            video_key: video_frames,
        }


@DataTransformFn.register_subclass("unify_wsa_pretrain_vqa_inputs")
@dataclass
class UnifyWSAPretrainVQAInputsTransformFn(DataTransformFn):
    """VQA unify transform that includes a dummy video_frames tensor.

    Ensures VQA samples have the same keys as robot samples so they can
    be collated in the same batch.
    """

    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224

    def __call__(self, data: DataDict) -> DataDict:
        from lerobot.utils.constants import OBS_STATE, ACTION, OBS_STR
        from lerobot.policies.wsa_pretrain.transform_wsa_pretrain import LABEL_MODE_TEXT
        import torch

        input_ids = data[f"{OBS_STR}.input_ids"]
        fast_token_mask = data.get(
            f"{OBS_STR}.fast_token_mask",
            torch.zeros_like(input_ids, dtype=torch.bool),
        )
        label_mode = data.get("label_mode", torch.tensor(LABEL_MODE_TEXT, dtype=torch.long))

        video_frames = torch.zeros(
            self.num_video_frames + 1, 3, self.video_height, self.video_width
        )

        return {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": input_ids,
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
            f"{OBS_STR}.fast_token_mask": fast_token_mask,
            "vqa_type": torch.tensor(1, dtype=torch.long),
            "VQA.labels": data["VQA.labels"],
            "label_mode": label_mode,
            "observation.video_frames": video_frames,
        }


@VQADatasetConfig.register_subclass("wsa_pretrain")
@dataclass
class WSAPretrainVQADatasetConfig(VQADatasetConfig):
    """VQA dataset config with dummy video_frames for shared batching."""

    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                # Note: If you resize the VQA images, the coordinates in the VQA Data should be scaled accordingly. 
                # Please pre-process the VQA data offline based on its format.
                ResizeVQAImagesWithPadFn(
                    height=WSAPretrainVQADatasetConfig.height,
                    width=WSAPretrainVQADatasetConfig.width,
                ),
                PadStateAndActionTransformFn(
                    max_state_dim=WSAPretrainVQADatasetConfig.max_state_dim,
                    max_action_dim=WSAPretrainVQADatasetConfig.max_action_dim,
                ),
                WSAPretrainVQAProcessorTransformFn(),
                UnifyWSAPretrainVQAInputsTransformFn(
                    num_video_frames=WSAPretrainVQADatasetConfig.num_video_frames,
                    video_height=WSAPretrainVQADatasetConfig.video_height,
                    video_width=WSAPretrainVQADatasetConfig.video_width,
                ),
            ],
            outputs=[],
        )
    )

    def __post_init__(self):
        inputs = list(self.data_transforms.inputs)
        processor_type = WSAPretrainVQAProcessorTransformFn
        inputs = [t for t in inputs if not isinstance(t, processor_type)]
        insert_idx = next(
            (
                i
                for i, t in enumerate(inputs)
                if isinstance(t, UnifyWSAPretrainVQAInputsTransformFn)
            ),
            len(inputs),
        )
        inputs.insert(insert_idx, WSAPretrainVQAProcessorTransformFn())
        self.data_transforms = replace(self.data_transforms, inputs=inputs)


@PreTrainedConfig.register_subclass("wsa_pretrain")
@dataclass
class WSAPretrainConfig(PreTrainedConfig):
    # VLM model selection - supports Qwen3.5-2B/4B/8B
    vlm_model_name_or_path: str = "Qwen/Qwen3.5-2B"

    # Action expert customization
    action_expert_hidden_size: int | None = 1024
    action_expert_intermediate_size: int | None = 3072

    dtype: str = "bfloat16"

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    image_resolution: tuple[int, int] = (224, 224)
    empty_cameras: int = 0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Training settings
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None

    # Optimizer settings. VLM is pretrained, while the action branch is mostly
    # initialized from scratch, so they use independent learning-rate curves.
    vlm_lr: float = 1.0e-5
    vlm_decay_lr: float = 1.0e-6
    vlm_warmup_steps: int = 2_000
    vlm_decay_steps: int = 100_000

    action_lr: float = 1.0e-4
    action_decay_lr: float = 1.0e-5
    action_warmup_steps: int = 1_000
    action_decay_steps: int = 100_000

    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    tokenizer_max_length: int = 48

    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    # VQA configurations
    # WSA pretraining defaults to flow-matching action supervision only.
    enable_vqa_loss: bool = False
    lambda_vqa: float = 1.0
    tokenize_state: bool = True

    # FAST action tokens
    action_token_min: int = 248077
    action_token_max: int = 250124

    # Knowledge insulation
    knowledge_insulation: bool = False
    block_action_attend_fast_tokens: bool = True

    inference_action_type: str = "fm"  # "fm" for flow matching, "fast" for fast-token supervision
    inference_backend: str = "standard"  # "standard" or "optimized"

    # Use SDPA (scaled_dot_product_attention) instead of eager attention in training
    use_sdpa: bool = False

    num_learnable_tokens: int = 50

    wan_checkpoint_path: str = f"{HF_HOME}/hub/Wan2.2-TI2V-5B"
    wan_config_path: str = f"{HF_HOME}/hub/Wan2.2-TI2V-5B"
    vae_path: str = f"{HF_HOME}/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    video_precision: str = "bfloat16"

    freeze_wan_dit: bool = True
    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224
    video_loss_weight: float = 1.0
    video_loss_only: bool = False
    action_loss_only: bool = True
    freeze_learnable_tokens: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.lambda_vqa < 0:
            raise ValueError(f"lambda_vqa must be >= 0, got {self.lambda_vqa}")
        if self.action_token_min > self.action_token_max:
            raise ValueError(
                f"action_token_min ({self.action_token_min}) must be <= action_token_max ({self.action_token_max})"
            )
        if self.inference_backend not in {"standard", "optimized"}:
            raise ValueError(
                "inference_backend must be either 'standard' or 'optimized', "
                f"got {self.inference_backend!r}"
            )
        if self.inference_backend == "optimized" and not self.action_loss_only:
            raise ValueError("inference_backend='optimized' requires action_loss_only=True")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features["action"] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            # Parameter-group learning rates returned by get_optim_params()
            # override this optimizer-level fallback.
            lr=self.action_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return MultiGroupCosineDecayWithWarmupSchedulerConfig(
            group_schedules={
                "vlm": {
                    "peak_lr": self.vlm_lr,
                    "decay_lr": self.vlm_decay_lr,
                    "num_warmup_steps": self.vlm_warmup_steps,
                    "num_decay_steps": self.vlm_decay_steps,
                },
                "action": {
                    "peak_lr": self.action_lr,
                    "decay_lr": self.action_decay_lr,
                    "num_warmup_steps": self.action_warmup_steps,
                    "num_decay_steps": self.action_decay_steps,
                },
            }
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def image_delta_indices(self) -> list | None:
        n = self.num_video_frames + 1
        return [self.chunk_size * i // (n - 1) for i in range(n)]
