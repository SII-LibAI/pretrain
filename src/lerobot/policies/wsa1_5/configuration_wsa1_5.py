"""Configuration and transform pipelines for the WSA1.5 policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from lerobot.configs.default import DatasetConfig, VQADatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.wsa1_5 import (
    CANONICAL_DIM,
    MOTION_MASK_ID_KEYS,
    MOTION_MASK_KEYS,
    CanonicalStateActionTransformFn,
    LoadWSA15PhysicalAnnotationsFn,
    PrepareWSA15MotionTargetsFn,
    SelectWSA15CurrentObservationFn,
    UnifyWSA15InputsFn,
    UnifyWSA15VQAInputsFn,
)
from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import (
    InternVLAA15Config,
    InternVLAA15DatasetConfig,
    InternVLAA15VQADatasetConfig,
)
from lerobot.policies.wsa1_5.transform_wsa1_5 import (
    WSA15ChatProcessorTransformFn,
    WSA15VQAProcessorTransformFn,
)
from lerobot.transforms.core import (
    ComposeFieldsTransform,
    LoadActionTextFromJsonlTransformFn,
    RemapImageKeyTransformFn,
    ResizeImagesWithPadFn,
    ResizeVQAImagesWithPadFn,
    TransformGroup,
)


def _robot_transforms() -> TransformGroup:
    return TransformGroup(
        inputs=[
            PrepareWSA15MotionTargetsFn(),
            ResizeImagesWithPadFn(height=224, width=224),
            RemapImageKeyTransformFn(),
            SelectWSA15CurrentObservationFn(),
            ComposeFieldsTransform(),
            CanonicalStateActionTransformFn(),
            LoadActionTextFromJsonlTransformFn(),
            LoadWSA15PhysicalAnnotationsFn(),
            WSA15ChatProcessorTransformFn(),
            UnifyWSA15InputsFn(),
        ],
        outputs=[],
    )


@DatasetConfig.register_subclass("wsa1_5")
@dataclass
class WSA15DatasetConfig(InternVLAA15DatasetConfig):
    """LeRobot v3 config with optional sidecar and motion supervision."""

    max_state_dim: int = CANONICAL_DIM
    max_action_dim: int = CANONICAL_DIM
    tokenize_state: bool = False
    use_fast_action_tokens: bool = False
    num_subtask_tokens: int = 16
    # Two 224-wide supervision images are packed side by side.
    video_width: int = 448
    data_transforms: TransformGroup = field(default_factory=_robot_transforms)
    supplemental_video_keys: tuple[str, ...] = MOTION_MASK_KEYS + MOTION_MASK_ID_KEYS

    def __post_init__(self) -> None:
        # Do not call InternVLAA15DatasetConfig.__post_init__: it intentionally
        # replaces subclasses of its processor with the base A1.5 processor.
        DatasetConfig.__post_init__(self)
        if self.max_state_dim != CANONICAL_DIM or self.max_action_dim != CANONICAL_DIM:
            raise ValueError(f"WSA1.5 uses a fixed {CANONICAL_DIM}D semantic state/action representation")
        if self.tokenize_state:
            raise ValueError(
                "WSA1.5 sends semantic state tokens to the action expert; tokenize_state must be false"
            )
        if self.use_fast_action_tokens:
            raise ValueError(
                "WSA1.5 predicts actions with flow matching; FAST action-token labels are disabled"
            )
        if self.action_mode == "delta":
            logging.warning(
                "WSA1.5 ignores dataset.action_mode='delta': joint targets remain absolute and "
                "EEF targets are converted to relative deltas by the semantic codec"
            )
        inputs = list(self.data_transforms.inputs)

        processor = WSA15ChatProcessorTransformFn(
            max_length=self.max_prompt_length,
            mode=self.mode,
            num_subtask_tokens=self.num_subtask_tokens,
        )
        inputs = [
            transform for transform in inputs if not isinstance(transform, WSA15ChatProcessorTransformFn)
        ]
        processor_index = next(
            (index for index, transform in enumerate(inputs) if isinstance(transform, UnifyWSA15InputsFn)),
            len(inputs),
        )
        inputs.insert(processor_index, processor)

        for transform in inputs:
            if isinstance(transform, (PrepareWSA15MotionTargetsFn, UnifyWSA15InputsFn)):
                transform.num_future_frames = self.num_video_frames
                transform.height = self.video_height
                transform.width = self.video_width
                if isinstance(transform, UnifyWSA15InputsFn):
                    transform.num_subtask_tokens = self.num_subtask_tokens

        self.data_transforms = replace(self.data_transforms, inputs=inputs)
        logging.info("Configured WSA1.5 dataset with optional subtask/world supervision")


def _vqa_transforms() -> TransformGroup:
    return TransformGroup(
        inputs=[
            ResizeVQAImagesWithPadFn(height=224, width=224),
            WSA15VQAProcessorTransformFn(),
            UnifyWSA15VQAInputsFn(),
        ],
        outputs=[],
    )


@VQADatasetConfig.register_subclass("wsa1_5")
@dataclass
class WSA15VQADatasetConfig(InternVLAA15VQADatasetConfig):
    max_state_dim: int = CANONICAL_DIM
    max_action_dim: int = CANONICAL_DIM
    video_width: int = 448
    num_subtask_tokens: int = 16
    data_transforms: TransformGroup = field(default_factory=_vqa_transforms)

    def __post_init__(self) -> None:
        inputs = list(self.data_transforms.inputs)
        inputs = [
            transform for transform in inputs if not isinstance(transform, WSA15VQAProcessorTransformFn)
        ]
        unify_index = next(
            (index for index, transform in enumerate(inputs) if isinstance(transform, UnifyWSA15VQAInputsFn)),
            len(inputs),
        )
        inputs.insert(unify_index, WSA15VQAProcessorTransformFn())
        for transform in inputs:
            if isinstance(transform, UnifyWSA15VQAInputsFn):
                transform.num_future_frames = self.num_video_frames
                transform.height = self.video_height
                transform.width = self.video_width
                transform.num_subtask_tokens = self.num_subtask_tokens
        self.data_transforms = replace(self.data_transforms, inputs=inputs)


@PreTrainedConfig.register_subclass("wsa1_5")
@dataclass
class WSA15Config(InternVLAA15Config):
    """WSA1.5: Qwen3.5 context + planning expert + action expert."""

    max_state_dim: int = CANONICAL_DIM
    max_action_dim: int = CANONICAL_DIM
    tokenize_state: bool = False
    video_width: int = 448
    num_subtask_tokens: int = 16
    # One 8x8 2D planning grid and one 12x12 future-3D query grid.  The latter
    # follows WSA-L's default per-view query count; this dataset has one view.
    num_future_2d_tokens: int = 64
    num_future_3d_tokens: int = 144
    future_query_init_std: float = 0.02
    subtask_loss_weight: float = 1.0
    supplemental_video_keys: tuple[str, ...] = MOTION_MASK_KEYS + MOTION_MASK_ID_KEYS
    motion_canvas_layout: str = "motion_mask|motion_mask_ids"

    def __post_init__(self) -> None:
        self.num_learnable_tokens = self.num_future_2d_tokens + self.num_future_3d_tokens
        super().__post_init__()
        if self.max_state_dim != CANONICAL_DIM or self.max_action_dim != CANONICAL_DIM:
            raise ValueError(f"WSA1.5 policy max_state_dim/max_action_dim must both be {CANONICAL_DIM}")
        if self.num_subtask_tokens < 1:
            raise ValueError("num_subtask_tokens must be positive")
        if self.num_future_2d_tokens < 1 or self.num_future_3d_tokens < 1:
            raise ValueError("future 2D/3D query counts must be positive")
        if self.future_query_init_std <= 0:
            raise ValueError("future_query_init_std must be positive")
        if self.subtask_loss_weight < 0:
            raise ValueError("subtask_loss_weight must be non-negative")
        if self.video_width % 2:
            raise ValueError("video_width must be even for the WSA1.5 motion canvas")
        if self.tokenize_state:
            raise ValueError("WSA1.5 requires tokenize_state=False")
        if self.inference_action_type != "fm":
            raise ValueError("WSA1.5 action inference is flow matching; inference_action_type must be 'fm'")
        if self.inference_backend == "optimized":
            raise ValueError("WSA1.5 three-expert inference does not support inference_backend='optimized'")
        if self.compile_model:
            raise ValueError("WSA1.5 does not yet support compile_model=True for three-expert execution")


# Keep old local launch files/checkpoints usable without restoring the removed
# wsa_pretrain implementation.  New runs should use type=wsa1_5.
@DatasetConfig.register_subclass("wsa_pretrain")
@dataclass
class WSAPretrainDatasetCompatibilityConfig(WSA15DatasetConfig):
    pass


@VQADatasetConfig.register_subclass("wsa_pretrain")
@dataclass
class WSAPretrainVQADatasetCompatibilityConfig(WSA15VQADatasetConfig):
    pass


@PreTrainedConfig.register_subclass("wsa_pretrain")
@dataclass
class WSAPretrainCompatibilityConfig(WSA15Config):
    pass


__all__ = [
    "WSA15Config",
    "WSA15DatasetConfig",
    "WSA15VQADatasetConfig",
    "WSAPretrainCompatibilityConfig",
]
