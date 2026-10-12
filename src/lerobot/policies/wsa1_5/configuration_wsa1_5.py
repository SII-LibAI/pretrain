"""Configuration and transform pipelines for the WSA1.5 policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import ClassVar

from lerobot.configs.default import DatasetConfig, VQADatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.synthetic_embodiment import HUMAN_EEF_EMBODIMENT
from lerobot.datasets.wsa1_5 import (
    CANONICAL_DIM,
    DEFAULT_HUMAN_ACTION_FILE,
    FUTURE_2D_SOURCE_KEY,
    FUTURE_3D_SOURCE_KEY,
    WSA15_DATAFLOW_NAME,
    PrepareWSA15MotionTargetsFn,
    UnifyWSA15InputsFn,
    UnifyWSA15VQAInputsFn,
    build_wsa15_data_transforms,
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
    DeltaActionTransformFn,
    ResizeVQAImagesWithPadFn,
    TransformGroup,
)


def _robot_transforms() -> TransformGroup:
    return TransformGroup(
        inputs=[
            *build_wsa15_data_transforms(),
            WSA15ChatProcessorTransformFn(),
            UnifyWSA15InputsFn(),
        ],
        outputs=[],
    )


def build_wsa15_robot_transforms(
    *,
    action_mode: str,
    max_prompt_length: int,
    mode: str,
    chunk_size: int,
    num_video_frames: int,
    video_height: int,
    video_width: int,
    num_subtask_tokens: int,
    transforms: TransformGroup | None = None,
) -> TransformGroup:
    """Configure one WSA batch pipeline for regular or synthetic datasets."""

    group = transforms or _robot_transforms()
    inputs = list(group.inputs)
    delta_transforms = [
        transform for transform in inputs if isinstance(transform, DeltaActionTransformFn)
    ]
    inputs = [transform for transform in inputs if not isinstance(transform, DeltaActionTransformFn)]
    if action_mode == "delta":
        inputs.insert(0, delta_transforms[0] if delta_transforms else DeltaActionTransformFn())

    processor = WSA15ChatProcessorTransformFn(
        max_length=max_prompt_length,
        mode=mode,
        num_subtask_tokens=num_subtask_tokens,
    )
    inputs = [
        transform for transform in inputs if not isinstance(transform, WSA15ChatProcessorTransformFn)
    ]
    processor_index = next(
        (index for index, transform in enumerate(inputs) if isinstance(transform, UnifyWSA15InputsFn)),
        len(inputs),
    )
    inputs.insert(processor_index, processor)

    if num_video_frames < 1:
        raise ValueError("WSA1.5 num_video_frames must be positive")
    for transform in inputs:
        if isinstance(transform, (PrepareWSA15MotionTargetsFn, UnifyWSA15InputsFn)):
            transform.num_future_frames = num_video_frames
            transform.height = video_height
            transform.width = video_width
            if isinstance(transform, UnifyWSA15InputsFn):
                transform.num_subtask_tokens = num_subtask_tokens
    return replace(group, inputs=inputs)


@DatasetConfig.register_subclass("wsa1_5")
@dataclass
class WSA15DatasetConfig(InternVLAA15DatasetConfig):
    """Mixed robot/human LeRobot v3 data-flow config for WSA1.5."""

    dataflow: ClassVar[str] = WSA15_DATAFLOW_NAME
    max_state_dim: int = CANONICAL_DIM
    max_action_dim: int = CANONICAL_DIM
    tokenize_state: bool = False
    use_fast_action_tokens: bool = False
    num_subtask_tokens: int = 16
    video_width: int = 224
    human_action_file: str = DEFAULT_HUMAN_ACTION_FILE
    human_stats_dir: str | None = None
    human_embodiments: list[str] = field(default_factory=lambda: [HUMAN_EEF_EMBODIMENT])
    human_annotation_row_group_cache_size: int = 0
    data_transforms: TransformGroup = field(default_factory=_robot_transforms)
    supplemental_video_keys: tuple[str, ...] = (FUTURE_2D_SOURCE_KEY, FUTURE_3D_SOURCE_KEY)

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
        if self.streaming:
            raise ValueError("WSA1.5 human action annotations do not support streaming")
        if self.human_annotation_row_group_cache_size < 0:
            raise ValueError("human_annotation_row_group_cache_size must be non-negative")
        if len(set(self.human_embodiments)) != len(self.human_embodiments):
            raise ValueError("human_embodiments contains duplicate names")
        self.data_transforms = build_wsa15_robot_transforms(
            action_mode=self.action_mode,
            max_prompt_length=self.max_prompt_length,
            mode=self.mode,
            chunk_size=self.chunk_size,
            num_video_frames=self.num_video_frames,
            video_height=self.video_height,
            video_width=self.video_width,
            num_subtask_tokens=self.num_subtask_tokens,
            transforms=self.data_transforms,
        )
        logging.info("Configured WSA1.5 mixed robot/human data flow")


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
    video_width: int = 224
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
    video_width: int = 224
    num_subtask_tokens: int = 16
    # One 8x8 2D planning grid and one 12x12 future-3D query grid.  The latter
    # follows WSA-L's default per-view query count; this dataset has one view.
    num_future_2d_tokens: int = 64
    num_future_3d_tokens: int = 144
    future_query_init_std: float = 0.02
    subtask_loss_weight: float = 1.0
    supplemental_video_keys: tuple[str, ...] = (FUTURE_2D_SOURCE_KEY, FUTURE_3D_SOURCE_KEY)

    def __post_init__(self) -> None:
        self.num_learnable_tokens = self.num_future_2d_tokens + self.num_future_3d_tokens
        super().__post_init__()
        if self.max_state_dim != CANONICAL_DIM or self.max_action_dim != CANONICAL_DIM:
            raise ValueError(f"WSA1.5 policy max_state_dim/max_action_dim must both be {CANONICAL_DIM}")
        if self.num_subtask_tokens < 1:
            raise ValueError("num_subtask_tokens must be positive")
        if self.num_future_2d_tokens < 1 or self.num_future_3d_tokens < 1:
            raise ValueError("future 2D/3D query counts must be positive")
        if self.num_future_3d_tokens < self.num_video_frames:
            raise ValueError("num_future_3d_tokens must be at least num_video_frames")
        if self.future_query_init_std <= 0:
            raise ValueError("future_query_init_std must be positive")
        if self.subtask_loss_weight < 0:
            raise ValueError("subtask_loss_weight must be non-negative")
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
    "build_wsa15_robot_transforms",
]
