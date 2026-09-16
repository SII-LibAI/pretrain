"""Configuration for the InternVLA synthetic-embodiment dataset adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from lerobot.configs.default import DatasetConfig
from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import (
    InternVLAA15DatasetConfig,
)


@DatasetConfig.register_subclass("synthetic_embodiment")
@dataclass
class SyntheticEmbodimentDatasetConfig(InternVLAA15DatasetConfig):
    """Train on selected frame-aligned synthetic embodiments."""

    action_mode: str = "delta"
    streaming: bool = False
    use_external_stats: bool = False
    use_fast_action_tokens: bool = False
    embodiments: list[str] = field(default_factory=list)
    stats_dir: str | None = None

    annotation_file: ClassVar[str] = "meta/annotations/action.parquet"
    embodiment_schema_prefix: ClassVar[str] = "egodex_synthetic_"
    default_stats_directory: ClassVar[str] = "synthetic_embodiment_stats"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.action_mode != "delta":
            raise ValueError("synthetic_embodiment requires action_mode='delta'")
        if self.streaming:
            raise ValueError("synthetic_embodiment does not support streaming")
        if self.use_external_stats:
            raise ValueError("synthetic_embodiment uses its own per-embodiment stats directory")
        if self.use_fast_action_tokens:
            raise ValueError("synthetic_embodiment currently uses the continuous action expert")
        if len(set(self.embodiments)) != len(self.embodiments):
            raise ValueError("embodiments contains duplicate names")
