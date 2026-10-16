"""LeRobot V3 configuration for world-model pretraining without actions."""
from dataclasses import dataclass, field

from lerobot.configs.default import DatasetConfig


@DatasetConfig.register_subclass("wsa1_5")
@dataclass
class WSA1_5DatasetConfig(DatasetConfig):
    # Canonical schema names, in WSA top / bottom-left / bottom-right order.
    view_names: list[str] = field(default_factory=lambda: [
        "observation.images.image0", "observation.images.image1", "observation.images.image2"
    ])
    # Optional source keys in precisely the same order; empty entries mean absent.
    view_keys: list[str] = field(default_factory=list)
    future_frames: int = 32
    frame_stride: int = 1
    rgb_height: int = 384
    rgb_width: int = 320
    motion_height: int = 192
    motion_width: int = 160
    motion_key: str = "observation.motion"
    sidecar_file: str = "meta/sidecar.json"
    text_embedding_dir: str = "meta/wsa1_5_text_embeddings"
    use_imagenet_stats: bool = False  # Normalization is explicit in the adapter.
    use_external_stats: bool = False
    motion_log_every: int = 1000

    def __post_init__(self):
        super().__post_init__()
        if self.streaming:
            raise ValueError("WSA endpoint sampling requires map-style LeRobot data")
        if self.use_imagenet_stats or self.use_external_stats or self.image_transforms.enable:
            raise ValueError("WSA uses explicit ImageNet normalization; external stats/image transforms are unsupported")
        if any(group.inputs or group.outputs for group in [self.repack_transforms, self.data_transforms, self.model_transforms]):
            raise ValueError("WSA adapter does not support custom transform groups; refusing to ignore them")
        for name in ["future_frames", "frame_stride", "rgb_height", "rgb_width", "motion_height", "motion_width"]:
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.motion_log_every) is not int or self.motion_log_every < 0:
            raise ValueError("motion_log_every must be a non-negative integer")
        if len(self.view_names) != 3 or len(set(self.view_names)) != 3:
            raise ValueError("view_names must specify three distinct ordered schema views")
        if self.view_keys and len(self.view_keys) != 3:
            raise ValueError("view_keys must specify exactly three ordered source keys")
        present = [key for key in self.view_keys if key]
        if len(set(present)) != len(present):
            raise ValueError("A source RGB view cannot occupy two slots")
        if self.future_frames != 32 or self.frame_stride <= 0:
            raise ValueError("First version requires 32 future frames and a positive frame_stride")
        if self.rgb_height % 3 or self.rgb_width % 2:
            raise ValueError("WSA layout requires height divisible by 3 and even width")
        if min(self.rgb_height, self.rgb_width, self.motion_height, self.motion_width) <= 0:
            raise ValueError("Image sizes must be positive")
