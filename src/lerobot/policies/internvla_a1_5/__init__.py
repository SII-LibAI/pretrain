from .configuration_internvla_a1_5 import (
    InternVLAA15Config,
    InternVLAA15DatasetConfig,
    InternVLAA15VQADatasetConfig,
)
def __getattr__(name):
    if name == "InternVLAA15Policy":
        from .modeling_internvla_a1_5 import InternVLAA15Policy
        return InternVLAA15Policy
    if name == "InternVLAA15Optimized":
        from .modeling_internvla_a1_5_optimized import InternVLAA15Optimized
        return InternVLAA15Optimized
    raise AttributeError(name)

__all__ = [
    "InternVLAA15Config",
    "InternVLAA15DatasetConfig",
    "InternVLAA15VQADatasetConfig",
    "InternVLAA15Optimized",
    "InternVLAA15Policy",
]
