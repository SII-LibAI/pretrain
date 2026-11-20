"""Source-specific retarget layouts and offline statistics, without model loading."""
from dataclasses import replace
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.dataset_schemas import get_registry, get_schema
from lerobot.datasets.retarget_actions import RetargetActionSource
from lerobot.datasets.synthetic_embodiment import embodiment_schema_prefix
from util_scripts.compute_synthetic_embodiment_stats import compute_embodiment_stats, write_embodiment_stats


EGO_SOURCES = {
    "egopro": {"observation.images.head_left": "observation.images.image0",
               "observation.images.left_wrist": "observation.images.image1",
               "observation.images.right_wrist": "observation.images.image2"},
    "egostand": {"observation.images.head_left": "observation.images.image0"},
}
UMI_SOURCES = {
    "fastumi_single": {"observation.images.camera_rgb_image": "observation.images.image0"},
    "fastumi_dual": {"observation.images.left_camera_rgb_image": "observation.images.image0",
                     "observation.images.right_camera_rgb_image": "observation.images.image1"},
    "Hunyuan_eef": {"observation.images.cam_high": "observation.images.image0",
                    "observation.images.cam_left_wrist": "observation.images.image1",
                    "observation.images.cam_right_wrist": "observation.images.image2"},
    "genrobot_das_bimanual": {"observation.images.left": "observation.images.image0",
                              "observation.images.right": "observation.images.image1"},
}
SOURCE_IMAGES = {**EGO_SOURCES, **UMI_SOURCES}
ARM_BODIES = ("piper", "franka", "arx_l5", "xarm7", "rm75", "ur7e")
DEX_BODIES = ("piper_inspire", "ur7e_wuji")


@pytest.mark.parametrize("source,images", SOURCE_IMAGES.items())
def test_source_schemas_preserve_egodex_layouts(source, images):
    registry = get_registry()
    assert source in registry.list_available()
    assert get_schema(source).image_mapping == images
    prefix = embodiment_schema_prefix(source)
    bodies = (*ARM_BODIES, *DEX_BODIES) if source in EGO_SOURCES else ARM_BODIES
    for body in ("human_eef", *bodies):
        original_name = f"dexhand_{body}" if body in DEX_BODIES else body
        original = get_schema(f"egodex_synthetic_{original_name}")
        actual = get_schema(f"{prefix}{body}")
        assert actual.robot_type in registry.list_available()
        assert actual == replace(original, robot_type=actual.robot_type, image_mapping=images)
    if source in UMI_SOURCES:
        assert all(f"{prefix}{body}" not in registry.list_available() for body in DEX_BODIES)


@pytest.mark.parametrize("source", SOURCE_IMAGES)
def test_source_stats_and_training_use_full_names(tmp_path, source):
    root = tmp_path / source
    annotation_dir = root / "meta/annotations"
    annotation_dir.mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"robot_type": source}), encoding="utf-8")
    frames = np.arange(6)
    columns = {"index": frames, "episode_index": np.zeros(6, dtype=np.int64), "frame_index": frames}
    bodies = (*ARM_BODIES, *DEX_BODIES) if source in EGO_SOURCES else ARM_BODIES
    for body in bodies:
        dim = get_schema(f"{embodiment_schema_prefix(source)}{body}").action_dim
        columns[f"action.{body}"] = (frames[:, None] + np.arange(dim)[None]).tolist()
        columns[f"action.{body}_valid"] = [True] * 6
    pq.write_table(pa.table(columns), annotation_dir / "action.parquet")
    stats, _ = compute_embodiment_stats([root], chunk_size=2, action_space="unified72")
    prefix = embodiment_schema_prefix(source)
    assert set(stats) == {f"{prefix}{body}" for body in bodies}
    write_embodiment_stats(tmp_path / "norm", stats)
    selected = f"{prefix}piper"
    training = RetargetActionSource(root, 2, embodiments=[selected], stats_dir=tmp_path / "norm")
    assert training.names == (selected,)
    state, future = training.read_window(0, 0)
    assert state.shape == (16,) and future.shape == (2, 16)
    for body in bodies:
        schema = get_schema(f"{prefix}{body}").for_action_space("unified72")
        slots = [slot for _, _, start, stop in schema.action_reorder for slot in range(start, stop)]
        assert len(slots) == len(set(slots)) and all(0 <= slot < 72 for slot in slots)


@pytest.mark.parametrize("source", SOURCE_IMAGES)
def test_source_rejects_wrong_annotation_dimension(tmp_path, source):
    annotation_dir = tmp_path / "meta/annotations"
    annotation_dir.mkdir(parents=True)
    (tmp_path / "meta/info.json").write_text(json.dumps({"robot_type": source}), encoding="utf-8")
    pq.write_table(pa.table({"index": [0, 1], "episode_index": [0, 0], "frame_index": [0, 1],
                            "action.piper": [[0.] * 8] * 2, "action.piper_valid": [True, True]}),
                   annotation_dir / "action.parquet")
    with pytest.raises(ValueError, match="annotation has 8"):
        RetargetActionSource(tmp_path, 1)
