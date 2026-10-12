from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.dataset_schemas import DatasetSchema
from lerobot.datasets.wsa1_5 import (
    ACTION_VALID_MASK_KEY,
    FUTURE_2D_TARGET_KEY,
    FUTURE_2D_VALID_KEY,
    FUTURE_3D_TARGET_KEY,
    FUTURE_3D_VALID_KEY,
    SUBTASK_TEXT_KEY,
    SUBTASK_VALID_KEY,
    CanonicalStateActionTransformFn,
    LoadWSA15PhysicalAnnotationsFn,
    PrepareWSA15MotionTargetsFn,
    decode_canonical_actions,
    resolve_wsa15_delta_timestamps,
    resolve_wsa15_video_key_mapping,
    wsa15_future_indices,
    wsa15_source_kind,
)
from lerobot.transforms.core import DeltaActionTransformFn
from lerobot.utils.constants import ACTION, OBS_STATE
from util_scripts.inspect_wsa1_5_dataflow import (
    INSPECTION_KEY,
    InspectWSA15NormalizationFn,
    InspectWSA15StateActionFn,
)

# Keep this pure attention-mask unit test runnable in component-only
# environments that do not install the Qwen3.5 Transformers extension.  A
# normal policy import still uses lerobot.policies.wsa1_5.attention.
_ATTENTION_MODULE = run_path(str(Path(__file__).parents[1] / "src/lerobot/policies/wsa1_5/attention.py"))
build_wsa_attention_mask = _ATTENTION_MODULE["build_wsa_attention_mask"]


def _inspect_normalized(state, action, state_stats, action_stats, inspector):
    normalizer = InspectWSA15NormalizationFn(
        selected_keys=[OBS_STATE, ACTION],
        norm_stats={
            key: {name: np.asarray(values) for name, values in stats.items()}
            for key, stats in ((OBS_STATE, state_stats), (ACTION, action_stats))
        },
        mapping={OBS_STATE: [OBS_STATE], ACTION: [ACTION]},
    )
    return inspector(normalizer({OBS_STATE: state, ACTION: action}))


def test_inspector_checks_nonidentity_stats_and_semantics_after_reorder() -> None:
    inspector = InspectWSA15StateActionFn(
        action_layout=[{"name": "left_arm", "size": 2}],
        state_reorder=[[1, 2, 0, 1], [0, 1, 1, 2]],
        action_reorder=[[1, 2, 0, 1], [0, 1, 1, 2]],
    )
    sample = _inspect_normalized(
        torch.tensor([6.0, 12.0]), torch.tensor([[5.0, 9.0], [9.0, 15.0]]),
        {"mean": [2.0, 4.0], "std": [2.0, 4.0]},
        {"mean": [1.0, 3.0], "std": [2.0, 3.0]}, inspector,
    )
    report = sample[INSPECTION_KEY]
    assert report["state"]["valid_indices"] == [0, 1]
    assert report["state"]["dimensions"][0]["source_index"] == 1
    assert report["state"]["dimensions"][0]["semantic"] == "left_arm.0"
    assert report["state"]["dimensions"][0]["value"] == pytest.approx(2.0)
    assert report["action"]["dimensions"][0]["value_t0"] == pytest.approx(2.0)
    assert report["state"]["normalization"]["verified"]
    assert report["action"]["normalization"]["verified"]
    assert report["action"]["normalization"]["checked_values"] == 4


@pytest.mark.parametrize("stats", [None, {}, {OBS_STATE: {"min": np.zeros(18), "max": np.ones(18)}}])
def test_wsa_rejects_missing_mean_std_instead_of_fabricating_stats(stats) -> None:
    dataset = SimpleNamespace(
        meta=SimpleNamespace(robot_type="egodex_synthetic_franka", stats=stats)
    )
    with pytest.raises(ValueError, match="missing"):
        CanonicalStateActionTransformFn().hydrate(dataset)
    with pytest.raises(ValueError, match="missing"):
        InspectWSA15NormalizationFn().hydrate(dataset)


def test_inspector_detects_incorrect_normalized_action(monkeypatch) -> None:
    original = CanonicalStateActionTransformFn.__call__

    def corrupted_codec(self, data):
        output = original(self, data)
        output[ACTION][-1, 0] += 1.0
        return output

    monkeypatch.setattr(CanonicalStateActionTransformFn, "__call__", corrupted_codec)
    stats = {"mean": [1.0], "std": [2.0]}
    sample = _inspect_normalized(
        torch.tensor([5.0]), torch.tensor([[7.0], [9.0]]), stats, stats,
        InspectWSA15StateActionFn(action_layout=[{"name": "left_arm", "size": 1}]),
    )
    report = sample[INSPECTION_KEY]["action"]["normalization"]
    assert not report["formula_matches"]
    assert report["max_abs_error"] == 1.0


def test_wsa_dataflow_query_plan_separates_robot_and_human_actions() -> None:
    config = SimpleNamespace(chunk_size=8, num_video_frames=2)
    metadata = SimpleNamespace(
        robot_type="egodex_v",
        fps=10,
        features={
            "observation.state": {"dtype": "float32"},
            "action": {"dtype": "float32"},
            "observation.image": {"dtype": "video"},
            "observation.mask": {"dtype": "video"},
            "observation.motion": {"dtype": "video"},
        },
    )

    assert wsa15_source_kind("egodex_v") == "human"
    assert wsa15_source_kind("franka") == "robot"
    assert wsa15_future_indices(8, 2) == [4, 8]

    robot = resolve_wsa15_delta_timestamps(config, metadata, human_source=False)
    assert robot["action"] == [index / 10 for index in range(8)]
    assert robot["observation.image"] == [0.0]
    assert robot["observation.mask"] == [0.4, 0.8]
    assert robot["observation.motion"] == [0.4, 0.8]

    human = resolve_wsa15_delta_timestamps(config, metadata, human_source=True)
    assert "action" not in human
    assert human["observation.image"] == [0.0]
    assert human["observation.mask"] == [0.4, 0.8]
    assert human["observation.motion"] == [0.4, 0.8]


def test_wsa_supplemental_video_storage_alias_is_explicit() -> None:
    metadata = SimpleNamespace(
        features={"observation.mask": {"dtype": "video"}, "observation.motion": {"dtype": "video"}},
        episodes=[{"videos/mask/chunk_index": 0, "videos/motion/chunk_index": 0}],
    )
    assert resolve_wsa15_video_key_mapping(metadata) == {
        "observation.mask": "mask", "observation.motion": "motion",
    }

    malformed = SimpleNamespace(
        features={"observation.mask": {"dtype": "video"}},
        episodes=[{}],
    )
    with pytest.raises(KeyError, match="episode metadata"):
        resolve_wsa15_video_key_mapping(malformed)


@pytest.mark.parametrize("has_mask,has_motion", [(True, False), (False, True), (True, True), (False, False)])
def test_optional_motion_targets_have_independent_valid_masks(has_mask, has_motion) -> None:
    transform = PrepareWSA15MotionTargetsFn(num_future_frames=2, height=8, width=8)
    data = {}
    if has_mask:
        data["observation.mask"] = torch.tensor([[0.0, 1.0], [1.0, 0.0]])[None, None].expand(2, 3, -1, -1)
    if has_motion:
        data["observation.motion"] = torch.rand(2, 3, 4, 4)
    output = transform(data)

    assert output[FUTURE_2D_TARGET_KEY].shape == (2, 3, 8, 8)
    assert bool(output[FUTURE_2D_VALID_KEY]) == has_mask
    if has_mask:
        assert set(output[FUTURE_2D_TARGET_KEY].unique().tolist()) == {-1.0, 1.0}
    else:
        assert torch.all(output[FUTURE_2D_TARGET_KEY] == -1)
    assert output[FUTURE_3D_TARGET_KEY].shape == (2, 3, 8, 8)
    assert bool(output[FUTURE_3D_VALID_KEY]) == has_motion
    if not has_motion:
        assert torch.all(output[FUTURE_3D_TARGET_KEY] == -1)


def test_optional_subtask_sidecar_sets_text_and_validity(tmp_path: Path) -> None:
    sidecar = tmp_path / "meta" / "sidecar.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(
        '{"episodes":[{"episode_index":2,"task_instruction":"fold",'
        '"subtasks":[{"start_frame":3,"end_frame":5,'
        '"subtask_text":"pick corner"}]}]}',
        encoding="utf-8",
    )
    transform = LoadWSA15PhysicalAnnotationsFn().hydrate(SimpleNamespace(root=tmp_path))

    annotated = transform(
        {"episode_index": torch.tensor(2), "frame_index": torch.tensor(4), "task": "old"}
    )
    assert annotated[SUBTASK_TEXT_KEY] == "pick corner"
    assert annotated[SUBTASK_VALID_KEY]
    assert annotated["task"] == "fold"

    missing = transform(
        {"episode_index": torch.tensor(2), "frame_index": torch.tensor(8), "task": "old"}
    )
    assert missing[SUBTASK_TEXT_KEY] == ""
    assert not missing[SUBTASK_VALID_KEY]


def test_joint_embodiment_uses_semantic_slots_and_round_trips() -> None:
    reorder = [
        [8, 14, 0, 6],
        [14, 16, 6, 8],
        [0, 6, 8, 14],
        [6, 8, 14, 16],
    ]
    transform = CanonicalStateActionTransformFn(
        action_layout=[
            {"name": "left_arm", "size": 6},
            {"name": "left_gripper", "size": 2},
            {"name": "right_arm", "size": 6},
            {"name": "right_gripper", "size": 2},
        ],
        action_reorder=reorder,
        state_reorder=reorder,
        robot_type="dual_piper",
    )
    raw_state = torch.arange(16, dtype=torch.float32)
    raw_action = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    output = transform({OBS_STATE: raw_state, ACTION: raw_action})

    assert output[OBS_STATE].shape == (32,)
    assert output[ACTION].shape == (2, 32)
    assert output[ACTION_VALID_MASK_KEY].sum().item() == 16
    assert output[ACTION_VALID_MASK_KEY][14:16].all()
    assert output[ACTION_VALID_MASK_KEY][30:32].all()
    assert not output[ACTION_VALID_MASK_KEY][6:14].any()
    assert not output[ACTION_VALID_MASK_KEY][22:30].any()

    decoded = decode_canonical_actions(output[ACTION][None], output)[0]
    torch.testing.assert_close(decoded, raw_action)


def test_eef_actions_are_only_mapped_and_never_pose_converted() -> None:
    layout = [
        {"name": "left_eef_position_absolute", "size": 3},
        {"name": "left_eef_quaternion_xyzw_absolute", "size": 4},
        {"name": "left_grasp_absolute", "size": 1},
        {"name": "right_eef_position_absolute", "size": 3},
        {"name": "right_eef_quaternion_xyzw_absolute", "size": 4},
        {"name": "right_grasp_absolute", "size": 1},
    ]
    transform = CanonicalStateActionTransformFn(action_layout=layout, robot_type="human_eef")
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    state = torch.cat(
        (
            torch.tensor([1.0, 2.0, 3.0]),
            identity,
            torch.tensor([0.2]),
            torch.tensor([-1.0, 0.0, 1.0]),
            identity,
            torch.tensor([0.8]),
        )
    )
    raw_quaternion = torch.tensor([0.1, -0.2, 0.3, 0.4])
    action = torch.cat(
        (
            torch.tensor([1.1, 2.0, 3.0]),
            raw_quaternion,
            torch.tensor([0.3]),
            torch.tensor([-1.0, 0.2, 1.0]),
            identity,
            torch.tensor([0.7]),
        )
    )[None]
    output = transform({OBS_STATE: state, ACTION: action})

    torch.testing.assert_close(output[ACTION][0, 7:10], action[0, :3])
    torch.testing.assert_close(output[ACTION][0, 10:14], raw_quaternion)
    torch.testing.assert_close(output[OBS_STATE][7:14], state[:7])
    assert output[ACTION_VALID_MASK_KEY].sum().item() == 16
    assert not output[ACTION_VALID_MASK_KEY][15]
    assert not output[ACTION_VALID_MASK_KEY][31]
    decoded = decode_canonical_actions(output[ACTION][None], output)[0]
    torch.testing.assert_close(decoded, action)


def test_schema_delta_mask_is_the_only_abs_delta_authority() -> None:
    state = torch.arange(8, dtype=torch.float32)
    action = (state + 10)[None]
    mapping = {OBS_STATE: [OBS_STATE], ACTION: [ACTION]}

    # A schema mask of [-7] means every EEF dimension remains absolute.
    eef_schema = DatasetSchema(robot_type="eef", action_mask_spec=[-7])
    eef_data = DeltaActionTransformFn(mask=eef_schema.action_mask, mapping=mapping)(
        {OBS_STATE: state[:7].clone(), ACTION: action[:, :7].clone()}
    )
    torch.testing.assert_close(eef_data[ACTION], action[:, :7])
    eef_encoded = CanonicalStateActionTransformFn(
        action_layout=[
            {"name": "left_eef_position_absolute", "size": 3},
            {"name": "left_eef_quaternion_xyzw_absolute", "size": 4},
        ],
        robot_type="eef",
    )(eef_data)
    torch.testing.assert_close(eef_encoded[ACTION][0, 7:14], action[0, :7])
    torch.testing.assert_close(
        decode_canonical_actions(eef_encoded[ACTION][None], eef_encoded)[0], action[:, :7]
    )

    # Positive mask entries become delta; negative entries stay absolute.
    joint_schema = DatasetSchema(robot_type="joint", action_mask_spec=[6, -2])
    joint_data = DeltaActionTransformFn(mask=joint_schema.action_mask, mapping=mapping)(
        {OBS_STATE: state.clone(), ACTION: action.clone()}
    )
    torch.testing.assert_close(joint_data[ACTION][:, :6], torch.full((1, 6), 10.0))
    torch.testing.assert_close(joint_data[ACTION][:, 6:], action[:, 6:])


def test_dexterous_hand_slots_are_rejected() -> None:
    transform = CanonicalStateActionTransformFn(
        action_layout=[{"name": "left_hand_joint", "size": 6}],
        robot_type="dexterous_hand",
    )
    with pytest.raises(ValueError, match="Dexterous-hand"):
        transform({OBS_STATE: torch.zeros(6), ACTION: torch.zeros(1, 6)})


def test_wsa_attention_dependencies_are_explicit() -> None:
    # context=2 and every other group has one token.
    mask = build_wsa_attention_mask(torch.ones(1, 7, dtype=torch.bool), 2, (1, 1, 1, 1, 1))[0]
    context, subtask, future_2d, future_3d, state, action = 1, 2, 3, 4, 5, 6

    assert mask[0, 0] and not mask[0, 1]
    assert mask[context, :2].all()
    assert mask[subtask, [0, 1, subtask]].all()
    assert not mask[subtask, future_2d]
    assert mask[future_2d, [0, 1, subtask, future_2d, future_3d]].all()
    assert not mask[future_2d, action]
    assert mask[future_3d, [0, 1, subtask, future_3d, action]].all()
    assert not mask[future_3d, future_2d]
    assert mask[state, [0, 1, subtask, future_3d, state]].all()
    assert mask[action, [0, 1, subtask, future_3d, state, action]].all()
