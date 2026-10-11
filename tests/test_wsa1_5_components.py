import math
from pathlib import Path
from runpy import run_path

import pytest
import torch

from lerobot.datasets.wsa1_5 import (
    ACTION_VALID_MASK_KEY,
    CanonicalStateActionTransformFn,
    decode_canonical_actions,
)
from lerobot.utils.constants import ACTION, OBS_STATE

# Keep this pure attention-mask unit test runnable in component-only
# environments that do not install the Qwen3.5 Transformers extension.  A
# normal policy import still uses lerobot.policies.wsa1_5.attention.
_ATTENTION_MODULE = run_path(str(Path(__file__).parents[1] / "src/lerobot/policies/wsa1_5/attention.py"))
build_wsa_attention_mask = _ATTENTION_MODULE["build_wsa_attention_mask"]


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


def test_eef_actions_become_relative_quaternions() -> None:
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
    z_quarter_turn = torch.tensor([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    action = torch.cat(
        (
            torch.tensor([1.1, 2.0, 3.0]),
            z_quarter_turn,
            torch.tensor([0.3]),
            torch.tensor([-1.0, 0.2, 1.0]),
            identity,
            torch.tensor([0.7]),
        )
    )[None]
    output = transform({OBS_STATE: state, ACTION: action})

    torch.testing.assert_close(output[ACTION][0, 7:10], torch.tensor([0.1, 0.0, 0.0]))
    torch.testing.assert_close(output[ACTION][0, 10:14], z_quarter_turn, atol=1e-5, rtol=1e-5)
    assert output[ACTION_VALID_MASK_KEY].sum().item() == 16
    assert not output[ACTION_VALID_MASK_KEY][15]
    assert not output[ACTION_VALID_MASK_KEY][31]
    decoded = decode_canonical_actions(output[ACTION][None], output)[0]
    torch.testing.assert_close(decoded[..., :3], action[..., :3], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(decoded[..., 3:7].abs(), action[..., 3:7].abs(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(decoded[..., 8:11], action[..., 8:11], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(decoded[..., 11:15].abs(), action[..., 11:15].abs(), atol=1e-5, rtol=1e-5)


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
