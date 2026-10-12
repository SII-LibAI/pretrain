import pytest
import torch

pytest.importorskip(
    "transformers.models.qwen3_5",
    reason="WSA1.5 data transforms require the repository's Qwen3.5 Transformers build",
)

from lerobot.datasets.wsa1_5 import (
    ACTION_DECODE_BIAS_KEY,
    ACTION_DECODE_MATRIX_KEY,
    ACTION_RAW_DIM_KEY,
    ACTION_VALID_MASK_KEY,
    CANONICAL_DIM,
    FUTURE_2D_TARGET_KEY,
    FUTURE_2D_VALID_KEY,
    FUTURE_3D_TARGET_KEY,
    FUTURE_3D_VALID_KEY,
    STATE_VALID_MASK_KEY,
    SUBTASK_TARGET_KEY,
    SUBTASK_TARGET_MASK_KEY,
)
from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA15Config
from lerobot.policies.wsa1_5.modeling_wsa1_5 import WSA15Policy
from lerobot.utils.constants import ACTION, OBS_PREFIX, OBS_STATE


def _batch(config: WSA15Config, batch_size: int = 2) -> dict[str, torch.Tensor]:
    token_count = 8
    return {
        OBS_STATE: torch.zeros(batch_size, CANONICAL_DIM),
        ACTION: torch.zeros(batch_size, config.chunk_size, CANONICAL_DIM),
        f"{OBS_PREFIX}pixel_values": torch.zeros(batch_size, 3, 16, 16),
        f"{OBS_PREFIX}image_grid_thw": torch.ones(batch_size, 3, dtype=torch.long),
        f"{OBS_PREFIX}input_ids": torch.zeros(batch_size, token_count, dtype=torch.long),
        f"{OBS_PREFIX}attention_mask": torch.ones(batch_size, token_count, dtype=torch.long),
        "VQA.labels": torch.full((batch_size, token_count), -100, dtype=torch.long),
        STATE_VALID_MASK_KEY: torch.ones(batch_size, CANONICAL_DIM, dtype=torch.bool),
        ACTION_VALID_MASK_KEY: torch.ones(batch_size, CANONICAL_DIM, dtype=torch.bool),
        ACTION_DECODE_MATRIX_KEY: torch.eye(CANONICAL_DIM).repeat(batch_size, 1, 1),
        ACTION_DECODE_BIAS_KEY: torch.zeros(batch_size, CANONICAL_DIM),
        ACTION_RAW_DIM_KEY: torch.full((batch_size,), CANONICAL_DIM, dtype=torch.long),
        SUBTASK_TARGET_KEY: torch.full(
            (batch_size, config.num_subtask_tokens), -100, dtype=torch.long
        ),
        SUBTASK_TARGET_MASK_KEY: torch.zeros(
            batch_size, config.num_subtask_tokens, dtype=torch.bool
        ),
        FUTURE_2D_TARGET_KEY: torch.zeros(
            batch_size,
            config.num_video_frames,
            3,
            config.video_height,
            config.video_width,
        ),
        FUTURE_2D_VALID_KEY: torch.zeros(batch_size, dtype=torch.bool),
        FUTURE_3D_TARGET_KEY: torch.zeros(
            batch_size,
            config.num_video_frames,
            3,
            config.video_height,
            config.video_width,
        ),
        FUTURE_3D_VALID_KEY: torch.zeros(batch_size, dtype=torch.bool),
    }


def test_wsa15_stub_validates_dataflow_and_backpropagates() -> None:
    config = WSA15Config(device="cpu", push_to_hub=False)
    policy = WSA15Policy(config)
    loss, metrics = policy(_batch(config))

    assert loss.requires_grad
    assert loss.item() == 0.0
    assert metrics["dataflow/batch_size"] == 2.0
    assert metrics["dataflow/action_horizon"] == float(config.chunk_size)
    loss.backward()
    assert policy.model.test_anchor.grad is not None


def test_wsa15_stub_rejects_missing_final_batch_key() -> None:
    config = WSA15Config(device="cpu", push_to_hub=False)
    policy = WSA15Policy(config)
    batch = _batch(config)
    del batch[FUTURE_3D_VALID_KEY]

    try:
        policy(batch)
    except KeyError as error:
        assert FUTURE_3D_VALID_KEY in str(error)
    else:
        raise AssertionError("The data-flow stub accepted an incomplete WSA1.5 batch")
