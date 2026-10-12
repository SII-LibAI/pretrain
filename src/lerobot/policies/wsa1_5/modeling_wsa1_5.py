"""Test-only WSA1.5 policy stub.

The real WSA1.5 model is intentionally absent while its dataset pipeline is
being validated. This module keeps the policy contract required by the
training loop, validates the final collated batch, and returns a differentiable
zero loss without loading a VLM, action expert, or world model.
"""

from __future__ import annotations

import logging
from collections import deque

import torch
from torch import Tensor, nn

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
    decode_canonical_actions,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA15Config
from lerobot.utils.constants import ACTION, OBS_PREFIX, OBS_STATE


class WSA15(nn.Module):
    """One-parameter placeholder used only to exercise backward/optimizer/DDP."""

    def __init__(self, config: WSA15Config) -> None:
        super().__init__()
        self.config = config
        self.test_anchor = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def zero_loss(self) -> Tensor:
        # Keep a real parameter in the graph so Accelerate/DDP, gradient
        # clipping, optimizer stepping, and checkpoint saving are all tested.
        return self.test_anchor.square()


class WSA15Policy(PreTrainedPolicy):
    """Empty policy that validates the complete WSA1.5 data-flow contract."""

    config_class = WSA15Config
    name = "wsa1_5"

    _REQUIRED_CONTEXT_KEYS = (
        f"{OBS_PREFIX}pixel_values",
        f"{OBS_PREFIX}image_grid_thw",
        f"{OBS_PREFIX}input_ids",
        f"{OBS_PREFIX}attention_mask",
        "VQA.labels",
    )
    _REQUIRED_WSA_KEYS = (
        STATE_VALID_MASK_KEY,
        ACTION_VALID_MASK_KEY,
        ACTION_DECODE_MATRIX_KEY,
        ACTION_DECODE_BIAS_KEY,
        ACTION_RAW_DIM_KEY,
        SUBTASK_TARGET_KEY,
        SUBTASK_TARGET_MASK_KEY,
        FUTURE_2D_TARGET_KEY,
        FUTURE_2D_VALID_KEY,
        FUTURE_3D_TARGET_KEY,
        FUTURE_3D_VALID_KEY,
    )

    def __init__(self, config: WSA15Config) -> None:
        super().__init__(config)
        config.validate_features()
        self.model = WSA15(config)
        self._logged_dataflow_summary = False
        self.reset()

    def __str__(self) -> str:
        return (
            "WSA15Policy(test-only stub: validates transformed batches; "
            "no VLM/action/world model is loaded)"
        )

    def get_optim_params(self):
        return self.parameters()

    def reset(self) -> None:
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    @staticmethod
    def _require_tensor(batch: dict[str, Tensor], key: str) -> Tensor:
        if key not in batch:
            raise KeyError(f"WSA1.5 data-flow stub is missing required batch key: {key}")
        value = batch[key]
        if not isinstance(value, Tensor):
            raise TypeError(
                f"WSA1.5 batch key {key!r} must be a torch.Tensor, got {type(value).__name__}"
            )
        return value

    def _validate_batch(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        state = self._require_tensor(batch, OBS_STATE)
        action = self._require_tensor(batch, ACTION)
        for key in self._REQUIRED_CONTEXT_KEYS + self._REQUIRED_WSA_KEYS:
            self._require_tensor(batch, key)

        if state.ndim != 2 or state.shape[-1] != CANONICAL_DIM:
            raise ValueError(
                f"{OBS_STATE} must be [B,{CANONICAL_DIM}], got {tuple(state.shape)}"
            )
        if action.ndim != 3 or action.shape[-1] != CANONICAL_DIM:
            raise ValueError(
                f"{ACTION} must be [B,T,{CANONICAL_DIM}], got {tuple(action.shape)}"
            )
        if action.shape[0] != state.shape[0]:
            raise ValueError(
                f"State/action batch sizes differ: {state.shape[0]} vs {action.shape[0]}"
            )
        if action.shape[1] != self.config.chunk_size:
            raise ValueError(
                f"Action horizon must equal chunk_size={self.config.chunk_size}, "
                f"got {action.shape[1]}"
            )

        state_mask = batch[STATE_VALID_MASK_KEY]
        action_mask = batch[ACTION_VALID_MASK_KEY]
        expected_mask_shape = state.shape
        if state_mask.shape != expected_mask_shape:
            raise ValueError(
                f"{STATE_VALID_MASK_KEY} must be {tuple(expected_mask_shape)}, "
                f"got {tuple(state_mask.shape)}"
            )
        if action_mask.shape != expected_mask_shape:
            raise ValueError(
                f"{ACTION_VALID_MASK_KEY} must be {tuple(expected_mask_shape)}, "
                f"got {tuple(action_mask.shape)}"
            )
        if state_mask.dtype is not torch.bool or action_mask.dtype is not torch.bool:
            raise TypeError("WSA1.5 state/action validity masks must have dtype=torch.bool")

        input_ids = batch[f"{OBS_PREFIX}input_ids"]
        attention_mask = batch[f"{OBS_PREFIX}attention_mask"]
        labels = batch["VQA.labels"]
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError(
                "WSA1.5 token tensors must be input_ids=[B,L] with a matching attention_mask"
            )
        if labels.shape != input_ids.shape:
            raise ValueError(
                f"VQA.labels must match input_ids shape {tuple(input_ids.shape)}, "
                f"got {tuple(labels.shape)}"
            )

        batch_size = state.shape[0]
        expected_shapes = {
            ACTION_DECODE_MATRIX_KEY: (batch_size, CANONICAL_DIM, CANONICAL_DIM),
            ACTION_DECODE_BIAS_KEY: (batch_size, CANONICAL_DIM),
            ACTION_RAW_DIM_KEY: (batch_size,),
            SUBTASK_TARGET_KEY: (batch_size, self.config.num_subtask_tokens),
            SUBTASK_TARGET_MASK_KEY: (batch_size, self.config.num_subtask_tokens),
            FUTURE_2D_TARGET_KEY: (
                batch_size,
                self.config.num_video_frames,
                3,
                self.config.video_height,
                self.config.video_width,
            ),
            FUTURE_2D_VALID_KEY: (batch_size,),
            FUTURE_3D_TARGET_KEY: (
                batch_size,
                self.config.num_video_frames,
                3,
                self.config.video_height,
                self.config.video_width,
            ),
            FUTURE_3D_VALID_KEY: (batch_size,),
        }
        for key, expected_shape in expected_shapes.items():
            if batch[key].shape != expected_shape:
                raise ValueError(
                    f"{key} must be {expected_shape}, got {tuple(batch[key].shape)}"
                )

        for key in (
            SUBTASK_TARGET_MASK_KEY,
            FUTURE_2D_VALID_KEY,
            FUTURE_3D_VALID_KEY,
        ):
            if batch[key].dtype is not torch.bool:
                raise TypeError(f"{key} must have dtype=torch.bool")

        if not torch.isfinite(state).all():
            raise ValueError(f"{OBS_STATE} contains NaN or Inf after WSA1.5 transforms")
        if not torch.isfinite(action).all():
            raise ValueError(f"{ACTION} contains NaN or Inf after WSA1.5 transforms")
        return state, action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        state, action = self._validate_batch(batch)
        loss = self.model.zero_loss()
        state_mask = batch[STATE_VALID_MASK_KEY]
        action_mask = batch[ACTION_VALID_MASK_KEY]
        future_2d_valid = batch[FUTURE_2D_VALID_KEY].bool()
        future_3d_valid = batch[FUTURE_3D_VALID_KEY].bool()
        subtask_mask = batch[SUBTASK_TARGET_MASK_KEY].bool()

        if not self._logged_dataflow_summary:
            logging.info(
                "WSA1.5 data-flow batch validated: state=%s action=%s input_ids=%s "
                "future_2d=%s future_3d=%s valid_ratio(state=%.3f action=%.3f future_2d=%.3f "
                "future_3d=%.3f subtask=%.3f)",
                tuple(state.shape),
                tuple(action.shape),
                tuple(batch[f"{OBS_PREFIX}input_ids"].shape),
                tuple(batch[FUTURE_2D_TARGET_KEY].shape),
                tuple(batch[FUTURE_3D_TARGET_KEY].shape),
                float(state_mask.float().mean()),
                float(action_mask.float().mean()),
                float(future_2d_valid.float().mean()),
                float(future_3d_valid.float().mean()),
                float(subtask_mask.float().mean()),
            )
            self._logged_dataflow_summary = True

        return loss, {
            "loss": float(loss.detach()),
            "loss_action": 0.0,
            "loss_vqa": 0.0,
            "loss_fast": 0.0,
            "loss_subtask": 0.0,
            "loss_physical": 0.0,
            "loss_video": 0.0,
            "dataflow/batch_size": float(state.shape[0]),
            "dataflow/action_horizon": float(action.shape[1]),
            "dataflow/state_valid_ratio": float(state_mask.float().mean()),
            "dataflow/action_valid_ratio": float(action_mask.float().mean()),
            "dataflow/future_2d_valid_ratio": float(future_2d_valid.float().mean()),
            "dataflow/future_3d_valid_ratio": float(future_3d_valid.float().mean()),
            "dataflow/subtask_valid_ratio": float(subtask_mask.float().mean()),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **_: object) -> Tensor:
        state = self._require_tensor(batch, OBS_STATE)
        if state.ndim != 2 or state.shape[-1] != CANONICAL_DIM:
            raise ValueError(
                f"{OBS_STATE} must be [B,{CANONICAL_DIM}], got {tuple(state.shape)}"
            )
        canonical = state.new_zeros(
            state.shape[0], self.config.chunk_size, self.config.max_action_dim
        )
        fallback_dim = self.config.output_features[ACTION].shape[0]
        return decode_canonical_actions(canonical, batch, fallback_output_dim=fallback_dim)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: object) -> Tensor:
        self.eval()
        if not self._action_queue:
            actions = self.predict_action_chunk(batch, **kwargs)
            actions = actions[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()


__all__ = ["WSA15", "WSA15Policy"]
