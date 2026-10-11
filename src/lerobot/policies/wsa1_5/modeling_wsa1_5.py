"""WSA1.5: Qwen3.5 context, learned planning queries, and flow actions."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.datasets.wsa1_5 import (
    ACTION_VALID_MASK_KEY,
    ARM_DIM,
    CANONICAL_DIM,
    SUBTASK_TARGET_KEY,
    SUBTASK_TARGET_MASK_KEY,
    WORLD_TARGET_KEY,
    WORLD_TARGET_VALID_KEY,
    decode_canonical_actions,
    unpack_motion_canvas,
)
from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import (
    InternVLAA15,
    InternVLAA15Policy,
    create_sinusoidal_pos_embedding,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.wsa1_5.attention import build_wsa_attention_mask
from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA15Config
from lerobot.policies.wsa1_5.mot import WSA15MixtureOfTransformers
from lerobot.utils.constants import ACTION, OBS_PREFIX, OBS_STATE, OPENPI_ATTENTION_MASK_VALUE
from lerobot.utils.utils import format_big_number


def _flat_to_effector_tokens(vector: Tensor) -> Tensor:
    """[B,T,32] -> [B,2T,16], preserving the two semantic arm slots."""

    if vector.shape[-1] != CANONICAL_DIM:
        raise ValueError(f"Expected {CANONICAL_DIM} canonical values, got {vector.shape[-1]}")
    return vector.reshape(*vector.shape[:-1], 2, ARM_DIM).flatten(-3, -2)


def _effector_tokens_to_flat(tokens: Tensor, timesteps: int) -> Tensor:
    """[B,2T,16] -> [B,T,32], inverse of ``_flat_to_effector_tokens``."""

    batch = tokens.shape[0]
    return tokens.reshape(batch, timesteps, CANONICAL_DIM)


class WSA15(InternVLAA15):
    """Full-depth WSA-B-style continuation on the InternVLA-A1.5 interface."""

    def __init__(self, config: WSA15Config) -> None:
        super().__init__(config)
        self.qwen3_5_with_expert = WSA15MixtureOfTransformers(self.qwen3_5_with_expert)

        hidden = self.qwen3_5_with_expert.action_expert.config.hidden_size
        qwen_hidden = self.qwen3_5_with_expert.qwen3_5.config.text_config.hidden_size
        # Compact left/right semantic slots are projected independently.
        self.state_proj = nn.Linear(ARM_DIM, hidden)
        self.state_mask_proj = nn.Linear(ARM_DIM, hidden, bias=False)
        self.action_in_proj = nn.Linear(ARM_DIM, hidden)
        self.action_mask_proj = nn.Linear(ARM_DIM, hidden, bias=False)
        self.action_out_proj = nn.Linear(hidden, ARM_DIM)

        self.subtask_queries = nn.Parameter(torch.empty(config.num_subtask_tokens, hidden))
        self.future_2d_queries = nn.Parameter(torch.empty(config.num_future_2d_tokens, hidden))
        self.future_3d_queries = nn.Parameter(torch.empty(config.num_future_3d_tokens, hidden))
        nn.init.trunc_normal_(self.subtask_queries, std=0.02)
        nn.init.trunc_normal_(self.future_2d_queries, std=config.future_query_init_std)
        nn.init.trunc_normal_(self.future_3d_queries, std=config.future_query_init_std)
        self.future_2d_in_proj = nn.Linear(hidden, hidden)
        self.future_3d_in_proj = nn.Linear(hidden, hidden)
        self.future_2d_to_world = nn.Linear(hidden, hidden)
        self.future_3d_to_world = nn.Linear(hidden, hidden)
        self.subtask_to_qwen = nn.Linear(hidden, qwen_hidden)
        # The parent allocates one undifferentiated query bank for WAN.  WSA1.5
        # replaces it with explicit 2D and 3D banks while retaining the WAN
        # projection and public InternVLA-A1.5 interface.
        del self.learnable_tokens
        del self.learnable_tokens_in_proj
        if config.freeze_learnable_tokens:
            for parameter in (
                self.subtask_queries,
                self.future_2d_queries,
                self.future_3d_queries,
            ):
                parameter.requires_grad = False
            for module in (
                self.future_2d_in_proj,
                self.future_3d_in_proj,
                self.future_2d_to_world,
                self.future_3d_to_world,
            ):
                module.requires_grad_(False)

    @property
    def planning_lengths(self) -> tuple[int, int, int]:
        return (
            self.config.num_subtask_tokens,
            self.config.num_future_2d_tokens,
            self.config.num_future_3d_tokens,
        )

    def gradient_checkpointing_enable(self) -> None:
        super().gradient_checkpointing_enable()
        self.qwen3_5_with_expert.world_expert.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        super().gradient_checkpointing_disable()
        self.qwen3_5_with_expert.world_expert.gradient_checkpointing = False

    def embed_planning(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        subtask = self.subtask_queries[None].expand(batch_size, -1, -1)
        future_2d = self._apply_checkpoint(self.future_2d_in_proj, self.future_2d_queries)
        future_3d = self._apply_checkpoint(self.future_3d_in_proj, self.future_3d_queries)
        planning = torch.cat(
            (
                subtask,
                future_2d[None].expand(batch_size, -1, -1),
                future_3d[None].expand(batch_size, -1, -1),
            ),
            dim=1,
        ).to(device=device)
        dtype = self.qwen3_5_with_expert.world_expert.layers[0].mlp.up_proj.weight.dtype
        planning = planning.to(dtype=dtype)
        return planning, torch.ones(planning.shape[:2], dtype=torch.bool, device=device)

    def world_condition(self, future_out: Tensor) -> Tensor:
        """Add branch identity before the shared projection to WAN context."""

        split = self.config.num_future_2d_tokens
        future_2d = self.future_2d_to_world(future_out[:, :split])
        future_3d = self.future_3d_to_world(future_out[:, split:])
        return torch.cat((future_2d, future_3d), dim=1)

    def embed_action(
        self,
        state: Tensor,
        noisy_actions: Tensor,
        timestep: Tensor,
        state_valid_mask: Tensor | None = None,
        action_valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch, horizon = noisy_actions.shape[:2]
        state_tokens = _flat_to_effector_tokens(state[:, None]).reshape(batch, 2, ARM_DIM)
        if state_valid_mask is None:
            state_mask_tokens = torch.ones_like(state_tokens)
        else:
            state_mask_tokens = _flat_to_effector_tokens(state_valid_mask[:, None].to(state)).reshape(
                batch, 2, ARM_DIM
            )
        state_embedding = self.state_proj(state_tokens.to(self.state_proj.weight.dtype))
        state_embedding = state_embedding + self.state_mask_proj(
            state_mask_tokens.to(self.state_mask_proj.weight.dtype)
        )

        action_tokens = _flat_to_effector_tokens(noisy_actions)
        action_embedding = self.action_in_proj(action_tokens.to(self.action_in_proj.weight.dtype))
        if action_valid_mask is not None:
            expanded_mask = action_valid_mask[:, None].expand(-1, horizon, -1)
            mask_tokens = _flat_to_effector_tokens(expanded_mask.to(noisy_actions))
            action_embedding = action_embedding + self.action_mask_proj(
                mask_tokens.to(self.action_mask_proj.weight.dtype)
            )

        time_embedding = create_sinusoidal_pos_embedding(
            timestep,
            action_embedding.shape[-1],
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        ).to(action_embedding)
        time_embedding = time_embedding[:, None].expand_as(action_embedding)
        action_time = torch.cat((action_embedding, time_embedding), dim=-1)
        action_time = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(action_time)))
        embeddings = torch.cat((state_embedding, action_time), dim=1)
        return embeddings, torch.ones(embeddings.shape[:2], dtype=torch.bool, device=state.device)

    def _joint_forward(
        self,
        prefix_embs: Tensor,
        prefix_pad: Tensor,
        planning_embs: Tensor,
        planning_pad: Tensor,
        action_embs: Tensor,
        action_pad: Tensor,
        lang_tokens: Tensor,
        image_grid_thw: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        target_dtype = self.qwen3_5_with_expert.action_expert.layers[0].mlp.up_proj.weight.dtype
        prefix_embs = prefix_embs.to(target_dtype)
        planning_embs = planning_embs.to(target_dtype)
        action_embs = action_embs.to(target_dtype)
        padding = torch.cat((prefix_pad, planning_pad, action_pad), dim=1)
        group_lengths = (
            self.config.num_subtask_tokens,
            self.config.num_future_2d_tokens,
            self.config.num_future_3d_tokens,
            2,
            2 * self.config.chunk_size,
        )
        allowed = build_wsa_attention_mask(padding, prefix_pad.shape[1], group_lengths)
        attention = torch.where(
            allowed[:, None],
            torch.tensor(0.0, device=allowed.device, dtype=prefix_embs.dtype),
            torch.tensor(OPENPI_ATTENTION_MASK_VALUE, device=allowed.device, dtype=prefix_embs.dtype),
        )
        position_ids, _ = self.get_position_ids(lang_tokens, image_grid_thw, padding)
        (prefix_out, planning_out, action_out), _ = self.qwen3_5_with_expert.forward(
            attention_mask=attention,
            position_ids=position_ids,
            inputs_embeds=[prefix_embs, planning_embs, action_embs],
            use_cache=False,
            knowledge_insulation=self.config.knowledge_insulation,
            linear_attn_mask=padding,
        )
        return prefix_out, planning_out, action_out

    def _subtask_logits(self, planning_out: Tensor) -> Tensor:
        subtask_out = planning_out[:, : self.config.num_subtask_tokens]
        projected = self.subtask_to_qwen(subtask_out.to(self.subtask_to_qwen.weight.dtype))
        lm_head = self.qwen3_5_with_expert.qwen3_5.lm_head
        return lm_head(projected.to(lm_head.weight.dtype)).float()

    def _velocity(self, action_out: Tensor) -> Tensor:
        action_out = action_out[:, -2 * self.config.chunk_size :].float()
        token_velocity = self.action_out_proj(action_out.to(self.action_out_proj.weight.dtype)).float()
        return _effector_tokens_to_flat(token_velocity, self.config.chunk_size)

    def _forward_vqa(
        self, prefix_out: Tensor, labels: Tensor | None, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if labels is None:
            batch = actions.shape[0]
            return (
                torch.zeros(batch, device=actions.device),
                torch.zeros(batch, 1, device=actions.device),
                torch.zeros(batch, 1, dtype=torch.bool, device=actions.device),
            )
        logits = self.qwen3_5_with_expert.qwen3_5.lm_head(prefix_out).float()
        labels = labels.to(logits.device)
        batch, length, vocab = logits.shape
        per_token = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab),
            labels[:, 1:].reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape(batch, length - 1)
        mask = labels[:, 1:] != -100
        per_sample = (per_token * mask).sum(1) / mask.sum(1).clamp_min(1)
        return per_sample, per_token, mask

    def forward(
        self,
        pixel_values: Tensor,
        image_grid_thw: Tensor,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        actions: Tensor,
        *,
        state_valid_mask: Tensor | None = None,
        action_valid_mask: Tensor | None = None,
        subtask_targets: Tensor | None = None,
        subtask_mask: Tensor | None = None,
        labels: Tensor | None = None,
        video_frames: Tensor | None = None,
        video_mask: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        noise = self.sample_noise(actions.shape, actions.device) if noise is None else noise
        time = self.sample_time(actions.shape[0], actions.device) if time is None else time
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        target_velocity = noise - actions

        prefix, prefix_pad, _ = self.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks, labels
        )
        planning, planning_pad = self.embed_planning(actions.shape[0], actions.device)
        action_embs, action_pad = self.embed_action(state, x_t, time, state_valid_mask, action_valid_mask)
        prefix_out, planning_out, action_out = self._joint_forward(
            prefix,
            prefix_pad,
            planning,
            planning_pad,
            action_embs,
            action_pad,
            lang_tokens,
            image_grid_thw,
        )

        predicted_velocity = self._velocity(action_out)
        action_loss = F.mse_loss(target_velocity, predicted_velocity, reduction="none")
        vqa_loss, loss_per_token, token_mask = self._forward_vqa(prefix_out, labels, actions)

        subtask_logits = self._subtask_logits(planning_out)
        if subtask_targets is not None and subtask_mask is not None and subtask_mask.any():
            targets = subtask_targets.to(subtask_logits.device)
            valid = subtask_mask.to(subtask_logits.device).bool()
            subtask_loss = F.cross_entropy(subtask_logits[valid], targets[valid])
        else:
            subtask_loss = action_loss.new_zeros(())

        if self.config.action_loss_only:
            world_loss = action_loss.new_zeros(())
        else:
            has_world = video_mask.any() if video_mask is not None else video_frames is not None
            if has_world:
                future_out = planning_out[:, self.config.num_subtask_tokens :].float()
                if video_mask is not None:
                    valid_rows = video_mask.to(device=future_out.device, dtype=torch.bool)
                    future_out = future_out[valid_rows]
                    video_frames = video_frames[valid_rows]
                world_loss = self._compute_video_loss(video_frames, self.world_condition(future_out))
            else:
                world_loss = action_loss.new_zeros(())
        return action_loss, vqa_loss, subtask_loss, world_loss, loss_per_token, token_mask

    @torch.no_grad()
    def sample_actions_with_plan(
        self,
        pixel_values: Tensor,
        image_grid_thw: Tensor,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        *,
        state_valid_mask: Tensor | None = None,
        action_valid_mask: Tensor | None = None,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch = state.shape[0]
        steps = num_steps or self.config.num_inference_steps
        x_t = (
            noise
            if noise is not None
            else self.sample_noise((batch, self.config.chunk_size, CANONICAL_DIM), state.device)
        )
        prefix, prefix_pad, _ = self.embed_prefix(pixel_values, image_grid_thw, lang_tokens, lang_masks)
        dt = torch.tensor(-1.0 / steps, dtype=torch.float32, device=state.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=state.device)
        planning_out = None
        while time >= -dt / 2:
            expanded_time = time.expand(batch)
            planning, planning_pad = self.embed_planning(batch, state.device)
            action_embs, action_pad = self.embed_action(
                state, x_t, expanded_time, state_valid_mask, action_valid_mask
            )
            _, planning_out, action_out = self._joint_forward(
                prefix,
                prefix_pad,
                planning,
                planning_pad,
                action_embs,
                action_pad,
                lang_tokens,
                image_grid_thw,
            )
            x_t = x_t + dt * self._velocity(action_out)
            time = time + dt
        assert planning_out is not None
        return x_t, self._subtask_logits(planning_out), planning_out[:, self.config.num_subtask_tokens :]

    @torch.no_grad()
    def sample_actions(
        self,
        pixel_values: Tensor,
        image_grid_thw: Tensor,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        noise: Tensor | None = None,
        *,
        state_valid_mask: Tensor | None = None,
        action_valid_mask: Tensor | None = None,
        **_: object,
    ) -> Tensor:
        actions, _, _ = self.sample_actions_with_plan(
            pixel_values,
            image_grid_thw,
            lang_tokens,
            lang_masks,
            state,
            state_valid_mask=state_valid_mask,
            action_valid_mask=action_valid_mask,
            noise=noise,
        )
        return actions


class WSA15Policy(InternVLAA15Policy):
    """InternVLA-A1.5-compatible policy backed by the refactored WSA1.5 model."""

    config_class = WSA15Config
    name = "wsa1_5"

    def __init__(self, config: WSA15Config) -> None:
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.model = WSA15(config)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    def __str__(self) -> str:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        context = sum(parameter.numel() for parameter in self.model.qwen3_5_with_expert.qwen3_5.parameters())
        planning = sum(
            parameter.numel() for parameter in self.model.qwen3_5_with_expert.world_expert.parameters()
        )
        action = sum(
            parameter.numel() for parameter in self.model.qwen3_5_with_expert.action_expert.parameters()
        )
        return "\n".join(
            (
                "=" * 60,
                "Policy: WSA15Policy",
                f"  - Total/trainable     : {format_big_number(total)} / {format_big_number(trainable)}",
                f"  - Context expert      : {format_big_number(context)} (Qwen3.5-2B)",
                f"  - Planning expert     : {format_big_number(planning)}",
                f"  - Action expert       : {format_big_number(action)}",
                f"  - Planning queries    : subtask={self.config.num_subtask_tokens}, "
                f"2D={self.config.num_future_2d_tokens}, 3D={self.config.num_future_3d_tokens}",
                f"  - Canonical state/action: {CANONICAL_DIM}D -> two {ARM_DIM}D effector tokens",
                "=" * 60,
            )
        )

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch[OBS_STATE]
        if state.shape[-1] != CANONICAL_DIM:
            raise ValueError(f"WSA1.5 expects canonical {CANONICAL_DIM}D state, got {state.shape[-1]}")
        return state

    def prepare_action(self, batch: dict[str, Tensor]) -> Tensor:
        actions = batch[ACTION]
        if actions.shape[-1] != CANONICAL_DIM:
            raise ValueError(f"WSA1.5 expects canonical {CANONICAL_DIM}D actions, got {actions.shape[-1]}")
        return actions

    def _decode_actions(self, canonical: Tensor, batch: dict[str, Tensor]) -> Tensor:
        return decode_canonical_actions(
            canonical,
            batch,
            fallback_output_dim=self.config.output_features[ACTION].shape[0],
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        actions = self.model.sample_actions(
            batch[f"{OBS_PREFIX}pixel_values"],
            batch[f"{OBS_PREFIX}image_grid_thw"],
            batch[f"{OBS_PREFIX}input_ids"],
            batch[f"{OBS_PREFIX}attention_mask"],
            self.prepare_state(batch),
            state_valid_mask=batch.get("wsa.state_valid_mask"),
            action_valid_mask=batch.get(ACTION_VALID_MASK_KEY),
        )
        return self._decode_actions(actions, batch)

    @torch.no_grad()
    def predict_action_chunk_with_world(
        self,
        batch: dict[str, Tensor],
        num_video_steps: int = 50,
        **_: object,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.config.action_loss_only:
            raise RuntimeError(
                "predict_action_chunk_with_world requires action_loss_only=False so that WAN is loaded"
            )
        self.eval()
        canonical, subtask_logits, future_queries = self.model.sample_actions_with_plan(
            batch[f"{OBS_PREFIX}pixel_values"],
            batch[f"{OBS_PREFIX}image_grid_thw"],
            batch[f"{OBS_PREFIX}input_ids"],
            batch[f"{OBS_PREFIX}attention_mask"],
            self.prepare_state(batch),
            state_valid_mask=batch.get("wsa.state_valid_mask"),
            action_valid_mask=batch.get(ACTION_VALID_MASK_KEY),
        )
        actions = self._decode_actions(canonical, batch)
        batch_size = canonical.shape[0]
        condition = -torch.ones(
            batch_size,
            3,
            1,
            self.config.video_height,
            self.config.video_width,
            dtype=canonical.dtype,
            device=canonical.device,
        )
        canvas = self.model.generate_video(
            self.model.world_condition(future_queries),
            condition,
            num_inference_steps=num_video_steps,
        )
        motion_mask, motion_mask_ids = unpack_motion_canvas(canvas)
        return actions, {
            "subtask_token_ids": subtask_logits.argmax(dim=-1),
            "subtask_logits": subtask_logits,
            "future_2d_queries": future_queries[:, : self.config.num_future_2d_tokens],
            "future_3d_queries": future_queries[:, self.config.num_future_2d_tokens :],
            "motion_mask": motion_mask,
            "motion_mask_ids": motion_mask_ids,
            "motion_canvas": canvas,
        }

    @torch.no_grad()
    def predict_action_chunk_with_video(
        self, batch: dict[str, Tensor], num_video_steps: int = 50
    ) -> tuple[Tensor, Tensor]:
        actions, world = self.predict_action_chunk_with_world(batch, num_video_steps)
        return actions, world["motion_canvas"]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        actions = self.prepare_action(batch)
        action_mask = batch[ACTION_VALID_MASK_KEY].to(actions.device).bool()
        state_mask = batch["wsa.state_valid_mask"].to(actions.device).bool()
        labels = batch["VQA.labels"] if self.config.enable_vqa_loss else None
        world_mask = batch.get(WORLD_TARGET_VALID_KEY)
        if world_mask is not None:
            world_mask = world_mask.to(actions.device).bool()

        action_loss_raw, vqa_per_sample, subtask_loss, world_loss, _, _ = self.model(
            batch[f"{OBS_PREFIX}pixel_values"],
            batch[f"{OBS_PREFIX}image_grid_thw"],
            batch[f"{OBS_PREFIX}input_ids"],
            batch[f"{OBS_PREFIX}attention_mask"],
            self.prepare_state(batch),
            actions,
            state_valid_mask=state_mask,
            action_valid_mask=action_mask,
            subtask_targets=batch.get(SUBTASK_TARGET_KEY),
            subtask_mask=batch.get(SUBTASK_TARGET_MASK_KEY),
            labels=labels,
            video_frames=batch.get(WORLD_TARGET_KEY),
            video_mask=world_mask,
        )

        vqa_type = batch.get("vqa_type")
        robot_rows = torch.ones(actions.shape[0], dtype=torch.bool, device=actions.device)
        vqa_rows = torch.zeros_like(robot_rows)
        if vqa_type is not None:
            vqa_type = vqa_type.to(actions.device)
            robot_rows = vqa_type != 1
            vqa_rows = vqa_type == 1
        valid_action = action_mask[:, None] & robot_rows[:, None, None]
        denominator = valid_action.sum().clamp_min(1) * actions.shape[1]
        loss_action = (action_loss_raw * valid_action).sum() / denominator
        loss_vqa = vqa_per_sample[vqa_rows].mean() if vqa_rows.any() else loss_action.new_zeros(())

        if self.config.video_loss_only:
            loss_action = loss_action.new_zeros(())
        loss = (
            loss_action
            + self.config.lambda_vqa * loss_vqa
            + self.config.subtask_loss_weight * subtask_loss
            + self.config.video_loss_weight * world_loss
        )
        zero = loss.new_zeros(())
        return loss, {
            "loss": loss.item(),
            "loss_action": loss_action.item(),
            "loss_vqa": loss_vqa.item(),
            "loss_fast": zero.item(),
            "loss_subtask": subtask_loss.item(),
            "loss_physical": subtask_loss.item(),
            "loss_video": world_loss.item(),
        }


__all__ = ["WSA15", "WSA15Policy"]
