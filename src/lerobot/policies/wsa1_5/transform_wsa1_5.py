"""Text transforms for WSA1.5's context and fixed planning queries."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lerobot.datasets.wsa1_5 import (
    PHYSICAL_TARGET_VALID_KEY,
    SUBTASK_TARGET_KEY,
    SUBTASK_TARGET_MASK_KEY,
)
from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
    FASTInternVLAA15ActionTokenizerTransformFn,
    InternVLAA15ChatProcessorTransformFn,
    InternVLAA15VQAProcessorTransformFn,
)
from lerobot.transforms.core import DataDict, DataTransformFn


@DataTransformFn.register_subclass("wsa1_5_chat_processor")
@dataclass
class WSA15ChatProcessorTransformFn(InternVLAA15ChatProcessorTransformFn):
    """Build context tokens and a fixed-width discrete subtask target.

    Robot subtask text is deliberately removed before the parent chat
    processor is called, so it cannot leak into the Qwen context.  The planning
    expert predicts its BPE IDs in parallel from learned query tokens.
    """

    num_subtask_tokens: int = 16
    use_fast_action_tokens: bool = False
    tokenize_state: bool = False
    eval_output_schema: str = "<Discrete Subtask>"

    def __call__(self, data: DataDict) -> DataDict:
        target_text = str(data.get("sub_task", "")).strip()
        context = dict(data)
        context.pop("sub_task", None)
        context.pop(self.action_text_key, None)
        # Reserved extension point: no episodic-memory representation has been
        # selected yet, so WSA1.5 does not silently reinterpret pose text as
        # memory or feed an unreviewed memory channel into Qwen.
        context.pop(self.language_memory_key, None)
        context = super().__call__(context)

        targets = torch.full((self.num_subtask_tokens,), -100, dtype=torch.long)
        target_mask = torch.zeros(self.num_subtask_tokens, dtype=torch.bool)
        if target_text and self.mode != "eval":
            encoded = self.processor.tokenizer(
                target_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.num_subtask_tokens,
            )["input_ids"]
            eos = self.processor.tokenizer.eos_token_id
            if eos is not None and len(encoded) < self.num_subtask_tokens:
                encoded = [*encoded, eos]
            count = min(len(encoded), self.num_subtask_tokens)
            if count:
                targets[:count] = torch.tensor(encoded[:count], dtype=torch.long)
                target_mask[:count] = True

        context[SUBTASK_TARGET_KEY] = targets
        context[SUBTASK_TARGET_MASK_KEY] = target_mask
        context[PHYSICAL_TARGET_VALID_KEY] = torch.tensor(bool(target_mask.any()), dtype=torch.bool)
        return context


@DataTransformFn.register_subclass("wsa1_5_vqa_processor")
@dataclass
class WSA15VQAProcessorTransformFn(InternVLAA15VQAProcessorTransformFn):
    """VQA processor compatible with WSA1.5 mixed-modal pretraining."""


@DataTransformFn.register_subclass("wsa1_5_fast_action_tokenizer")
@dataclass
class FASTWSA15ActionTokenizerTransformFn(FASTInternVLAA15ActionTokenizerTransformFn):
    """Named WSA1.5 alias of the checkpoint-compatible FAST tokenizer."""


__all__ = [
    "FASTWSA15ActionTokenizerTransformFn",
    "WSA15ChatProcessorTransformFn",
    "WSA15VQAProcessorTransformFn",
]
