#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import draccus
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from lerobot.datasets.utils import write_json
from lerobot.utils.constants import SCHEDULER_STATE
from lerobot.utils.io_utils import deserialize_json_into_object


@dataclass
class LRSchedulerConfig(draccus.ChoiceRegistry, abc.ABC):
    num_warmup_steps: int

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LRScheduler | None:
        raise NotImplementedError


@LRSchedulerConfig.register_subclass("diffuser")
@dataclass
class DiffuserSchedulerConfig(LRSchedulerConfig):
    name: str = "cosine"
    num_warmup_steps: int | None = None

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        from diffusers.optimization import get_scheduler

        kwargs = {**asdict(self), "num_training_steps": num_training_steps, "optimizer": optimizer}
        return get_scheduler(**kwargs)


@LRSchedulerConfig.register_subclass("vqbet")
@dataclass
class VQBeTSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int
    num_vqvae_training_steps: int
    num_cycles: float = 0.5

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        def lr_lambda(current_step):
            if current_step < self.num_vqvae_training_steps:
                return float(1)
            else:
                adjusted_step = current_step - self.num_vqvae_training_steps
                if adjusted_step < self.num_warmup_steps:
                    return float(adjusted_step) / float(max(1, self.num_warmup_steps))
                progress = float(adjusted_step - self.num_warmup_steps) / float(
                    max(1, num_training_steps - self.num_warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("cosine_decay_with_warmup")
@dataclass
class CosineDecayWithWarmupSchedulerConfig(LRSchedulerConfig):
    """Used by Physical Intelligence to train Pi0.

    Automatically scales warmup and decay steps if num_training_steps < num_decay_steps.
    This ensures the learning rate schedule completes properly even with shorter training runs.
    """

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        # Auto-scale scheduler parameters if training steps are shorter than configured decay steps
        actual_warmup_steps = self.num_warmup_steps
        actual_decay_steps = self.num_decay_steps

        if num_training_steps < self.num_decay_steps:
            # Calculate scaling factor to fit the schedule into the available training steps
            scale_factor = num_training_steps / self.num_decay_steps
            actual_warmup_steps = int(self.num_warmup_steps * scale_factor)
            actual_decay_steps = num_training_steps

            logging.info(
                f"Auto-scaling LR scheduler: "
                f"num_training_steps ({num_training_steps}) < num_decay_steps ({self.num_decay_steps}). "
                f"Scaling warmup: {self.num_warmup_steps} → {actual_warmup_steps}, "
                f"decay: {self.num_decay_steps} → {actual_decay_steps} "
                f"(scale factor: {scale_factor:.3f})"
            )

        def lr_lambda(current_step):
            def linear_warmup_schedule(current_step):
                if current_step <= 0:
                    return 1 / (actual_warmup_steps + 1)
                frac = 1 - current_step / actual_warmup_steps
                return (1 / (actual_warmup_steps + 1) - 1) * frac + 1

            def cosine_decay_schedule(current_step):
                step = min(current_step, actual_decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
                alpha = self.decay_lr / self.peak_lr
                decayed = (1 - alpha) * cosine_decay + alpha
                return decayed

            if current_step < actual_warmup_steps:
                return linear_warmup_schedule(current_step)

            return cosine_decay_schedule(current_step)

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("multi_group_cosine_decay_with_warmup")
@dataclass
class MultiGroupCosineDecayWithWarmupSchedulerConfig(LRSchedulerConfig):
    """Independent warmup/cosine schedules for named optimizer parameter groups.

    ``group_schedules`` is keyed by the ``name`` stored in each optimizer
    parameter group. Every group must provide ``peak_lr``, ``decay_lr``,
    ``num_warmup_steps`` and ``num_decay_steps``.
    """

    # Kept for compatibility with the LRSchedulerConfig interface. Each group
    # uses its own warmup value from ``group_schedules``.
    num_warmup_steps: int = 0
    group_schedules: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        def make_lr_lambda(group_name: str, schedule: dict[str, float | int]):
            required = {"peak_lr", "decay_lr", "num_warmup_steps", "num_decay_steps"}
            missing = required - schedule.keys()
            if missing:
                raise ValueError(
                    f"LR schedule for optimizer group {group_name!r} is missing: {sorted(missing)}"
                )

            peak_lr = float(schedule["peak_lr"])
            decay_lr = float(schedule["decay_lr"])
            configured_warmup_steps = int(schedule["num_warmup_steps"])
            configured_decay_steps = int(schedule["num_decay_steps"])

            if peak_lr <= 0:
                raise ValueError(f"peak_lr for optimizer group {group_name!r} must be positive")
            if not 0 <= decay_lr <= peak_lr:
                raise ValueError(
                    f"decay_lr for optimizer group {group_name!r} must be in [0, peak_lr]"
                )
            if configured_warmup_steps < 0 or configured_decay_steps <= 0:
                raise ValueError(
                    f"Invalid warmup/decay steps for optimizer group {group_name!r}"
                )

            # Preserve the existing scheduler behavior: when a run is shorter
            # than the configured schedule, fit the full curve into the run.
            if num_training_steps < configured_decay_steps:
                scale = num_training_steps / configured_decay_steps
                warmup_steps = int(configured_warmup_steps * scale)
                decay_steps = num_training_steps
                logging.info(
                    "Auto-scaling LR schedule for group %s: warmup %d -> %d, decay %d -> %d",
                    group_name,
                    configured_warmup_steps,
                    warmup_steps,
                    configured_decay_steps,
                    decay_steps,
                )
            else:
                warmup_steps = configured_warmup_steps
                decay_steps = configured_decay_steps

            warmup_steps = min(warmup_steps, decay_steps)
            min_lr_ratio = decay_lr / peak_lr

            def lr_lambda(current_step: int) -> float:
                if warmup_steps > 0 and current_step < warmup_steps:
                    return (current_step + 1) / warmup_steps

                # Decay starts at peak_lr after warmup and reaches decay_lr at
                # decay_steps. Past decay_steps it remains at decay_lr.
                decay_span = max(1, decay_steps - warmup_steps)
                progress = (current_step - warmup_steps) / decay_span
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            return lr_lambda

        lr_lambdas = []
        seen_groups = set()
        for group in optimizer.param_groups:
            group_name = group.get("name")
            if not group_name:
                raise ValueError("Every optimizer parameter group must have a non-empty 'name'")
            if group_name in seen_groups:
                raise ValueError(f"Duplicate optimizer parameter group name: {group_name!r}")
            if group_name not in self.group_schedules:
                raise ValueError(f"No LR schedule configured for optimizer group {group_name!r}")

            seen_groups.add(group_name)
            lr_lambdas.append(make_lr_lambda(group_name, self.group_schedules[group_name]))

        unused_schedules = set(self.group_schedules) - seen_groups
        if unused_schedules:
            logging.info("Ignoring LR schedules for empty parameter groups: %s", sorted(unused_schedules))

        return LambdaLR(optimizer, lr_lambda=lr_lambdas, last_epoch=-1)


def save_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> None:
    state_dict = scheduler.state_dict()
    write_json(state_dict, save_dir / SCHEDULER_STATE)


def load_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> LRScheduler:
    state_dict = deserialize_json_into_object(save_dir / SCHEDULER_STATE, scheduler.state_dict())
    scheduler.load_state_dict(state_dict)
    return scheduler
