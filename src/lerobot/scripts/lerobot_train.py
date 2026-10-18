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
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
import multiprocessing as mp
from accelerate import Accelerator
from accelerate.utils import send_to_device
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.configuration_synthetic_embodiment import SyntheticEmbodimentDatasetConfig
from lerobot.datasets.factory import make_dataset, make_dataloader
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker, format_time
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
    gather_object,
)

SYNTHETIC_INPUT_LOG_FREQ = 200


def log_synthetic_input(step: int, batch: dict[str, Any], tokenizer: Any) -> None:
    """Log the first sample exactly as it enters the synthetic action policy."""
    prompt_mask = (
        batch["observation.attention_mask"][0].bool()
        & (batch["VQA.labels"][0] == -100)
    )
    prompt_ids = batch["observation.input_ids"][0][prompt_mask].detach().cpu().tolist()
    prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    logging.info(
        "[synthetic input] step=%d rank=0 sample=0\n"
        "prompt: %s\n",
        step,
        prompt,
    )


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)


    is_wsa = getattr(accelerator.unwrap_model(policy), "name", None) == "WSA_1_5"
    if is_wsa:
        from lerobot.policies.wsa1_5.timing import StageTimer

        update_timer = StageTimer(loss.device)
    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if not accelerator.sync_gradients:
        grad_norm = torch.zeros((), device=loss.device)
    elif grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    if is_wsa:
        update_timer.mark("backward_s")
    # Optimizer step
    if accelerator.sync_gradients:
        if getattr(accelerator.unwrap_model(policy), "name", None) == "WSA_1_5" and not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite WSA gradients; optimizer update was not applied")
        with lock if lock is not None else nullcontext():
            optimizer.step()
        optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None and accelerator.sync_gradients:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    if is_wsa:
        update_timer.mark("optimizer_s")
        output_dict.update(update_timer.values())
    train_metrics.loss = loss.item()
    if "loss_action" in output_dict:
        train_metrics.loss_action = output_dict["loss_action"]
    if "loss_video" in output_dict:
        train_metrics.loss_video = output_dict["loss_video"]
    if "loss_vqa" in output_dict:
        train_metrics.loss_vqa = output_dict["loss_vqa"]
    if "loss_fast" in output_dict:
        train_metrics.loss_fast = output_dict["loss_fast"]
    if "loss_subtask" in output_dict:
        train_metrics.loss_subtask = output_dict["loss_subtask"]
    if "loss_physical" in output_dict:
        train_metrics.loss_physical = output_dict["loss_physical"]
    for key in ["loss_rgb_flow", "loss_motion_flow", "loss_3d_feature", "input_s", "mot_s", "pred_s",
                "backward_s", "optimizer_s"]:
        if key in output_dict:
            setattr(train_metrics, key, output_dict[key])
    train_metrics.grad_norm = grad_norm.item()
    group_lrs = {
        group["name"]: group["lr"]
        for group in optimizer.param_groups
        if "name" in group
    }
    # Keep the existing aggregate metric meaningful for action training, and
    # expose every named group separately to structured loggers such as WandB.
    train_metrics.lr = group_lrs.get("action", optimizer.param_groups[0]["lr"])
    if output_dict is not None and group_lrs:
        output_dict.update({f"lr/{name}": lr for name, lr in group_lrs.items()})
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    # mp.set_start_method("spawn", force=True)
    
    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from datetime import timedelta

        from accelerate.utils import (
            DistributedDataParallelKwargs,
            InitProcessGroupKwargs,
        )

        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=True,
            gradient_as_bucket_view=cfg.policy.type == "WSA_1_5",
        )

        nccl_timeout_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(hours=2)
        )

        wsa_options = {}
        if cfg.policy.type == "WSA_1_5":
            from accelerate.utils import GradientAccumulationPlugin
            wsa_options = {
                "mixed_precision": {"bfloat16": "bf16", "float16": "fp16", "float32": "no"}[cfg.policy.dtype],
                "gradient_accumulation_plugin": GradientAccumulationPlugin(
                    num_steps=cfg.policy.gradient_accumulation_steps, sync_with_dataloader=False),
            }
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            **wsa_options,
            kwargs_handlers=[
                ddp_kwargs,
                nccl_timeout_kwargs,
            ],
        )

    init_logging(accelerator=accelerator)
    accumulation_steps = getattr(cfg.policy, "gradient_accumulation_steps", 1) if cfg.policy.type == "WSA_1_5" else 1
    if cfg.policy.type == "WSA_1_5" and accelerator.gradient_accumulation_steps != accumulation_steps:
        raise ValueError("Provided Accelerator gradient accumulation disagrees with WSA policy config")
    if cfg.policy.type == "WSA_1_5":
        expected_precision = {"bfloat16": "bf16", "float16": "fp16", "float32": "no"}[cfg.policy.dtype]
        if accelerator.mixed_precision != expected_precision:
            raise ValueError("Provided Accelerator precision disagrees with WSA policy config")

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))
    
    try:
        cfg.validate()
    except FileExistsError:
        if is_main_process:
            raise
        logging.warning(f"Ignoring existing output_dir on non-main process: {cfg.output_dir}")
    accelerator.wait_for_everyone()

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # Pin this rank to Accelerate's device before any model/CUDA initialization.
    device = accelerator.device
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
    cfg.policy.device = str(device)

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # # Use accelerator's device
    # device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
    dataset, data_stats = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    # if not is_main_process:
    #     dataset, data_stats = make_dataset(cfg)

    # accelerator.wait_for_everyone()

    if accelerator.num_processes>1:
        all_data_stats = gather_object(data_stats, accelerator)
    else:
        all_data_stats = [data_stats]
        
    if is_main_process:
        merged_data_stats = {}
        for rank_stats in all_data_stats:
            merged_data_stats.update(rank_stats)
        data_stats = merged_data_stats
    else:
        data_stats = None

    accelerator.wait_for_everyone()

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
    )
    if cfg.policy.type == "WSA_1_5":
        policy.prepare_for_training(device, log_parameters=is_main_process)

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    if cfg.dataset.dist_loading:
        num_frames = sum(gather_object(dataset.num_frames, accelerator))
        num_episodes = sum(gather_object(dataset.num_episodes, accelerator))
    else:
        num_frames = dataset.num_frames
        num_episodes = dataset.num_episodes
    num_processes = accelerator.num_processes
    effective_bs = cfg.batch_size * num_processes * accumulation_steps

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"\033[91m\033[1mnum_frames={num_frames} ({format_big_number(num_frames)})\033[0m")
        logging.info(f"\033[91m\033[1mnum_episodes={num_episodes} ({format_big_number(num_episodes)})\033[0m")
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} x {accumulation_steps} = {effective_bs}")
        if cfg.policy.type != "WSA_1_5":
            logging.info(f"policy info:\n{policy}")

    # create dataloader for offline training
    dataloader, dl_self_managed = make_dataloader(cfg, dataset)

    # When the dataloader manages its own sampling (MixedMultimodalDataset,
    # dist_loading), it must NOT be wrapped by accelerator.prepare and
    # batches are sent to device manually in the training loop.
    accelerator.wait_for_everyone()
    if cfg.dataset.dist_loading:
        policy, optimizer, lr_scheduler = accelerator.prepare(
            policy, optimizer, lr_scheduler
        )
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )
    dl_iter = cycle(dataloader)

    policy.train()

    if cfg.policy.type == "internvla_a1_5":
        train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "loss_action": AverageMeter("loss_action", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }
    else:
        train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }
  
    if getattr(cfg.policy, "enable_vqa_loss", False):
        train_metrics["loss_vqa"] = AverageMeter("loss_vqa", ":.3f")
        train_metrics["loss_action"] = AverageMeter("loss_action", ":.3f")
    
    if cfg.policy.type == "internvla_a1_5":
        train_metrics["loss_video"] = AverageMeter("loss_video", ":.3f")

    if cfg.policy.type == "internvla_a1_5":
        train_metrics["loss_fast"] = AverageMeter("loss_fast", ":.3f")
        train_metrics["loss_subtask"] = AverageMeter("loss_subtask", ":.3f")

    if cfg.policy.type == "WSA_1_5":
        for metric in ["loss_rgb_flow", "loss_motion_flow", "loss_3d_feature"]:
            train_metrics[metric] = AverageMeter(metric, ":.3f")
        for metric in ["input_s", "mot_s", "pred_s", "backward_s", "optimizer_s"]:
            train_metrics[metric] = AverageMeter(metric, ":.3f")


    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes * accumulation_steps
    train_tracker = MetricsTracker(
        effective_batch_size,
        num_frames,
        num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
        log_episode_progress=cfg.policy.type != "WSA_1_5",
    )

    if is_main_process:
        logging.info("Start offline training on a fixed dataset")
        training_start_time = time.perf_counter()

    input_tokenizer = None
    for _ in range(step, cfg.steps):
        for _micro_step in range(accumulation_steps):
            start_time = time.perf_counter()
            batch = next(dl_iter)
            if cfg.dataset.dist_loading or dl_self_managed:
                batch = send_to_device(batch, accelerator.device, non_blocking=True)
            train_tracker.dataloading_s = time.perf_counter() - start_time
            accumulation_context = accelerator.accumulate(policy) if cfg.policy.type == "WSA_1_5" else nullcontext()
            with accumulation_context:
                train_tracker, output_dict = update_policy(
                    train_tracker,
                    policy,
                    batch,
                    optimizer,
                    cfg.optimizer.grad_clip_norm,
                    accelerator=accelerator,
                    lr_scheduler=lr_scheduler,
                )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        if cfg.policy.type == "WSA_1_5" and is_main_process and cfg.eval_freq > 0 and step % cfg.eval_freq == 0:
            from lerobot.policies.wsa1_5.visualization import save_training_sample

            snapshot_dir = save_training_sample(batch, dataset, cfg.output_dir / "validation", step)
            logging.info("[WSA sample] saved current batch snapshot to %s", snapshot_dir)
        if (
            is_main_process
            and isinstance(cfg.dataset, SyntheticEmbodimentDatasetConfig)
            and step % SYNTHETIC_INPUT_LOG_FREQ == 0
        ):
            if input_tokenizer is None:
                from transformers.models.qwen3_5 import Qwen3_5Tokenizer

                input_tokenizer = Qwen3_5Tokenizer.from_pretrained(
                    cfg.policy.vlm_model_name_or_path
                )
            log_synthetic_input(step, batch, input_tokenizer)
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps

        if is_log_step:
            avg_update_time = train_tracker.update_s.avg if hasattr(train_tracker.update_s, 'avg') else train_tracker.update_s.val
            steps_per_second = 1.0 / avg_update_time if avg_update_time > 0 else 0

            elapsed_time = time.perf_counter() - training_start_time if training_start_time else 0
            remaining_steps = cfg.steps - step
            estimated_remaining_time = remaining_steps * avg_update_time if avg_update_time > 0 else 0

            elapsed_str = format_time(elapsed_time)
            remaining_str = format_time(estimated_remaining_time)

            logging.info(f" \033[92m\033[1m{elapsed_str} << {remaining_str}\033[0m | \033[96m\033[1m{steps_per_second:.2f} iters/s\033[0m | {train_tracker}")
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                logging.info(colored("Checkpoint saved at:", "cyan", attrs=["bold"]) + f" {checkpoint_dir}")
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    data_stats=data_stats, 
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
