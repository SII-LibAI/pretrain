"""SERVER ONLY: actual-model smoke/overfit/cache/checkpoint validation.

Uses the repository's update_policy, optimizer and checkpoint helpers, rather
than providing a replacement trainer. This tool is never run with weights by
the local development agent.
"""
import argparse
import json
import logging
from pathlib import Path

import draccus
import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin, send_to_device

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset, make_dataloader
from lerobot.datasets.wsa1_5 import imagenet_normalize
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.wsa1_5.modeling_wsa1_5 import WSA1_5_Policy
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.train_utils import load_training_state, save_checkpoint


def synthetic_batch(cfg, device):
    """Explicit test fixture only, never a production-data/model fallback."""
    b = cfg.batch_size
    p = cfg.policy
    rgb = lambda frames: imagenet_normalize(torch.rand(b, frames, 3, p.rgb_height, p.rgb_width, device=device))
    return {"observed_rgb": rgb(1)[:, 0], "condition_text": ["Task: move block\nSubtask: place\nHand: both hands\nScene: table"] * b,
            "observed_valid": torch.ones(b, 1, p.rgb_height, p.rgb_width, dtype=torch.bool, device=device),
            "targets": {
                "future_rgb": rgb(32),
                "future_motion": imagenet_normalize(torch.rand(b, 32, 3, p.motion_height, p.motion_width, device=device)),
                "goal_rgb_views": [[torch.rand(3, 224, 224) for _ in range(3)] for _ in range(b)],
                "rgb_valid": torch.ones(b, 32, 1, p.rgb_height, p.rgb_width, dtype=torch.bool, device=device),
                "motion_valid": torch.ones(b, 32, dtype=torch.bool, device=device),
                "goal_view_valid": torch.ones(b, 3, dtype=torch.bool, device=device),
            }, "sample_id": [f"synthetic-test:{i}" for i in range(b)],
            "sidecar_digest": ["explicit-test-fixture"] * b, "prompt_version": ["synthetic.v1"] * b,
            "frame_timestamps": torch.arange(1, 33, device=device).double()[None].expand(b, -1) / 30}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--execute-on-server", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--steps", type=int, default=1, help="Use >1 with fixed-noise updates for the overfit check")
    parser.add_argument("--reload-checkpoint", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--cache-batches", type=int, default=1)
    parser.add_argument("--report", type=Path, default=Path("outputs/wsa1_5_server_validation.json"))
    args = parser.parse_args()
    if not args.execute_on_server:
        parser.error("Model execution requires --execute-on-server")
    if args.steps < 1 or args.cache_batches < 1:
        parser.error("steps and cache-batches must be positive")
    logging.basicConfig(level=logging.INFO)
    cfg = draccus.parse(TrainPipelineConfig, str(args.config_path), args=[])
    # Explicit diagnostic schedule: production never auto-rescales its LR curve.
    cfg.steps = args.steps
    if cfg.policy.type != "WSA_1_5":
        parser.error("This verifier accepts only WSA_1_5")
    cfg.policy.scheduler_warmup_steps = min(cfg.policy.scheduler_warmup_steps, args.steps // 5)
    cfg.policy.scheduler_decay_steps = args.steps
    accelerator = Accelerator(
        mixed_precision={"float32": "no", "bfloat16": "bf16", "float16": "fp16"}[cfg.policy.dtype],
        step_scheduler_with_optimizer=False,
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=cfg.policy.gradient_accumulation_steps, sync_with_dataloader=False))
    if accelerator.num_processes != 1:
        parser.error("Run this diagnostic in one process; use the pretrain launcher for DDP validation")
    cfg.policy.device = str(accelerator.device)
    cfg.optimizer = cfg.policy.get_optimizer_preset()
    cfg.scheduler = cfg.policy.get_scheduler_preset()
    data_stats = {}
    loader = None
    if args.synthetic:
        torch.manual_seed(37)
        batch = synthetic_batch(cfg, accelerator.device)
        num_frames, num_episodes = cfg.batch_size, 1
    else:
        dataset, data_stats = make_dataset(cfg)
        loader, _ = make_dataloader(cfg, dataset)
        batch = send_to_device(next(iter(loader)), accelerator.device)
        num_frames, num_episodes = dataset.num_frames, dataset.num_episodes
    policy = make_policy(cfg.policy)
    if args.cache_only:
        if not cfg.policy.cache_dir:
            parser.error("cache-only requires policy.cache_dir in the config")
        batches = [batch] if loader is None else loader
        count = 0
        for cached_batch in batches:
            cached_batch = send_to_device(cached_batch, accelerator.device)
            observed, *_ = policy.encode_observation(cached_batch["observed_rgb"], cached_batch["condition_text"],
                                                      cached_batch.get("observed_valid"))
            policy.prepare_training_targets(cached_batch, observed)
            count += 1
            if count >= args.cache_batches:
                break
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"cached_batches": count, "cache_dir": cfg.policy.cache_dir}, indent=2), encoding="utf-8")
        return

    optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
    policy, optimizer, scheduler = accelerator.prepare(policy, optimizer, scheduler)
    metrics = {key: AverageMeter(key, ":.6f") for key in [
        "loss", "loss_rgb_flow", "loss_motion_flow", "loss_3d_feature", "grad_norm", "lr", "update_s", "dataloading_s"]}
    tracker = MetricsTracker(cfg.batch_size * cfg.policy.gradient_accumulation_steps,
                             num_frames, num_episodes, metrics, accelerator=accelerator)
    history = []
    for step in range(args.steps):
        # Fixed noise/time for a meaningful same-sample overfit diagnostic.
        torch.manual_seed(123)
        for _ in range(cfg.policy.gradient_accumulation_steps):
            with accelerator.accumulate(policy):
                tracker, outputs = update_policy(tracker, policy, batch, optimizer,
                                                  cfg.optimizer.grad_clip_norm, accelerator, scheduler)
        tracker.step()
        history.append(outputs)
        logging.info("server_validation step=%d metrics=%s", step + 1, outputs)

    report = {"executed_on_server": True, "synthetic": args.synthetic, "steps": args.steps, "metrics": history,
              "overfit_decreased": history[-1]["loss"] < history[0]["loss"] if args.steps > 1 else None}
    if args.reload_checkpoint:
        checkpoint = Path(cfg.output_dir) / "validation_checkpoint"
        if checkpoint.exists():
            raise FileExistsError(f"Choose a new output_dir; diagnostic checkpoint already exists: {checkpoint}")
        unwrapped = accelerator.unwrap_model(policy)
        reference = unwrapped.predict_future(batch["observed_rgb"], batch["condition_text"], seed=37,
                                              observed_valid=batch.get("observed_valid"))
        reference = {key: value.detach().cpu() for key, value in reference.items() if torch.is_tensor(value)}
        save_checkpoint(checkpoint, args.steps, cfg, unwrapped, optimizer, scheduler, data_stats)
        del unwrapped, policy, optimizer, scheduler
        accelerator.free_memory()
        restored = WSA1_5_Policy.from_pretrained(checkpoint / "pretrained_model")
        prediction = restored.predict_future(batch["observed_rgb"], batch["condition_text"], seed=37,
                                             observed_valid=batch.get("observed_valid"))
        errors = {}
        for key, value in reference.items():
            result = prediction[key].detach().cpu()
            torch.testing.assert_close(value, result, atol=1e-5, rtol=1e-4)
            errors[key] = float((value != result).sum()) if value.dtype == torch.bool else float((value - result).abs().max())
        for name, size in [("future_rgb", (32, 3, cfg.policy.rgb_height, cfg.policy.rgb_width)),
                           ("future_motion", (32, 3, cfg.policy.motion_height, cfg.policy.motion_width))]:
            if prediction[name] is not None:
                assert prediction[name].shape[1:] == size
        if prediction["goal_3d_features"] is not None:
            assert prediction["goal_3d_features"].shape[1:] == (3, 273, 2048)
        optimizer, scheduler = make_optimizer_and_scheduler(cfg, restored)
        resumed_step, optimizer, scheduler = load_training_state(checkpoint, optimizer, scheduler)
        assert resumed_step == args.steps
        restored, optimizer, scheduler = accelerator.prepare(restored, optimizer, scheduler)
        for _ in range(cfg.policy.gradient_accumulation_steps):
            with accelerator.accumulate(restored):
                tracker, resumed_metrics = update_policy(tracker, restored, batch, optimizer,
                                                         cfg.optimizer.grad_clip_norm, accelerator, scheduler)
        report.update({"reload_max_errors": errors, "resumed_step": resumed_step,
                       "post_resume_metrics": resumed_metrics, "checkpoint": str(checkpoint)})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.steps > 1 and not report["overfit_decreased"]:
        raise AssertionError("Same-sample fixed-noise loss did not decrease; see report before a large training run")
    accelerator.end_training()


if __name__ == "__main__":
    main()
