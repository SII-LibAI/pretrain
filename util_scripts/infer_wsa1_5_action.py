"""Server-only chunk inference from a saved, observation-only tensor batch."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True, help="One dataset, for actual schema/statistics")
    parser.add_argument("--external-stats-path", type=Path, help="Original plain per-feature stats.json")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-world", action="store_true", help="Decode the same rollout used by Action; no second plan")
    args = parser.parse_args()

    import torch
    from accelerate.utils import send_to_device

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import _apply_dataset_stats_overrides
    from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA1_5Config
    from lerobot.policies.wsa1_5.dataset_wsa1_5 import ActionCodec, action_prompt
    from lerobot.policies.wsa1_5.modeling_wsa1_5 import WSA1_5_Policy
    from lerobot.policies.wsa1_5.text_embeddings import TextEmbeddingCache

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = args.device
    if not isinstance(cfg, WSA1_5Config) or not cfg.enable_action:
        raise ValueError("Action inference requires a Stage-2 checkpoint")
    info = json.loads((args.dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    stats = json.loads((args.dataset_root / "meta/stats.json").read_text(encoding="utf-8"))
    saved_stats = args.checkpoint / "stats.json"
    if args.external_stats_path is None and saved_stats.is_file():
        stats = json.loads(saved_stats.read_text(encoding="utf-8"))[info["robot_type"]]
    meta = SimpleNamespace(robot_type=info["robot_type"], features=info["features"], stats=stats)
    if args.external_stats_path is not None or not saved_stats.is_file():
        data_cfg = SimpleNamespace(use_imagenet_stats=False, use_external_stats=True,
                                   external_stats_path=args.external_stats_path, action_mode=cfg.action_mode)
        _apply_dataset_stats_overrides(SimpleNamespace(dataset=data_cfg), SimpleNamespace(meta=meta))
    codec = ActionCodec(meta, cfg)
    inputs = torch.load(args.input, map_location="cpu", weights_only=True)
    tasks, raw_state = inputs["task"], inputs["raw_state"]
    if not isinstance(tasks, list) or len(tasks) != inputs["observed_rgb"].shape[0]:
        raise ValueError("input.task must be one instruction per batch item")
    states = [codec.encode({key: raw_state[key][i] for key in codec.state_keys})["state"] for i in range(len(tasks))]
    batch = {key: inputs[key] for key in ["observed_rgb", "observed_valid"]}
    if cfg.num_history_frame:
        batch.update(history_rgb=inputs["history_rgb"], history_valid=inputs["history_valid"])
    batch["state"] = torch.stack(states)
    batch["condition_text"] = [action_prompt(task, codec.control_mode, cfg.action_mode) for task in tasks]
    if not cfg.load_text_encoder:
        cache = TextEmbeddingCache(args.text_cache or args.dataset_root / "meta/wsa1_5_text_embeddings",
                                   cfg.max_text_length, cfg.text_dim)
        pairs = [cache.get(text) for text in batch["condition_text"]]
        batch.update(text_embedding=torch.stack([p[0] for p in pairs]), text_valid=torch.stack([p[1] for p in pairs]))
    policy = WSA1_5_Policy.from_pretrained(args.checkpoint, config=cfg)
    prediction = policy.predict_action_chunk(send_to_device(batch, args.device), codec=codec,
                                             raw_state=send_to_device(raw_state, args.device), seed=args.seed,
                                             return_world=args.include_world)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(send_to_device(prediction, "cpu"), args.output)


if __name__ == "__main__":
    main()
