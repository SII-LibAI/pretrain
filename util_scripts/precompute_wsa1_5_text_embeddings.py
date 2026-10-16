"""在 GPU 服务器为每个 LeRobot 数据集预计算唯一一种 WSA1.5 文本缓存。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from safetensors.torch import save_file

from lerobot.policies.wsa1_5.dataset_wsa1_5 import condition_text
from lerobot.policies.wsa1_5.text_embeddings import PROMPT_VERSION, encode_text, text_embedding_path


def dataset_prompts(root, sidecar_file="meta/sidecar.json"):
    path = Path(root) / sidecar_file
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if not isinstance(rows, list):
        raise ValueError(f"Sidecar must be a list: {path}")
    prompts = {""}  # Unannotated frames use the same explicit blank condition as training.
    for row in rows:
        if not isinstance(row, dict) or row.get("episode_index") is None:
            continue
        subtasks = row.get("subtasks")
        if subtasks is not None and not isinstance(subtasks, list):
            raise ValueError(f"subtasks must be a list or null: {path}")
        for subtask in subtasks or []:
            required = ["start_frame", "end_frame", "subtask_id"]
            if isinstance(subtask, dict) and all(subtask.get(key) is not None for key in required):
                prompts.add(condition_text(row, subtask))
    return sorted(prompts)


@torch.no_grad()
def precompute_dataset(root, encoder, tokenizer, device, *, max_length=512, text_dim=4096,
                       sidecar_file="meta/sidecar.json", embedding_dir="meta/wsa1_5_text_embeddings"):
    directory = Path(root) / embedding_dir
    directory.mkdir(parents=True, exist_ok=True)
    prompts = dataset_prompts(root, sidecar_file)
    for text in prompts:
        context, valid = encode_text(encoder, tokenizer, [text], device, max_length, text_dim)
        context = context[0].to(device="cpu", dtype=torch.bfloat16)
        valid = valid[0].cpu()
        if not torch.isfinite(context).all():
            raise FloatingPointError("Non-finite UMT5 embedding after BF16 conversion")
        length = valid.nonzero()[-1].item() + 1
        metadata = {"prompt_version": PROMPT_VERSION, "text": text,
                    "max_length": str(max_length), "text_dim": str(text_dim)}
        path = text_embedding_path(directory, text)
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            save_file({"context": context[:length].contiguous(), "valid": valid[:length].contiguous()},
                      temporary, metadata=metadata)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"[WSA text] {root}: {len(prompts)} prompts -> {directory}", flush=True)
    return len(prompts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="单个数据集或含任意层级数据集的目录")
    parser.add_argument("--text-encoder-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-text-length", type=int, default=512)
    parser.add_argument("--sidecar-file", default="meta/sidecar.json")
    parser.add_argument("--embedding-dir", default="meta/wsa1_5_text_embeddings")
    args = parser.parse_args()
    if args.max_text_length <= 0:
        parser.error("--max-text-length must be positive")
    repos = subprocess.check_output(
        ["bash", str(PROJECT_ROOT / "util_scripts/discover_wsa1_5_datasets.sh"), str(args.data_root)],
        encoding="utf-8",
    ).split()
    for path in [args.text_encoder_path, args.tokenizer_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    from lerobot.policies.wsa1_5.initialization import load_report, read_checkpoint
    from lerobot.policies.wsa1_5.modeling_wsa1_5 import construction_dtype
    from lerobot.policies.wsa1_5.vendor.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder

    with construction_dtype(torch.bfloat16):
        encoder = WanTextEncoder()
    load_report(encoder, read_checkpoint(args.text_encoder_path), name="offline Wan UMT5", strict=True)
    encoder.to(device=args.device, dtype=torch.bfloat16).eval().requires_grad_(False)
    tokenizer = HuggingfaceTokenizer(str(args.tokenizer_path), seq_len=args.max_text_length,
                                    clean="whitespace", local_files_only=True)
    for repo in repos:
        precompute_dataset(args.data_root / repo, encoder, tokenizer, args.device,
                           max_length=args.max_text_length, sidecar_file=args.sidecar_file,
                           embedding_dir=args.embedding_dir)


if __name__ == "__main__":
    main()
