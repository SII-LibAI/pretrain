"""在 GPU 服务器为每个 LeRobot 数据集预计算唯一一种 WSA1.5 文本缓存。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
"""
run:
WAN_MODEL_PATH=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/Wan2.2-TI2V-5B
DATA_ROOT=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series/data/ego/Egodex-LerobotV3
cd /inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/InternVLA-A-series
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  util_scripts/precompute_wsa1_5_text_embeddings.py \
  --data-root "$DATA_ROOT" \
  --text-encoder-path "$WAN_MODEL_PATH/models_t5_umt5-xxl-enc-bf16.pth" \
  --tokenizer-path "$WAN_MODEL_PATH/google/umt5-xxl" \
  --batch-size 128

check
python - "$DATA_ROOT" <<'PY'
import random
import sys
from pathlib import Path

import torch
from safetensors import safe_open

files = list(Path(sys.argv[1]).glob(
    "**/meta/wsa1_5_text_embeddings/*.safetensors"
))
if not files:
    raise SystemExit("没有找到文本 embedding 缓存")

path = random.choice(files)
print("文件：", path)

with safe_open(str(path), framework="pt", device="cpu") as f:
    print("\n元数据：")
    for key, value in (f.metadata() or {}).items():
        print(f"{key}: {value}")

    print("\n张量：")
    for key in f.keys():
        tensor = f.get_tensor(key)
        print(f"{key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

    context = f.get_tensor("context")
    valid = f.get_tensor("valid")
    values = context[valid].float()

    print("\n有效 token 数：", valid.sum().item())
    print("context 全部有限：", torch.isfinite(context).all().item())
    if values.numel():
        print(
            f"有效 embedding：mean={values.mean().item():.6f}, "
            f"std={values.std(unbiased=False).item():.6f}, "
            f"min={values.min().item():.6f}, "
            f"max={values.max().item():.6f}"
        )
PY
"""

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
from lerobot.policies.wsa1_5.text_embeddings import PROMPT_VERSION, TextEmbeddingCache, encode_text, text_embedding_path


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


def pending_prompts(directory, prompts, max_length, text_dim):
    cache = TextEmbeddingCache(directory, max_length, text_dim)
    pending = []
    for text in prompts:
        if text_embedding_path(directory, text).exists():
            cache.get(text)  # Validate existing files with the same rules used by training.
        else:
            pending.append(text)
    return pending


@torch.no_grad()
def precompute_dataset(root, encoder, tokenizer, device, *, max_length=512, text_dim=4096,
                       sidecar_file="meta/sidecar.json", embedding_dir="meta/wsa1_5_text_embeddings", batch_size=8,
                       prompts=None):
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Text encoding batch_size must be a positive integer")
    directory = Path(root) / embedding_dir
    directory.mkdir(parents=True, exist_ok=True)
    prompts = dataset_prompts(root, sidecar_file) if prompts is None else prompts
    total = len(prompts)
    prompts = pending_prompts(directory, prompts, max_length, text_dim)
    for offset in range(0, len(prompts), batch_size):
        texts = prompts[offset:offset + batch_size]
        context, valid = encode_text(encoder, tokenizer, texts, device, max_length, text_dim)
        context = context.to(device="cpu", dtype=torch.bfloat16)
        valid = valid.cpu()
        if not torch.isfinite(context).all():
            raise FloatingPointError("Non-finite UMT5 embedding after BF16 conversion")
        for text, embedding, mask in zip(texts, context, valid, strict=True):
            length = mask.nonzero()[-1].item() + 1
            metadata = {"prompt_version": PROMPT_VERSION, "text": text,
                        "max_length": str(max_length), "text_dim": str(text_dim)}
            path = text_embedding_path(directory, text)
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
            try:
                save_file({"context": embedding[:length].contiguous(), "valid": mask[:length].contiguous()},
                          temporary, metadata=metadata)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    print(f"[WSA text] {root}: computed={len(prompts)} skipped={total - len(prompts)} -> {directory}", flush=True)
    return len(prompts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="单个数据集或含任意层级数据集的目录")
    parser.add_argument("--text-encoder-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda", help="单卡可指定 cuda:0；torchrun 多卡使用 cuda 自动绑定 LOCAL_RANK")
    parser.add_argument("--max-text-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8, help="每批编码的文本数量")
    parser.add_argument("--sidecar-file", default="meta/sidecar.json")
    parser.add_argument("--embedding-dir", default="meta/wsa1_5_text_embeddings")
    args = parser.parse_args()
    if args.max_text_length <= 0:
        parser.error("--max-text-length must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    try:
        rank, world_size = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        parser.error("RANK/WORLD_SIZE must be integers")
    if world_size < 1 or not 0 <= rank < world_size:
        parser.error("Require WORLD_SIZE >= 1 and 0 <= RANK < WORLD_SIZE")
    if world_size > 1:
        if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
            parser.error("Multi-GPU execution requires torchrun RANK and LOCAL_RANK")
        if args.device != "cuda":
            parser.error("Use --device cuda with torchrun; each process binds to its LOCAL_RANK")
        try:
            local_rank = int(os.environ["LOCAL_RANK"])
        except ValueError:
            parser.error("LOCAL_RANK must be a non-negative integer")
        if local_rank < 0:
            parser.error("LOCAL_RANK must be a non-negative integer")
        args.device = f"cuda:{local_rank}"
    repos = subprocess.check_output(
        ["bash", str(PROJECT_ROOT / "util_scripts/discover_wsa1_5_datasets.sh"), str(args.data_root)],
        encoding="utf-8",
    ).split()
    # Discovery shuffles paths; every rank must partition the same ordered tasks.
    roots = sorted({(args.data_root / repo).resolve() for repo in repos})
    work, offset, skipped = [], 0, 0
    for root in roots:
        prompts = dataset_prompts(root, args.sidecar_file)
        assigned = [text for index, text in enumerate(prompts, start=offset) if index % world_size == rank]
        offset += len(prompts)
        pending = pending_prompts(root / args.embedding_dir, assigned, args.max_text_length, 4096)
        skipped += len(assigned) - len(pending)
        if pending:
            work.append((root, pending))
    print(f"[WSA text rank {rank}/{world_size}] pending={sum(len(texts) for _, texts in work)} "
          f"skipped={skipped} global_total={offset} "
          f"across {len(work)} datasets; device={args.device}", flush=True)
    if not work:
        return
    for path in [args.text_encoder_path, args.tokenizer_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    if world_size > 1:
        torch.cuda.set_device(local_rank)
    from lerobot.policies.wsa1_5.initialization import load_report, read_checkpoint
    from lerobot.policies.wsa1_5.modeling_wsa1_5 import construction_dtype
    from lerobot.policies.wsa1_5.vendor.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder

    with construction_dtype(torch.bfloat16):
        encoder = WanTextEncoder()
    load_report(encoder, read_checkpoint(args.text_encoder_path), name="offline Wan UMT5", strict=True)
    encoder.to(device=args.device, dtype=torch.bfloat16).eval().requires_grad_(False)
    tokenizer = HuggingfaceTokenizer(str(args.tokenizer_path), seq_len=args.max_text_length,
                                    clean="whitespace", local_files_only=True)
    for root, prompts in work:
        precompute_dataset(root, encoder, tokenizer, args.device,
                           max_length=args.max_text_length, sidecar_file=args.sidecar_file,
                           embedding_dir=args.embedding_dir, batch_size=args.batch_size, prompts=prompts)
    print(f"[WSA text rank {rank}/{world_size}] done", flush=True)


if __name__ == "__main__":
    main()
