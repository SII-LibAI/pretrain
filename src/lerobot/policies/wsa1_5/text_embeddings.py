"""The single WSA1.5 cache: frozen, offline UMT5 text embeddings."""
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

PROMPT_VERSION = "wsa1_5.task-subtask-hand-scene.v1"
PROMPT_PREFIX = "You are a robot control agent,follow the instruction."


def text_embedding_path(directory, text):
    return Path(directory) / f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}.safetensors"


def validate_text_embedding(context, valid, max_length, text_dim):
    if context.ndim != 3 or context.shape[1:] != (max_length, text_dim):
        raise ValueError("Text embedding must be [B,max_text_length,text_dim]")
    if valid.dtype != torch.bool or valid.shape != context.shape[:2] or not valid.any(1).all():
        raise ValueError("Text validity must be boolean [B,max_text_length] with a valid token per sample")
    if not context.is_floating_point() or not torch.isfinite(context).all():
        raise ValueError("Text embeddings must contain finite floating-point values")


class TextEmbeddingCache:
    def __init__(self, directory, max_length, text_dim):
        self.directory = Path(directory)
        self.max_length, self.text_dim = max_length, text_dim

    def require(self, texts):
        missing = [text_embedding_path(self.directory, text) for text in set(texts)
                   if not text_embedding_path(self.directory, text).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} offline text embeddings, e.g. {missing[0]}. "
                                    "Run util_scripts/precompute_wsa1_5_text_embeddings.py on the server first.")

    def get(self, text):
        path = text_embedding_path(self.directory, text)
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            expected = {"prompt_version": PROMPT_VERSION, "text": text,
                        "max_length": str(self.max_length), "text_dim": str(self.text_dim)}
            if metadata != expected or set(handle.keys()) != {"context", "valid"}:
                raise ValueError(f"Text embedding metadata/keys mismatch: {path}")
            context, valid = handle.get_tensor("context"), handle.get_tensor("valid")
        if (context.dtype != torch.bfloat16 or context.ndim != 2 or context.shape[1] != self.text_dim
                or not 0 < context.shape[0] <= self.max_length or valid.shape != context.shape[:1]
                or valid.dtype != torch.bool):
            raise ValueError(f"Invalid BF16 text embedding shape/mask: {path}")
        padding = self.max_length - context.shape[0]
        context, valid = F.pad(context, (0, 0, 0, padding)), F.pad(valid, (0, padding))
        validate_text_embedding(context[None], valid[None], self.max_length, self.text_dim)
        return context.masked_fill(~valid[:, None], 0), valid


@torch.no_grad()
def encode_text(encoder, tokenizer, texts, device, max_length, text_dim):
    ids, valid = tokenizer(texts, return_mask=True, add_special_tokens=True)
    if not torch.isfinite(valid).all() or not ((valid == 0) | (valid == 1)).all():
        raise ValueError("Tokenizer attention mask must contain only finite binary values")
    ids, valid = ids.to(device), valid.to(device).bool()
    context = encoder(ids, valid).masked_fill(~valid[..., None], 0)
    validate_text_embedding(context, valid, max_length, text_dim)
    return context, valid
