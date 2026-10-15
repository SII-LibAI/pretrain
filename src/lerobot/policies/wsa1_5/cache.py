"""Content-addressed frozen-artifact cache with atomic, verified metadata."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PIPELINE_VERSION = "wsa1_5.v2.wan32211e4.omega48b23c8.imagenet.33frames.fp32flow"


def fingerprint_path(path):
    if not path or not str(path).strip():
        raise ValueError("Cannot fingerprint an empty asset path")
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    # Directory filters exclude unrelated documents; a named weight file always counts.
    files = [root]
    if root.is_dir():
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {
            ".json", ".safetensors", ".pth", ".pt", ".bin", ".model", ".txt"})
    if not files:
        raise ValueError(f"No fingerprintable assets in {root}")
    digest = hashlib.sha256()
    for file in files:
        digest.update((file.relative_to(root).as_posix() if root.is_dir() else file.name).encode())
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def identity_value(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous()
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(k): identity_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [identity_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


class ArtifactCache:
    def __init__(self, root):
        self.root = Path(root)

    @staticmethod
    def _validate(tensors):
        if not isinstance(tensors, dict) or not tensors or not all(torch.is_tensor(v) for v in tensors.values()):
            raise TypeError("Frozen artifact cache requires a non-empty tensor dictionary")
        if any(not torch.isfinite(value).all() for value in tensors.values()):
            raise FloatingPointError("Frozen artifact contains non-finite values")

    @staticmethod
    def _content_manifest(tensors):
        return json.dumps(identity_value(tensors), sort_keys=True, separators=(",", ":"))

    def get_or_compute(self, kind, identity, compute):
        manifest = json.dumps({"pipeline": PIPELINE_VERSION, "kind": kind, "identity": identity_value(identity)},
                              sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        key = hashlib.sha256(manifest.encode()).hexdigest()
        directory = self.root / kind
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{key}.safetensors"
        if path.exists():
            with safe_open(str(path), framework="pt", device="cpu") as reader:
                if reader.metadata().get("manifest") != manifest:
                    raise RuntimeError(f"Cache manifest mismatch: {path}")
                tensors = {name: reader.get_tensor(name) for name in reader.keys()}
                self._validate(tensors)
                if reader.metadata().get("content") != self._content_manifest(tensors):
                    raise RuntimeError(f"Cache tensor checksum/schema mismatch: {path}")
                return tensors
        result = compute()
        self._validate(result)
        tensors = {name: value.detach().cpu().clone().contiguous() for name, value in result.items()}
        descriptor, temporary_path = tempfile.mkstemp(prefix=f"{key}.", suffix=".tmp", dir=directory)
        os.close(descriptor)
        try:
            save_file(tensors, temporary_path, metadata={"manifest": manifest, "content": self._content_manifest(tensors)})
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        return tensors
