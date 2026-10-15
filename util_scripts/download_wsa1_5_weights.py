"""Download WSA1.5 assets from ModelScope, without importing or loading a model.

The SHA256 catalog was checked against ModelScope file metadata on 2026-10-15.
Only the original Wan TI2V-5B and Omega 256 Text-Alignment files are accepted.
--dry-run and --verify-only require only the Python standard library.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import tempfile


@dataclass(frozen=True)
class Asset:
    path: str
    size: int
    sha256: str


CATALOG = {
    "wan": (
        Asset("diffusion_pytorch_model-00001-of-00003.safetensors", 9825014472,
              "720b06c4ade5e87c1246bba8ac95b664c638749cd9b102cf84d823bb44c026a1"),
        Asset("diffusion_pytorch_model-00002-of-00003.safetensors", 9995661736,
              "09ec5ef720d8396f6cfa51fbdcbdb2327e37722afd6e89fd38f1e7e5e782c283"),
        Asset("diffusion_pytorch_model-00003-of-00003.safetensors", 178558176,
              "6306f7894c345de9093ad588771c2abfaeb668a81f7a6d9a918bd26ba3568e49"),
        Asset("diffusion_pytorch_model.safetensors.index.json", 72865,
              "bfa2337f1163e195d24151a72298daf34a620543898109be47e414c8daa5b3fe"),
        Asset("Wan2.2_VAE.pth", 2818839170,
              "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"),
        Asset("models_t5_umt5-xxl-enc-bf16.pth", 11361920418,
              "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"),
        Asset("google/umt5-xxl/spiece.model", 4548313,
              "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"),
        Asset("google/umt5-xxl/tokenizer.json", 16837417,
              "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b"),
        Asset("google/umt5-xxl/tokenizer_config.json", 61728,
              "ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05"),
        Asset("google/umt5-xxl/special_tokens_map.json", 6623,
              "7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d"),
    ),
    "omega": (
        Asset("vggt_omega_1b_256_text.pt", 5399490950,
              "2c59393e2d15569498d42191e03c29aabc72bdb243a20870b22a0b24a8f8095c"),
    ),
}

DEFAULT_REPOS = {"wan": "Wan-AI/Wan2.2-TI2V-5B", "omega": "facebook/VGGT-Omega"}
DIRECTORIES = {"wan": "Wan2.2-TI2V-5B", "omega": "VGGT-Omega-1B-256-Text-Alignment"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(directory, asset):
    path = directory / asset.path
    if not path.is_file():
        raise FileNotFoundError(f"Missing asset: {path}")
    if path.stat().st_size != asset.size:
        raise RuntimeError(f"Asset size mismatch: {path}; expected {asset.size} bytes")
    if sha256_file(path) != asset.sha256:
        raise RuntimeError(f"Asset SHA256 mismatch: {path}; refusing to use or overwrite this file")


def make_plan(args):
    root = args.output_root.expanduser().resolve()
    plans = []
    for component in dict.fromkeys(args.components):
        model_id = getattr(args, f"{component}_model_id")
        revision = getattr(args, f"{component}_revision")
        if len(model_id.split("/")) != 2 or any(not part or part in {".", ".."} for part in model_id.split("/")):
            raise ValueError("ModelScope model IDs must be owner/model")
        if not revision.strip():
            raise ValueError("ModelScope revision must not be blank")
        plans.append({"component": component, "model_id": model_id, "revision": revision,
                      "directory": str(root / DIRECTORIES[component]),
                      "files": [asset.__dict__ for asset in CATALOG[component]]})
    return root, plans


def atomic_write(path, content):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_outputs(root, plans):
    exports = {}
    for plan in plans:
        directory = Path(plan["directory"])
        if plan["component"] == "wan":
            exports.update({"WAN_MODEL_PATH": directory, "VAE_PATH": directory / "Wan2.2_VAE.pth",
                            "TEXT_ENCODER_PATH": directory / "models_t5_umt5-xxl-enc-bf16.pth",
                            "TOKENIZER_PATH": directory / "google/umt5-xxl"})
        else:
            exports["OMEGA_CHECKPOINT_PATH"] = directory / "vggt_omega_1b_256_text.pt"
    env = "# Generated after SHA256 verification. Paths are for Bash on this machine.\n"
    env += "".join(f"export {name}={shlex.quote(str(path))}\n" for name, path in exports.items())
    atomic_write(root / "wsa1_5_weights.env", env)
    manifest = {"source": "modelscope", "catalog_date": "2026-10-15", "assets": plans}
    atomic_write(root / "wsa1_5_weights_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def download(plans, max_workers):
    snapshot_download = None
    for plan in plans:
        directory = Path(plan["directory"])
        assets = CATALOG[plan["component"]]
        missing = []
        for asset in assets:
            if (directory / asset.path).exists():
                verify_asset(directory, asset)
            else:
                missing.append(asset)
        if not missing:
            print(f"Already verified: {plan['component']} ({directory})")
            continue
        if snapshot_download is None:
            try:
                from modelscope.hub.snapshot_download import snapshot_download
            except ModuleNotFoundError as error:
                raise RuntimeError("Install the download extra: pip install -e '.[wsa1_5_download]'") from error
        directory.mkdir(parents=True, exist_ok=True)
        # Never fetch unrelated checkpoints, images, demos or Markdown files.
        snapshot_download(model_id=plan["model_id"], revision=plan["revision"], local_dir=str(directory),
                          allow_file_pattern=[asset.path for asset in missing], max_workers=max_workers,
                          token=os.environ.get("MODELSCOPE_API_TOKEN"))
        for asset in missing:
            verify_asset(directory, asset)
        print(f"Downloaded and verified: {plan['component']} ({directory})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("weights"))
    parser.add_argument("--components", choices=tuple(CATALOG), nargs="+", default=list(CATALOG))
    parser.add_argument("--max-workers", type=int, default=4)
    for component in CATALOG:
        parser.add_argument(f"--{component}-model-id", default=DEFAULT_REPOS[component])
        parser.add_argument(f"--{component}-revision", default="master")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print the plan without network access or writing files")
    mode.add_argument("--verify-only", action="store_true", help="Offline SHA256 verification; no writes or downloads")
    args = parser.parse_args(argv)
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    try:
        root, plans = make_plan(args)
        if args.dry_run:
            print(json.dumps({"source": "modelscope", "assets": plans,
                              "total_bytes": sum(asset["size"] for plan in plans for asset in plan["files"])},
                             ensure_ascii=False, indent=2))
            return 0
        if args.verify_only:
            for plan in plans:
                for asset in CATALOG[plan["component"]]:
                    verify_asset(Path(plan["directory"]), asset)
            print("All selected assets passed offline SHA256 verification")
            return 0
        download(plans, args.max_workers)
        root.mkdir(parents=True, exist_ok=True)
        write_outputs(root, plans)
        print(f"Bash environment: {root / 'wsa1_5_weights.env'}")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
