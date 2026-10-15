"""Download contracts use tiny byte fixtures; no network, models or real weights."""
import builtins
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from util_scripts import download_wsa1_5_weights as downloader


@pytest.fixture
def tiny_catalog(monkeypatch):
    content = {"wan": b"test-wan-data", "omega": b"test-omega-data"}
    catalog = {name: (downloader.Asset(f"{name}.bin", len(value), hashlib.sha256(value).hexdigest()),)
               for name, value in content.items()}
    monkeypatch.setattr(downloader, "CATALOG", catalog)
    return content


def test_dry_run_does_not_import_sdk_or_create_files(tmp_path, monkeypatch, capsys):
    original = builtins.__import__

    def forbid_models(name, *args, **kwargs):
        if name.split(".")[0] in {"modelscope", "torch", "transformers"}:
            pytest.fail(f"Dry-run imported {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_models)
    root = tmp_path / "not-created"
    assert downloader.main(["--output-root", str(root), "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert [item["model_id"] for item in plan["assets"]] == [
        "Wan-AI/Wan2.2-TI2V-5B", "facebook/VGGT-Omega"]
    assert plan["total_bytes"] == sum(asset.size for assets in downloader.CATALOG.values() for asset in assets)
    assert plan["assets"][1]["files"][0]["path"] == "vggt_omega_1b_256_text.pt"
    assert not root.exists()


def test_verify_only_is_offline_and_read_only(tmp_path, monkeypatch, tiny_catalog):
    for component, content in tiny_catalog.items():
        directory = tmp_path / downloader.DIRECTORIES[component]
        directory.mkdir()
        (directory / f"{component}.bin").write_bytes(content)
    monkeypatch.setattr(downloader, "download", lambda *args: pytest.fail("Must not download"))
    env = tmp_path / "wsa1_5_weights.env"
    env.write_text("preserve existing environment")
    assert downloader.main(["--output-root", str(tmp_path), "--verify-only"]) == 0
    assert env.read_text() == "preserve existing environment"
    assert not (tmp_path / "wsa1_5_weights_manifest.json").exists()


@pytest.mark.parametrize("contents", [b"short", b"x" * len(b"test-wan-data")])
def test_corrupt_existing_files_are_not_overwritten(tmp_path, tiny_catalog, contents):
    directory = tmp_path / downloader.DIRECTORIES["wan"]
    directory.mkdir()
    path = directory / "wan.bin"
    path.write_bytes(contents)
    with pytest.raises(SystemExit) as error:
        downloader.main(["--output-root", str(tmp_path), "--components", "wan"])
    assert error.value.code == 1 and path.read_bytes() == contents
    assert not (tmp_path / "wsa1_5_weights.env").exists()


def test_download_only_missing_assets_and_validate_sdk_output(tmp_path, monkeypatch, tiny_catalog):
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        component = "wan" if kwargs["model_id"] == downloader.DEFAULT_REPOS["wan"] else "omega"
        assert kwargs["allow_file_pattern"] == [f"{component}.bin"]
        (Path(kwargs["local_dir"]) / f"{component}.bin").write_bytes(tiny_catalog[component])
        return kwargs["local_dir"]

    for name in ["modelscope", "modelscope.hub", "modelscope.hub.snapshot_download"]:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys.modules["modelscope.hub.snapshot_download"], "snapshot_download", snapshot_download, raising=False)
    assert downloader.main(["--output-root", str(tmp_path)]) == 0
    assert len(calls) == 2 and all(call["max_workers"] == 4 for call in calls)
    env = (tmp_path / "wsa1_5_weights.env").read_text()
    assert "export WAN_MODEL_PATH=" in env and "export OMEGA_CHECKPOINT_PATH=" in env
    manifest = json.loads((tmp_path / "wsa1_5_weights_manifest.json").read_text())
    assert manifest["source"] == "modelscope" and len(manifest["assets"]) == 2
    assert downloader.main(["--output-root", str(tmp_path)]) == 0
    assert len(calls) == 2  # Already verified files must not be downloaded again.


def test_download_failure_does_not_create_success_manifest(tmp_path, monkeypatch, tiny_catalog):
    def fail(*args):
        raise PermissionError("repository access denied")
    monkeypatch.setattr(downloader, "download", fail)
    with pytest.raises(SystemExit) as error:
        downloader.main(["--output-root", str(tmp_path)])
    assert error.value.code == 1
    assert not (tmp_path / "wsa1_5_weights.env").exists()
    assert not (tmp_path / "wsa1_5_weights_manifest.json").exists()


def test_corrupt_download_is_rejected_before_environment_export(tmp_path, monkeypatch, tiny_catalog):
    def corrupt_snapshot(**kwargs):
        directory = Path(kwargs["local_dir"])
        (directory / "wan.bin").write_bytes(b"x" * len(tiny_catalog["wan"]))

    module = ModuleType("modelscope.hub.snapshot_download")
    module.snapshot_download = corrupt_snapshot
    monkeypatch.setitem(sys.modules, "modelscope.hub.snapshot_download", module)
    with pytest.raises(SystemExit) as error:
        downloader.main(["--output-root", str(tmp_path), "--components", "wan"])
    assert error.value.code == 1
    assert not (tmp_path / "wsa1_5_weights.env").exists()
    assert not (tmp_path / "wsa1_5_weights_manifest.json").exists()


@pytest.mark.parametrize("args", [["--max-workers", "0"], ["--wan-model-id", "bad-id"],
                                 ["--omega-revision", ""], ["--dry-run", "--verify-only"]])
def test_invalid_cli_values_rejected(args, tmp_path):
    with pytest.raises(SystemExit):
        downloader.main(["--output-root", str(tmp_path), *args])
    assert not list(tmp_path.iterdir())
