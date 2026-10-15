import inspect

import draccus
import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.configuration_wsa1_5 import WSA1_5DatasetConfig
from lerobot.datasets.factory import make_dataloader
from lerobot.datasets.transformed_dataset import MultiLeRobotDataset
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.wsa1_5.cache import ArtifactCache, fingerprint_path
from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA1_5Config
from lerobot.policies.wsa1_5.future_3d import Future3DExpert
from lerobot.policies.wsa1_5.modeling_wsa1_5 import WSA1_5_Policy, WorldComponents
from test_wsa1_5_dataset import make_data
from test_wsa1_5_experts import small_flow, inputs


def small_config(**kwargs):
    return WSA1_5Config(device="cpu", dtype="float32", push_to_hub=False,
                       rgb_hidden_dim=16, rgb_ffn_dim=32, motion_hidden_dim=12, motion_ffn_dim=24,
                       goal_hidden_dim=10, goal_ffn_dim=20, num_layers=3, num_heads=2,
                       head_dim=8, latent_channels=4, text_dim=8, freq_dim=8, **kwargs)


def small_policy(**kwargs):
    config = small_config(**kwargs)
    student = Future3DExpert(10, 20, 3, 2, 8, 8)
    # Explicit small random components only. No VAE, UMT5, Omega, weights or
    # optimizer are constructed, and pixel-space execution remains unavailable.
    return WSA1_5_Policy(config, components=WorldComponents(small_flow(), student))


def test_real_registries_and_serialization(tmp_path):
    assert PreTrainedConfig.get_choice_class("WSA_1_5") is WSA1_5Config
    assert DatasetConfig.get_choice_class("wsa1_5") is WSA1_5DatasetConfig
    assert get_policy_class("WSA_1_5") is WSA1_5_Policy
    assert make_policy_config("wsa1_5", device="cpu", push_to_hub=False).type == "WSA_1_5"
    config = small_config()
    config._save_pretrained(tmp_path)
    with draccus.config_type("json"):
        restored = draccus.parse(PreTrainedConfig, str(tmp_path / "config.json"), args=[])
    assert restored == config and restored.action_delta_indices is None
    assert not restored.output_features


def test_optimizer_groups_are_disjoint_and_complete():
    policy = small_policy()
    groups = policy.get_optim_params()
    assert {group["name"] for group in groups} == {"rgb", "motion", "3d", "heads"}
    parameters = [p for group in groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in policy.parameters() if p.requires_grad}
    frozen = small_policy(freeze_rgb=True, freeze_heads=True)
    assert {group["name"] for group in frozen.get_optim_params()} == {"motion", "3d"}
    assert set(frozen.config.get_scheduler_preset().group_schedules) == {"rgb", "motion", "3d", "heads"}


def test_joint_feature_interface_and_backward():
    policy = small_policy()
    obs, text, text_valid, rgb, motion, *_ = inputs()
    observation_valid = torch.ones(2, 1, 1, 4, 4, dtype=torch.bool)
    pixel_valid = torch.ones(2, 1, 12, 10, dtype=torch.bool)
    prediction = policy.flow.velocity(obs, text, text_valid, rgb, motion,
                                      torch.zeros(2), torch.zeros(2), observed_valid=observation_valid,
                                      future_3d=policy.future_3d, goal_view_valid=policy.observed_view_validity(pixel_valid))
    features = prediction["3d"]
    assert features[23].shape == (2, 3, 273, 2048)
    features[23].square().mean().backward()
    assert policy.future_3d.query_tokens.grad.abs().sum() > 0
    assert policy.teacher.model is None
    assert "teacher" not in policy._modules


def test_inference_accepts_no_target_fields():
    params = inspect.signature(WSA1_5_Policy.predict_future).parameters
    assert set(params) == {"self", "observed_rgb", "text", "seed", "observed_valid"}
    with pytest.raises(ValueError, match="explicit targets"):
        small_policy().training_forward({})
    with pytest.raises(ValueError, match="joint"):
        small_config(goal_conditioning="noisy_target_estimate")


def test_cache_identity_includes_content_and_metadata(tmp_path):
    cache = ArtifactCache(tmp_path)
    calls = []
    def compute():
        calls.append(1)
        return {"feature": torch.randn(3, 4)}
    identity = {"video": torch.zeros(2, 3, 4), "sidecar": "a", "template": "v1", "weights": "sha256a"}
    a = cache.get_or_compute("test", identity, compute)
    b = cache.get_or_compute("test", identity, compute)
    torch.testing.assert_close(a["feature"], b["feature"], rtol=0, atol=0)
    assert len(calls) == 1
    for key, value in [("video", torch.ones(2, 3, 4)), ("sidecar", "b"), ("template", "v2"), ("weights", "sha256b")]:
        cache.get_or_compute("test", {**identity, key: value}, compute)
    assert len(calls) == 5


def test_saved_frozen_asset_identity_does_not_require_source_weights():
    policy = small_policy(frozen_asset_fingerprints={"vae_path": "verified-vae-content-sha256",
                                                   "text_encoder_path": "verified-text-content-sha256"})
    assert policy._asset_fingerprint("vae_path") == "verified-vae-content-sha256"
    assert policy._asset_fingerprint("text_encoder_path") == "verified-text-content-sha256"


def test_weight_fingerprint_hashes_any_single_file(tmp_path):
    weight = tmp_path / "teacher.ckpt"
    weight.write_bytes(b"first version")
    before = fingerprint_path(weight)
    weight.write_bytes(b"other version")
    assert fingerprint_path(weight) != before


def test_weight_directory_requires_hashable_assets(tmp_path):
    with pytest.raises(ValueError, match="No fingerprintable"):
        fingerprint_path(tmp_path)


@pytest.mark.parametrize("path", ["", " "])
def test_empty_asset_path_cannot_hash_the_workspace(path):
    with pytest.raises(ValueError, match="empty asset path"):
        fingerprint_path(path)


def test_tokenizer_fingerprint_ignores_unrelated_files(tmp_path):
    (tmp_path / "tokenizer.json").write_text('{"version": 1}')
    preview = tmp_path / "preview.png"
    preview.write_bytes(b"preview")
    before = fingerprint_path(tmp_path)
    preview.write_bytes(b"changed preview")
    assert fingerprint_path(tmp_path) == before


def test_existing_multi_dataset_and_weighted_dataloader(tmp_path, monkeypatch):
    a, _ = make_data(tmp_path / "a", monkeypatch)
    b, _ = make_data(tmp_path / "b", monkeypatch)
    ds = MultiLeRobotDataset([a, b], dataset_weights=[0.25, 0.75])
    cfg = TrainPipelineConfig(dataset=WSA1_5DatasetConfig(repo_id=["a", "b"]), policy=small_config(),
                              num_workers=0, batch_size=2)
    loader, _ = make_dataloader(cfg, ds)
    batch = next(iter(loader))
    assert len(ds) == 80 and batch["observed_rgb"].shape[0] == 2
    assert "action" not in batch
    assert not batch["targets"]["goal_view_valid"][:, 1:].any()


def test_mixing_metadata_uses_loaded_episode_offsets(tmp_path, monkeypatch):
    ds, _ = make_data(tmp_path, monkeypatch)
    # Source metadata can still describe the full unfiltered collection.
    ds.base.meta.total_episodes = 99
    ds.base.meta.episodes["dataset_from_index"] = [1000]
    ds.base.meta.episodes["dataset_to_index"] = [1040]
    assert ds.meta.total_episodes == 1
    mixed = MultiLeRobotDataset([ds, ds], dataset_weights=[0.5, 0.5])
    assert mixed.num_episodes == 2
    assert mixed.meta.episodes == {"dataset_from_index": [0, 40], "dataset_to_index": [40, 80]}


def test_factory_rejects_incompatible_dataset():
    cfg = TrainPipelineConfig(dataset=WSA1_5DatasetConfig(repo_id="x"), policy=make_policy_config("pi0", device="cpu"))
    # Test the early factory mismatch before any metadata/file access.
    from lerobot.datasets.factory import _build_single_dataset
    with pytest.raises(ValueError, match="WSA_1_5"):
        _build_single_dataset(cfg, "x", None, 0)
