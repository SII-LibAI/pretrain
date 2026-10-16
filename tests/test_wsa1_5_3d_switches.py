"""Independent G queries and optional teacher supervision; no real weights."""
import pytest
import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.configuration_wsa1_5 import WSA1_5DatasetConfig
from lerobot.datasets.factory import _build_single_dataset
from lerobot.datasets.wsa1_5 import WSA1_5Dataset, wsa1_5_collate
from lerobot.policies.wsa1_5.configuration_wsa1_5 import WSA1_5Config
from test_wsa1_5_dataset import make_data
from test_wsa1_5_observation_encoding import make_policy


@pytest.mark.parametrize("queries", [True, False])
@pytest.mark.parametrize("weight", [1.0, 0.0])
def test_3d_supervision_requires_queries_switch_and_positive_weight(queries, weight):
    cfg = WSA1_5Config(device="cpu", enable_3d_queries=queries, lambda_3d=weight)
    assert cfg.enable_3d_queries is queries
    assert cfg.use_3d_supervision == (queries and weight > 0)


@pytest.mark.parametrize("legacy", [True, False])
def test_legacy_enable_3d_normalizes_into_query_switch(legacy):
    cfg = WSA1_5Config(device="cpu", enable_3d=legacy)
    assert cfg.enable_3d_queries is legacy and cfg.enable_3d is None


@pytest.mark.parametrize("kwargs", [{"enable_3d_queries": False},
                                   {"lambda_3d": 0}, {"feature_mse_weight": 0, "feature_cosine_weight": 0}])
def test_no_supervised_modality_is_rejected(kwargs):
    with pytest.raises(ValueError):
        WSA1_5Config(device="cpu", enable_rgb=False, enable_motion=False, **kwargs)


@pytest.mark.parametrize("kwargs,queries", [
    ({"enable_3d_queries": True, "lambda_3d": 0}, True),
    ({"enable_3d_queries": True, "feature_mse_weight": 0, "feature_cosine_weight": 0}, True),
    ({"enable_3d_queries": False}, False),
])
def test_disabled_supervision_skips_teacher_targets_and_loss(monkeypatch, kwargs, queries):
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled 3D supervision must not create a teacher or compute feature loss")

    module = "lerobot.policies.wsa1_5.modeling_wsa1_5"
    monkeypatch.setattr(f"{module}.OmegaTeacher", forbidden)
    monkeypatch.setattr(f"{module}.geometric_feature_loss", forbidden)
    policy, _ = make_policy(monkeypatch, **kwargs)
    assert policy.teacher is None
    assert (policy.future_3d is not None) == queries
    assert {group["name"] for group in policy.get_optim_params()} == {"rgb", "motion"}
    if queries:
        assert not any(p.requires_grad for p in policy.future_3d.parameters())
    # Endpoint images/masks are deliberately absent, not zero placeholders.
    batch = {
        "observed_rgb": torch.randn(1, 3, 96, 32), "condition_text": ["move"],
        "observed_valid": torch.ones(1, 1, 96, 32, dtype=torch.bool),
        "targets": {"future_rgb": torch.randn(1, 32, 3, 96, 32),
                    "rgb_valid": torch.ones(1, 32, dtype=torch.bool),
                    "future_motion": torch.randn(1, 32, 3, 32, 32),
                    "motion_valid": torch.ones(1, 32, dtype=torch.bool)},
    }
    groups = []
    policy.flow.mot.register_forward_pre_hook(lambda module, args: groups.append(set(args[0])))
    loss, metrics = policy.training_forward(batch)
    assert groups == [{"O", "R", "M"} | ({"G"} if queries else set())]
    assert "loss_3d_feature" not in metrics and not any(name.startswith("3d/") for name in metrics)
    loss.backward()
    for expert in [policy.flow.rgb, policy.flow.motion]:
        grad = expert.patch_embedding.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    policy.config.sampler_steps = 1
    result = policy.predict_future(batch["observed_rgb"], batch["condition_text"], observed_valid=batch["observed_valid"])
    assert len(groups) == 2 and groups[-1] == groups[0]
    assert (result["goal_3d_features"] is not None) == queries
    if queries:
        assert result["goal_3d_features"].shape == (1, 3, 273, 2048)
    else:
        assert not result["goal_3d_valid"].any()
    assert result["future_rgb"].shape == (1, 32, 3, 96, 32)
    assert result["future_motion"].shape == (1, 32, 3, 32, 32)


@pytest.mark.parametrize("weight", [0.0, 1.0])
@pytest.mark.parametrize("return_features", [False, True])
def test_query_sampling_is_independent_of_feature_return(monkeypatch, weight, return_features):
    policy, _ = make_policy(monkeypatch, enable_3d_queries=True, lambda_3d=weight)
    pixel_valid = torch.ones(1, 1, 96, 32, dtype=torch.bool)
    groups = []
    policy.flow.mot.register_forward_pre_hook(lambda module, args: groups.append(set(args[0])))
    with torch.no_grad():
        observed, context, context_valid, observed_valid, _ = policy.encode_observation(
            torch.zeros(1, 3, 96, 32), ["move"], pixel_valid)
        result = policy.sample_generated_latents(
            observed, context, context_valid, observed_valid, pixel_valid=pixel_valid,
            seed=7, steps=2, return_features=return_features)
    assert groups == [{"O", "R", "M", "G"}] * 2
    assert len(result) == (3 if return_features else 2)
    assert (policy.teacher is not None) == (weight > 0)
    if return_features:
        assert set(result[2]) == set(policy.config.teacher_layers)


@pytest.mark.parametrize("queries,weight", [(False, 1.0), (True, 0.0), (True, 1.0)])
@pytest.mark.parametrize("online_text", [False, True])
def test_factory_only_loads_endpoint_targets_for_supervision(tmp_path, monkeypatch, queries, weight, online_text):
    source, seen = make_data(tmp_path, monkeypatch)
    if not online_text:
        from test_wsa1_5_text_embeddings import precompute

        precompute(tmp_path)
    monkeypatch.setattr("lerobot.datasets.factory.LeRobotDataset", lambda *args, **kwargs: source.base)
    cfg = TrainPipelineConfig(
        policy=WSA1_5Config(device="cpu", enable_3d_queries=queries, lambda_3d=weight,
                            load_text_encoder=online_text, max_text_length=4, text_dim=8),
        dataset=WSA1_5DatasetConfig(repo_id="test", root=str(tmp_path), view_keys=["rgb", "", ""]),
    )
    dataset, _, _ = _build_single_dataset(cfg, "test", None, 0)
    assert (dataset.text_embeddings is None) == online_text
    sample = dataset[0]
    supervised = queries and weight > 0
    assert dataset.include_3d_targets == supervised
    assert ("goal_rgb_views" in sample["targets"]) == supervised
    assert ("goal_view_valid" in sample["targets"]) == supervised
    assert len(seen[0][1]) == (34 if supervised else 33)
    batch = wsa1_5_collate([sample, dataset[1]])
    assert ("goal_rgb_views" in batch["targets"]) == supervised


def test_collate_rejects_mixed_endpoint_loading_modes(tmp_path, monkeypatch):
    source, _ = make_data(tmp_path, monkeypatch)
    without_targets = WSA1_5Dataset(source.base, source.config, include_3d_targets=False)
    with pytest.raises(ValueError, match="different 3D supervision"):
        wsa1_5_collate([source[0], without_targets[0]])


def test_disabled_rgb_has_no_valid_prediction_pixels(monkeypatch):
    policy, _ = make_policy(monkeypatch, enable_rgb=False)
    policy.config.sampler_steps = 1
    result = policy.predict_future(torch.zeros(1, 3, 96, 32), ["move"])
    assert result["future_rgb"] is None
    assert not result["future_rgb_valid"].any()
    assert result["future_motion"] is not None
