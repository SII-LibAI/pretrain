"""Unified action slots, IK sampling and matching statistics; tiny CPU models only."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from lerobot.dataset_schemas import DatasetSchema, get_registry, get_schema
from lerobot.datasets.retarget_actions import RetargetActionSource
from lerobot.policies.wsa1_5.dataset_wsa1_5 import WSA1_5Dataset, action_prompt, wsa1_5_collate
from test_wsa1_5_action import action_config, action_policy
from test_wsa1_5_dataset import FrameTable, make_data
from util_scripts.compute_synthetic_embodiment_stats import compute_embodiment_stats, write_embodiment_stats
from util_scripts.precompute_wsa1_5_text_embeddings import dataset_prompts


def write_retarget(root, bodies=("piper", "ur7e_wuji"), *, split_episode=False, eef=True, robot_type="egodex_v"):
    sizes = {"piper": 16, "piper_inspire": 40, "ur7e_wuji": 52, "franka": 18}
    frames = np.arange(40)
    episode = frames // 20 if split_episode else np.zeros(40, dtype=int)
    payload = {"index": frames, "episode_index": episode,
               "frame_index": frames % 20 if split_episode else frames}
    for name in bodies:
        payload[f"action.{name}"] = (frames[:, None] * 0.1 + np.arange(sizes[name])[None] * 0.01).tolist()
        valid = np.ones(40, dtype=bool)
        if name == "piper":
            valid[10] = False
        payload[f"action.{name}_valid"] = valid
    if eef:
        for side, origin in (("right", 100), ("left", 200)):
            payload[f"human_eef.{side}.pos"] = (origin + frames[:, None] + np.arange(3)).tolist()
            payload[f"human_eef.{side}.rot"] = [[0., 0., 0., 1.]] * 40
            payload[f"human_eef.{side}.grasp"] = [0.99] * 40
            payload[f"human_eef.{side}.valid"] = [True] * 40
    directory = root / "meta/annotations"
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(payload), directory / "action.parquet", row_group_size=9)
    (root / "meta/info.json").write_text(json.dumps({"robot_type": robot_type}), encoding="utf-8")


def mid_config(**kwargs):
    return action_config(action_space="unified72", state_dim=72, action_dim=72, load_text_encoder=True, **kwargs)


def retarget_dataset(root, monkeypatch, *, mode="delta", stride=1, bodies=("piper", "ur7e_wuji"), robot_type="egodex_v"):
    fixture, _ = make_data(root, monkeypatch, future_frames=4)
    fixture.base.hf_dataset["index"] = list(range(40))
    fixture.base.meta.robot_type = robot_type
    write_retarget(root, bodies, robot_type=robot_type)
    stats, _ = compute_embodiment_stats([root], chunk_size=4, stride=stride, action_mode=mode, action_space="unified72")
    write_embodiment_stats(root / "norm", stats)
    cfg = replace(fixture.config, use_subtask=False, allow_endframe_3d=True, action_mode=mode,
                  rgb_height=96, rgb_width=64, motion_height=32, motion_width=32,
                  retarget_stats_dir=str(root / "norm"))
    return WSA1_5Dataset(fixture.base, cfg, action_config=mid_config(action_mode=mode, action_frame_stride=stride))


@pytest.mark.parametrize("name", ["egodex_synthetic_piper", "egodex_synthetic_piper_inspire",
                                  "egoverse_synthetic_ur7e_wuji", "aloha", "aloha_robotwin", "piper"])
def test_unified_schema_roundtrip_and_legacy_preservation(name):
    original = get_schema(name)
    restored = DatasetSchema.from_config(original.to_config())
    assert restored == original
    unified = restored.for_action_space("unified72")
    assert unified.action_reorder == original.unified_action_reorder
    assert get_schema(name).action_reorder == original.action_reorder
    used = [i for _, _, first, last in unified.action_reorder for i in range(first, last)]
    assert len(used) == len(set(used)) and min(used) >= 0 and max(used) < 72


@pytest.mark.parametrize("mode", ["abs", "delta"])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("robot_type,prefix", [("egodex_v", "egodex_synthetic_"),
                                               ("egoverse_human_bimanual", "egoverse_synthetic_")])
def test_training_and_stats_share_windows_and_normalization(tmp_path, monkeypatch, mode, stride, robot_type, prefix):
    dataset = retarget_dataset(tmp_path, monkeypatch, mode=mode, stride=stride, robot_type=robot_type)
    source = dataset.retarget_source
    assert source.names == (f"{prefix}piper", f"{prefix}ur7e_wuji") and all(source.with_eef)
    seen = set()
    torch.manual_seed(17)
    for _ in range(30):
        sample = dataset[0]
        body = source.names.index(sample["embodiment"])
        seen.add(sample["embodiment"])
        row = dataset.annotation_rows[dataset.sample_indices[0]]
        state, future = source.read_window(body, row)
        codec = dataset.retarget_codecs[body]
        physical = codec.decode(sample["targets"]["action"], {"observation.state": state})
        expected = torch.zeros(4, 72)
        for lo, hi, first, last in codec.schema.action_reorder:
            expected[:, first:last] = torch.from_numpy(future[:, lo:hi])
        torch.testing.assert_close(physical, expected, atol=2e-5, rtol=1e-5)
        mask = sample["targets"]["action_valid_mask"]
        assert not sample["targets"]["action"][~mask].any()
        assert f"Robot Embodiment: {sample['embodiment']}" in sample["condition_text"]
        assert f"Action Mode: {mode}" in sample["condition_text"]
        # IK gripper labels are kept; the shared EEF grasp=0.99 must not replace them.
        if sample["embodiment"] == f"{prefix}piper":
            torch.testing.assert_close(physical[:, 14:16], torch.from_numpy(future[:, 14:16]))
        np.testing.assert_array_equal(state[-14:-11], [100 + row, 101 + row, 102 + row])
        np.testing.assert_array_equal(state[-7:-4], [200 + row, 201 + row, 202 + row])
    assert seen == {f"{prefix}piper", f"{prefix}ur7e_wuji"}
    for body in source.active_bodies:
        anchors = np.flatnonzero(source.candidate_masks & (np.uint64(1) << np.uint64(body)))
        vectors = [source.read_window(body, int(row)) for row in anchors]
        state_expected = np.stack([state for state, _ in vectors])
        action_expected = np.concatenate([future - state * source.schemas[body].action_mask.numpy()
                                          if mode == "delta" else future for state, future in vectors])
        stats = source.stats[source.names[body]]
        np.testing.assert_allclose(stats["observation.state"]["mean"], state_expected.mean(0), atol=3e-5)
        np.testing.assert_allclose(stats["action"]["mean"], action_expected.mean(0), atol=3e-5)


def test_inspire_hand_slots_exclude_old_gripper_and_reorder_fingers(tmp_path, monkeypatch):
    dataset = retarget_dataset(tmp_path, monkeypatch, mode="abs", bodies=("piper_inspire",))
    sample = dataset[0]
    codec = dataset.retarget_codecs[0]
    state, future = dataset.retarget_source.read_window(0, 0)
    physical = codec.decode(sample["targets"]["action"], {"observation.state": state})
    assert not sample["targets"]["action_valid_mask"][:, 14:18].any()
    torch.testing.assert_close(physical[:, 18:22], torch.from_numpy(future[:, 36:40]))
    torch.testing.assert_close(physical[:, 30], torch.from_numpy(future[:, 34]))  # left ring before pinky
    torch.testing.assert_close(physical[:, 34], torch.from_numpy(future[:, 32]))
    assert sample["targets"]["action_valid_mask"][0].sum() == 50  # arms12 + hands24 + EEF14


def test_stats_union_handles_source_without_dexterous_columns(tmp_path):
    human, umi = tmp_path / "human", tmp_path / "umi"
    write_retarget(human, ("piper", "ur7e_wuji"))
    write_retarget(umi, ("piper",))
    stats, specs = compute_embodiment_stats([human, umi], chunk_size=4, action_space="unified72",
                                           embodiments=["piper", "ur7e_wuji"])
    assert set(stats) == {spec.name for spec in specs} == {"egodex_synthetic_piper", "egodex_synthetic_ur7e_wuji"}
    assert stats["egodex_synthetic_piper"]["observation.state"]["count"] > stats["egodex_synthetic_ur7e_wuji"]["observation.state"]["count"]
    selected, _ = compute_embodiment_stats([human, umi], chunk_size=4, action_space="unified72", embodiments=["piper"])
    assert set(selected) == {"egodex_synthetic_piper"}


@pytest.mark.parametrize("workers", [1, 2])
def test_stats_merge_only_matching_full_schema_names(tmp_path, workers):
    roots = [tmp_path / name for name in ("egodex_a", "egodex_b", "egoverse")]
    for root, robot_type, offset in zip(roots, ("egodex_v", "egodex_v", "egoverse_human_bimanual"),
                                        (0, 20, 100), strict=True):
        write_retarget(root, ("piper",), robot_type=robot_type)
        path = root / "meta/annotations/action.parquet"
        data = pq.read_table(path).to_pydict()
        data["action.piper"] = (np.asarray(data["action.piper"]) + offset).tolist()
        pq.write_table(pa.table(data), path)
    stats, specs = compute_embodiment_stats(roots, chunk_size=4, action_space="unified72", num_workers=workers)
    assert set(stats) == {spec.name for spec in specs} == {"egodex_synthetic_piper", "egoverse_synthetic_piper"}
    anchors = [t for t in range(36) if 10 not in range(t, t + 5)]
    for name, count, offset in (("egodex_synthetic_piper", 2, 10), ("egoverse_synthetic_piper", 1, 100)):
        state = stats[name]["observation.state"]
        assert state["count"] == count * len(anchors)
        assert state["mean"][0] == pytest.approx(np.mean(anchors) * 0.1 + offset)
    write_embodiment_stats(tmp_path / "norm", stats)
    for root, expected in zip(roots, ("egodex_synthetic_piper", "egodex_synthetic_piper", "egoverse_synthetic_piper"), strict=True):
        source = RetargetActionSource(root, 4, stats_dir=tmp_path / "norm")
        assert source.names == (expected,) and set(source.stats) == {expected}


def test_source_specific_reorder_reaches_training(tmp_path, monkeypatch):
    name = "egoverse_synthetic_piper"
    mapping = [[0, 6, 58, 64], [6, 8, 14, 16], [8, 14, 65, 71], [14, 16, 16, 18]]
    schema = replace(get_schema(name), unified_action_reorder=mapping, unified_state_reorder=mapping)
    monkeypatch.setitem(get_registry()._schemas, name, schema)
    dataset = retarget_dataset(tmp_path, monkeypatch, mode="abs", bodies=("piper",), robot_type="egoverse_human_bimanual")
    sample = dataset[0]
    assert sample["embodiment"] == name and f"Robot Embodiment: {name}" in sample["condition_text"]
    state, future = dataset.retarget_source.read_window(0, 0)
    physical = dataset.retarget_codecs[0].decode(sample["targets"]["action"], {"observation.state": state})
    torch.testing.assert_close(physical[:, 58:64], torch.from_numpy(future[:, :6]))
    torch.testing.assert_close(physical[:, 65:71], torch.from_numpy(future[:, 8:14]))


def test_full_schema_whitelist_and_unknown_annotation_name(tmp_path):
    write_retarget(tmp_path, ("piper", "franka"))
    source = RetargetActionSource(tmp_path, 4, embodiments=["egodex_synthetic_franka", "egoverse_synthetic_piper"])
    assert source.names == ("egodex_synthetic_franka",)
    assert source.names == RetargetActionSource(tmp_path, 4, embodiments=["franka"]).names
    path = tmp_path / "meta/annotations/action.parquet"
    table = pq.read_table(path)
    table = table.rename_columns([name.replace("action.piper", "action.another_robot") for name in table.column_names])
    pq.write_table(table, path)
    with pytest.raises(KeyError, match="No dataset schema registered.*another_robot"):
        RetargetActionSource(tmp_path, 4)


def test_validity_episode_boundaries_and_filtered_row_alignment(tmp_path):
    write_retarget(tmp_path, ("piper", "ur7e_wuji"), split_episode=True)
    path = tmp_path / "meta/annotations/action.parquet"
    rows = pq.read_table(path).to_pydict()
    rows["action.piper"][10] = [float("nan")] * 16
    pq.write_table(pa.table(rows), path, row_group_size=9)
    source = RetargetActionSource(tmp_path, 4, stride=2)
    assert source.candidates(2) == [1]  # piper's sampled future includes invalid frame 10
    assert source.candidates(3) == [0, 1]  # invalid unsampled frame 10 does not reject this window
    assert np.isfinite(source.read_window(0, 3)[1]).all()
    assert source.candidates(13) == []  # cannot cross episode boundary
    base = type("Base", (), {"hf_dataset": FrameTable(index=list(range(20, 40)),
                episode_index=[1] * 20, frame_index=list(range(20)))})()
    np.testing.assert_array_equal(source.rows_for_base(base), np.arange(20, 40))
    base.hf_dataset["frame_index"][0] = 1
    with pytest.raises(ValueError, match="do not align"):
        source.rows_for_base(base)


def test_retarget_rejects_offline_cache_and_stale_statistics(tmp_path, monkeypatch):
    dataset = retarget_dataset(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="load_text_encoder=true"):
        WSA1_5Dataset(dataset.base, dataset.config, action_config=replace(dataset.action_config, load_text_encoder=False))
    with pytest.raises(ValueError, match="precompute only standard"):
        dataset_prompts(tmp_path, enable_action=True, action_space="unified72")
    with pytest.raises(ValueError, match="window/representation"):
        RetargetActionSource(tmp_path, 4, stride=2, stats_dir=tmp_path / "norm")
    with pytest.raises(ValueError, match="select IK embodiment names"):
        RetargetActionSource(tmp_path, 4, embodiments=["human_eef"])


def test_native_and_retarget_mixed_batch_backward(tmp_path, monkeypatch):
    from lerobot.datasets.transformed_dataset import MultiLeRobotDataset

    retarget = retarget_dataset(tmp_path / "retarget", monkeypatch)
    fixture, _ = make_data(tmp_path / "native", monkeypatch, future_frames=4)
    fixture.base.meta.robot_type = "aloha"
    for key in ("observation.state", "action"):
        fixture.base.meta.features[key] = {"dtype": "float32", "shape": [14]}
        fixture.base.meta.stats[key] = {"mean": [0.] * 14, "std": [1.] * 14}
        fixture.base.hf_dataset[key] = [[frame * 0.01] * 14 for frame in range(40)]
    native = WSA1_5Dataset(fixture.base, retarget.config, action_config=retarget.action_config)
    sample = native[2]
    assert sample["embodiment"] == "aloha" and sample["targets"]["action_valid_mask"][0].sum() == 14
    assert sample["condition_text"] == action_prompt("knit", "joint", "delta", embodiment="aloha")
    mixed = MultiLeRobotDataset([native, retarget], dataset_weights=[0.5, 0.5])
    loader = torch.utils.data.DataLoader(mixed, batch_size=2, sampler=[2, len(native) + 2],
                                        collate_fn=wsa1_5_collate)
    batch = next(iter(loader))
    policy = action_policy(action_space="unified72", state_dim=72, action_dim=72,
                           load_text_encoder=True, num_history_frame=4)
    monkeypatch.setattr(policy, "_encode_text", lambda texts: (torch.zeros(len(texts), 4, 8),
                                                             torch.ones(len(texts), 4, dtype=torch.bool)))
    loss, metrics = policy.training_forward(batch)
    assert torch.isfinite(loss) and metrics["loss_action_flow"] > 0
    loss.backward()
    assert policy.action_expert.action_encoder.weight.grad.abs().sum() > 0
    assert policy.proprio_encoder.weight.grad.abs().sum() > 0


def test_all_registered_unified_maps_have_unique_destinations():
    for name in get_registry().list_available():
        schema = get_schema(name)
        if schema.unified_action_reorder is None:
            continue
        for mapping in (schema.unified_action_reorder, schema.unified_state_reorder):
            destinations = [dim for _, _, lo, hi in mapping for dim in range(lo, hi)]
            assert len(destinations) == len(set(destinations)), name
            assert all(0 <= dim < 72 for dim in destinations), name


@pytest.mark.parametrize("mode", ["abs", "delta"])
@pytest.mark.parametrize("stride", [1, 2])
def test_native_statistics_match_masked_training_windows(monkeypatch, mode, stride):
    from util_scripts import compute_norm_stats_multi as script

    state = torch.arange(5 * 14).reshape(5, 14).float() / 100
    actions = state * 2 + 0.2
    class Columns(dict):
        def select(self, indices):
            return {key: list(value[indices]) for key, value in self.items()}
    base = SimpleNamespace(meta=SimpleNamespace(robot_type="aloha", video_keys=[], image_keys=[],
        features={key: {"shape": [14]} for key in ("observation.state", "action")},
        episodes={"dataset_from_index": [0, 2], "dataset_to_index": [2, 5]}), num_episodes=2,
        hf_dataset=Columns({"observation.state": state, "action": actions}))
    monkeypatch.setattr(script, "LeRobotDataset", lambda _: base)
    result = script._compute_one_repo("native", mode, 4, stride)
    expected = []
    for first, stop in ((0, 2), (2, 5)):
        for anchor in range(first, stop):
            for offset in range(0, 4 * stride, stride):
                if anchor + offset < stop:
                    target = actions[anchor + offset].clone()
                    if mode == "delta":
                        target -= state[anchor] * get_schema("aloha").action_mask
                    expected.append(target)
    payload = result["payload"]
    assert payload["action"]["count"] == len(expected)
    np.testing.assert_allclose(payload["action"]["mean"], torch.stack(expected).mean(0), atol=1e-6)
    np.testing.assert_allclose(payload["observation.state"]["mean"], state.mean(0), atol=1e-6)
    assert payload["observation.state"]["count"] == 5 and result["skipped_episodes"] == 0


def test_native_unified_cache_and_robotwin_physical_action(tmp_path, monkeypatch):
    from evaluation.RoboTwin.wsa1_5.policy import prepare_action
    from test_wsa1_5_text_embeddings import TinyEncoder, tokenize
    from util_scripts.precompute_wsa1_5_text_embeddings import precompute_dataset

    fixture, _ = make_data(tmp_path, monkeypatch, future_frames=4)
    fixture.base.meta.robot_type = "aloha"
    (tmp_path / "meta/info.json").write_text(json.dumps({"robot_type": "aloha"}), encoding="utf-8")
    for key in ("observation.state", "action"):
        fixture.base.meta.features[key] = {"shape": [14], "dtype": "float32"}
        fixture.base.meta.stats[key] = {"mean": [0.] * 14, "std": [1.] * 14}
        fixture.base.hf_dataset[key] = [[i * 0.01] * 14 for i in range(40)]
    precompute_dataset(tmp_path, TinyEncoder(), tokenize, "cpu", max_length=4, text_dim=8,
                       enable_action=True, action_space="unified72")
    dataset = WSA1_5Dataset(fixture.base, replace(fixture.config, use_subtask=False, action_mode="delta"),
                           action_config=replace(mid_config(), load_text_encoder=False), text_embedding_shape=(4, 8))
    sample = dataset[2]
    assert sample["condition_text"] in dataset_prompts(tmp_path, enable_action=True, action_space="unified72")
    assert sample["text_embedding"].shape == (4, 8)
    physical = dataset.action_codec.decode(sample["targets"]["action"],
                                            {"observation.state": torch.full((14,), 0.02)})
    raw = prepare_action(physical[0], dataset.action_codec)
    np.testing.assert_allclose(raw, np.full(14, 0.02), atol=1e-6)


@pytest.mark.parametrize("source_mode,target_mode", [("full", "full"), ("lora", "full"), ("full", "lora"), ("lora", "lora")])
def test_midtrain_to_cached_posttrain_preserves_trained_action(tmp_path, monkeypatch, source_mode, target_mode):
    from copy import deepcopy
    from lerobot.policies.wsa1_5.initialization import LoRALinear
    from lerobot.policies.wsa1_5.modeling_wsa1_5 import WSA1_5_Policy, WorldComponents

    source = action_policy(action_space="unified72", state_dim=72, action_dim=72,
                           load_text_encoder=True, rgb_finetune_mode=source_mode, rgb_lora_rank=2)
    source.text_encoder = torch.nn.Linear(8, 8).requires_grad_(False)
    with torch.no_grad():
        source.action_expert.action_encoder.weight.fill_(0.123)
        for layer in source.flow.rgb.modules():
            if isinstance(layer, LoRALinear):
                layer.lora_b.fill_(0.025)
    source.eval().save_pretrained(tmp_path)
    template = action_policy(enable_action=False, lambda_rgb=1)
    monkeypatch.setattr(WSA1_5_Policy, "_build_components", lambda self, init: deepcopy(
        WorldComponents(template.flow, template.future_3d, template.vae)))
    monkeypatch.setattr("lerobot.policies.wsa1_5.modeling_wsa1_5.initialize_action_from_rgb",
                        lambda *args: pytest.fail("Trained Action must not be reinitialized"))
    config = replace(source.config, init_checkpoint_path=str(tmp_path), load_text_encoder=False,
                     rgb_finetune_mode=target_mode)
    target = WSA1_5_Policy(config).eval()
    assert target.text_encoder is None
    for key, value in source.state_dict().items():
        if key.startswith(("action_expert.", "proprio_encoder.")):
            torch.testing.assert_close(target.state_dict()[key], value, atol=0, rtol=0)
    probe = torch.randn(2, 16)
    torch.testing.assert_close(target.flow.rgb.blocks[0].self_attn.q(probe),
                               source.flow.rgb.blocks[0].self_attn.q(probe))
    with pytest.raises(ValueError, match="layout differs"):
        WSA1_5_Policy(replace(config, action_space="legacy", state_dim=14, action_dim=14))


def test_factory_retarget_skips_native_stats_override(tmp_path, monkeypatch):
    from lerobot.datasets import factory

    dataset = retarget_dataset(tmp_path, monkeypatch)
    config = SimpleNamespace(dataset=dataset.config, policy=dataset.action_config)
    config.dataset = replace(config.dataset, root=str(tmp_path.parent))
    monkeypatch.setattr(factory, "LeRobotDataset", lambda *a, **kw: dataset.base)
    monkeypatch.setattr(factory, "_apply_dataset_stats_overrides",
                        lambda *a: pytest.fail("Native human skeleton statistics must not be used"))
    built, stats, robot = factory._build_single_dataset(config, tmp_path.name, None, 0)
    assert robot == "egodex_v" and set(stats) == {"egodex_synthetic_piper", "egodex_synthetic_ur7e_wuji"}
    assert built[0]["embodiment"] in stats
